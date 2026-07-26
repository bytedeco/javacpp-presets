package org.bytedeco.pytorch.rl.agent;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.trl.loss.GRPOLoss;
import org.bytedeco.pytorch.llm.trl.loss.PPOLoss;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCritic;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * <b>Group-Relative Policy Optimization</b> (DeepSeek-R1 / HF-TRL GRPO).
 *
 * <p>For each prompt, {@code G} completions are scored; advantages are
 * <em>group-normalized</em> (no learned critic required). A PPO-style clip is
 * applied on the importance sampling ratio. Loss math delegates to
 * {@link GRPOLoss} in {@code llm.trl.loss} (shared with
 * {@link org.bytedeco.pytorch.llm.trl.GRPOTrainer}).
 *
 * <p><b>Not</b> the guided-reward agent — that is {@link GuidedRewardPPOAgent}
 * / {@link GRPOAgent}.
 *
 * <h2>Batch layout</h2>
 * Flat tensors of length {@code B = prompts × groupSize}:
 * states/actions/oldLogProbs/rewards all length {@code B}, with groups laid out
 * contiguously (prompt0's G completions, then prompt1, …).
 */
public class GroupRelativePPOAgent extends AbstractRLAgent {
    private final float clipEps;
    private final int groupSize;
    private final float entropyCoeff;
    private final float maxGradNorm;
    private final int epochs;

    public GroupRelativePPOAgent(AbstractActorCritic model,
                                 Optimizer optimizer,
                                 ReplayBuffer replayBuffer,
                                 float clipEps,
                                 int groupSize,
                                 float entropyCoeff,
                                 float maxGradNorm,
                                 int epochs) {
        super(model, optimizer, replayBuffer);
        this.clipEps = clipEps;
        this.groupSize = Math.max(2, groupSize);
        this.entropyCoeff = entropyCoeff;
        this.maxGradNorm = maxGradNorm;
        this.epochs = Math.max(1, epochs);
    }

    public GroupRelativePPOAgent(long stateDim, long actionDim,
                                 AbstractActorCritic.ActionSpaceType actionSpaceType) {
        this(build(stateDim, actionDim, actionSpaceType, 3e-4f),
                0.2f, 4, 0.01f, 0.5f, 4);
    }

    public static GroupRelativePPOAgent create(long stateDim, long actionDim,
                                               AbstractActorCritic.ActionSpaceType type) {
        return new GroupRelativePPOAgent(stateDim, actionDim, type);
    }

    public static GroupRelativePPOAgent create(long stateDim, long actionDim,
                                               AbstractActorCritic.ActionSpaceType type,
                                               int groupSize, float lr) {
        Object[] built = build(stateDim, actionDim, type, lr);
        return new GroupRelativePPOAgent(
                (AbstractActorCritic) built[0], (Optimizer) built[1], new ReplayBuffer(),
                0.2f, groupSize, 0.01f, 0.5f, 4);
    }

    private GroupRelativePPOAgent(Object[] built, float clipEps, int groupSize,
                                  float entropyCoeff, float maxGradNorm, int epochs) {
        super((AbstractActorCritic) built[0], (Optimizer) built[1], new ReplayBuffer());
        this.clipEps = clipEps;
        this.groupSize = Math.max(2, groupSize);
        this.entropyCoeff = entropyCoeff;
        this.maxGradNorm = maxGradNorm;
        this.epochs = Math.max(1, epochs);
    }

    private static Object[] build(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType type, float lr) {
        AbstractActorCritic model = type == AbstractActorCritic.ActionSpaceType.CONTINUOUS
                ? new ActorCritic(stateDim, actionDim)
                : new CartPoleActorCritic(stateDim, actionDim);
        AdamOptions opt = new AdamOptions();
        opt.lr().put(lr);
        return new Object[]{model, new Adam(model.parameters(), opt)};
    }

    @Override
    public Tensor trainStep() {
        ReplayBuffer buffer = super.getReplayBuffer();
        if (buffer.size() == 0) {
            throw new IllegalStateException("GroupRelativePPOAgent buffer is empty");
        }
        Tensor states = buffer.getStates();
        Tensor actions = buffer.getActions();
        Tensor oldLogProbs = buffer.getOldLogProbs() != null
                ? buffer.getOldLogProbs() : buffer.getLogProbs();
        Tensor rewards = buffer.getRewards();
        if (rewards == null || !rewards.defined()) {
            // try env rewards
            rewards = buffer.getEnvRewards();
        }
        return update(states, actions, oldLogProbs, rewards);
    }

    /**
     * Multi-epoch GRPO update. {@code rewards} length must be divisible by
     * {@link #groupSize}.
     */
    public Tensor update(Tensor states, Tensor actions, Tensor oldLogProbs, Tensor rewards) {
        AbstractActorCritic model = super.getModel();
        Tensor flatR = rewards.reshape(-1);
        long n = flatR.numel();
        int g = groupSize;
        if (n % g != 0) {
            // fall back to one big group
            g = (int) n;
        }
        Tensor last = null;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Distribution dist = model.getDistribution(states);
            Tensor currLp = sumActionLogProb(dist.log_prob(actions)).reshape(-1);
            Tensor oldLp = oldLogProbs.reshape(-1);
            Tensor loss = GRPOLoss.computeClipped(currLp, oldLp, flatR, g, clipEps);
            // optional entropy bonus
            Tensor ent = dist.entropy().mean();
            Tensor total = loss.sub(ent.mul(new Scalar(entropyCoeff)));

            super.optimizer.zero_grad();
            total.backward();
            clip_grad_norm_(model.parameters(), maxGradNorm);
            super.optimizer.step();
            last = total.detach();
        }
        return last;
    }

    /**
     * Convenience: group-relative loss only (no optimizer step) — for tests.
     */
    public static Tensor computeGroupRelativeLoss(Distribution dist, Tensor actions,
                                                  Tensor oldLogProbs, Tensor groupRewards,
                                                  int groupSize, float clipEps) {
        Tensor curr = sumActionLogProb(dist.log_prob(actions)).reshape(-1);
        Tensor old = oldLogProbs.reshape(-1);
        Tensor flatR = groupRewards.reshape(-1);
        int g = groupSize;
        if (flatR.numel() % g != 0) g = (int) flatR.numel();
        return GRPOLoss.computeClipped(curr, old, flatR, g, clipEps);
    }

    @Override
    public Tensor[] sample(Tensor state) {
        AbstractActorCritic model = super.getModel();
        model.train(true);
        Distribution dist = model.getDistribution(state);
        Tensor action = dist.sample();
        Tensor logProb = sumActionLogProb(dist.log_prob(action));
        Tensor value = model.getValue(state); // optional; GRPO is critic-free
        return new Tensor[]{
                action.detach().clone(),
                logProb.detach().clone(),
                value.detach().clone()
        };
    }

    private static Tensor sumActionLogProb(Tensor logProb) {
        if (logProb.dim() <= 1) return logProb;
        return logProb.sum(-1);
    }

    public float getClipEps() { return clipEps; }
    public int getGroupSize() { return groupSize; }
    public float getEntropyCoeff() { return entropyCoeff; }
    public int getEpochs() { return epochs; }

    @Override
    public void close() {
        super.close();
    }
}
