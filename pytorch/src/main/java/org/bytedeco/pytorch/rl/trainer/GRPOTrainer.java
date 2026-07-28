package org.bytedeco.pytorch.rl.trainer;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.trl.loss.GRPOLoss;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;

/**
 * Classic <b>Group-Relative</b> Policy Optimization trainer (DeepSeek-R1 style).
 *
 * <p>Reuses shared {@link org.bytedeco.pytorch.llm.trl.loss.GRPOLoss}. This is
 * <em>not</em> the guided-reward agent
 * ({@link org.bytedeco.pytorch.rl.agent.GuidedRewardPPOAgent} /
 * {@link org.bytedeco.pytorch.rl.agent.GRPOAgent}).
 *
 * <p>For full LLM GRPO prefer {@link org.bytedeco.pytorch.llm.trl.GRPOTrainer}
 * or {@link org.bytedeco.pytorch.rl.agent.GroupRelativePPOAgent}.
 *
 * <h2>{@link ReplayBuffer#getAll()} layout</h2>
 * {@code [states, actions, oldLogProbs, advantages, returns]}.
 * Group scores for GRPO are taken from <b>returns</b> (index 4) — callers that
 * store raw rewards in the return slot (e.g. {@code TradingSystem}) are supported.
 * If returns are missing, advantages (index 3) are used as a fallback.
 */
public class GRPOTrainer implements RLTrainer {
    private final AbstractActorCritic model;
    private final Optimizer optimizer;
    private final double clipRange;
    private final int groupSize;

    public GRPOTrainer(AbstractActorCritic model) {
        this(model, 1e-4, 0.2, 4);
    }

    public GRPOTrainer(AbstractActorCritic model, double lr, double clipRange, int groupSize) {
        this.model = model;
        AdamOptions option = new AdamOptions();
        option.lr().put(lr);
        this.optimizer = new Adam(model.parameters(), option);
        this.clipRange = clipRange;
        this.groupSize = groupSize;
    }

    public GRPOTrainer(AbstractActorCritic model, Optimizer optimizer, double clipRange, int groupSize) {
        this.model = model;
        this.optimizer = optimizer;
        this.clipRange = clipRange;
        this.groupSize = groupSize;
    }

    /** @deprecated prefer {@link #GRPOTrainer(AbstractActorCritic)} */
    @Deprecated
    public GRPOTrainer(ActorCriticNetwork model) {
        this((AbstractActorCritic) model);
    }

    /** @deprecated prefer {@link #GRPOTrainer(AbstractActorCritic, double, double, int)} */
    @Deprecated
    public GRPOTrainer(ActorCriticNetwork model, double lr, double clipRange, int groupSize) {
        this((AbstractActorCritic) model, lr, clipRange, groupSize);
    }

    /**
     * One GRPO update given a distribution over actions and group rewards
     * shaped {@code [Batch, GroupSize]} or flat {@code [Batch*GroupSize]}.
     */
    public Tensor trainStep(Distribution dist, Tensor actions, Tensor oldLps, Tensor groupRewards) {
        Tensor flatRewards = groupRewards.dim() > 1 ? groupRewards.flatten() : groupRewards;
        int g = groupSize > 0 ? groupSize : (int) groupRewards.size(groupRewards.dim() - 1);
        if (flatRewards.numel() % g != 0) {
            g = (int) flatRewards.numel();
        }

        Tensor currLps = actionLogProb(dist, actions);
        Tensor loss;
        if (oldLps != null && oldLps.defined() && clipRange > 0.0) {
            loss = GRPOLoss.computeClipped(currLps, flat1d(oldLps), flatRewards, g, clipRange);
        } else {
            loss = GRPOLoss.compute(currLps, flatRewards, g);
        }

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        return loss.detach();
    }

    /** @deprecated use {@link #trainStep} */
    @Deprecated
    public void train_step(Distribution dist, Tensor actions, Tensor oldLps, Tensor groupRewards) {
        trainStep(dist, actions, oldLps, groupRewards);
    }

    @Override
    public void trainBatch(ReplayBuffer buffer) {
        if (buffer == null || buffer.size() == 0) {
            return;
        }
        Tensor[] data = buffer.getAll();
        if (data == null) {
            return;
        }
        // getAll: [states, actions, oldLogProbs, advantages, returns]
        Tensor states = data[0];
        Tensor actions = data[1];
        Tensor oldLps = data[2];
        Tensor advantages = data.length > 3 ? data[3] : null;
        Tensor returns = data.length > 4 ? data[4] : null;
        // Group-relative scores = returns (raw reward often stored there) > advantages
        Tensor scores = (returns != null && returns.defined()) ? returns : advantages;
        if (scores == null || !scores.defined()) {
            return;
        }

        Distribution dist = model.getDistribution(states);
        int g = groupSize;
        if (g <= 0 || scores.numel() % g != 0) {
            g = (int) scores.numel();
        }
        Tensor currLps = actionLogProb(dist, actions);
        Tensor loss = GRPOLoss.computeClipped(currLps, flat1d(oldLps), flat1d(scores), g, clipRange);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    /**
     * Per-sample action log-prob. Discrete Categorical is already {@code [N]} —
     * do <em>not</em> {@code sum(-1)} or the batch collapses to a scalar.
     * Continuous / multi-dim actions keep a trailing action axis to reduce.
     */
    static Tensor actionLogProb(Distribution dist, Tensor actions) {
        Tensor lp = dist.log_prob(actions);
        return flat1d(sumActionDimsOnly(lp));
    }

    static Tensor sumActionDimsOnly(Tensor logProb) {
        if (logProb.dim() <= 1) {
            return logProb;
        }
        return logProb.sum(-1);
    }

    static Tensor flat1d(Tensor t) {
        Tensor x = t;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) {
            x = x.squeeze(x.dim() - 1);
        }
        if (x.dim() == 0) {
            x = x.reshape(1);
        }
        if (x.dim() > 1) {
            x = x.reshape(-1);
        }
        return x;
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public AbstractActorCritic model() {
        return model;
    }

    @Override
    public String algorithm() {
        return "grpo-group-relative";
    }
}
