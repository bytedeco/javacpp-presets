package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.GAE;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.ActorCritic;
import org.bytedeco.pytorch.rl.critic.CartPoleActorCritic;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Proximal Policy Optimization (clipped surrogate + value + entropy).
 *
 * <p>Supports discrete ({@link CartPoleActorCritic}) and continuous
 * ({@link ActorCritic}) action spaces. Prefer the full constructor or
 * {@link #create} so the optimizer is bound to the live model parameters.
 */
public class PPOAgent extends AbstractRLAgent {
    private final float clipEps;
    private final float gamma;
    private final float gaeLambda;
    private final float entropyCoeff;
    private final float valueCoeff;
    private final float maxGradNorm;
    /** Number of PPO epochs over the same rollout (default 4). */
    private final int ppoEpochs;
    /** Mini-batch size within an epoch; 0 = full batch. */
    private final int miniBatchSize;
    /** Running observation normalizer (optional; enabled by default for classic control). */
    private final RunningMeanStd obsNormalizer;
    private boolean normalizeObs;

    public PPOAgent(AbstractActorCritic model,
                    Optimizer optimizer,
                    ReplayBuffer replayBuffer,
                    float clipEps,
                    float gamma,
                    float gaeLambda,
                    float entropyCoeff,
                    float valueCoeff,
                    float maxGradNorm) {
        this(model, optimizer, replayBuffer, clipEps, gamma, gaeLambda, entropyCoeff,
                valueCoeff, maxGradNorm, 4, 0, true);
    }

    public PPOAgent(AbstractActorCritic model,
                    Optimizer optimizer,
                    ReplayBuffer replayBuffer,
                    float clipEps,
                    float gamma,
                    float gaeLambda,
                    float entropyCoeff,
                    float valueCoeff,
                    float maxGradNorm,
                    int ppoEpochs,
                    int miniBatchSize,
                    boolean normalizeObs) {
        super(model, optimizer, replayBuffer);
        this.clipEps = clipEps;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.entropyCoeff = entropyCoeff;
        this.valueCoeff = valueCoeff;
        this.maxGradNorm = maxGradNorm;
        this.ppoEpochs = Math.max(1, ppoEpochs);
        this.miniBatchSize = Math.max(0, miniBatchSize);
        this.normalizeObs = normalizeObs;
        this.obsNormalizer = new RunningMeanStd(model != null ? guessStateDim(model) : 4);
    }

    public PPOAgent(AbstractActorCritic model, Optimizer optimizer, ReplayBuffer replayBuffer) {
        this(model, optimizer, replayBuffer, 0.2f, 0.99f, 0.95f, 0.01f, 0.5f, 0.5f);
    }

    /** Discrete or continuous agent with default PPO hyper-parameters. */
    public PPOAgent(long stateDim, long actionDim, AbstractActorCritic.ActionSpaceType actionSpaceType) {
        this(build(stateDim, actionDim, actionSpaceType, 3e-4f),
                0.2f, 0.99f, 0.95f, 0.01f, 0.5f, 0.5f, 4, 64, true);
    }

    public static PPOAgent create(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType) {
        return new PPOAgent(stateDim, actionDim, actionSpaceType);
    }

    public static PPOAgent create(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType,
                                  float lr, float clipEps, float gamma, float gaeLambda,
                                  float entropyCoeff, float valueCoeff, float maxGradNorm) {
        return new PPOAgent(build(stateDim, actionDim, actionSpaceType, lr),
                clipEps, gamma, gaeLambda, entropyCoeff, valueCoeff, maxGradNorm, 4, 64, true);
    }

    /** Full factory including multi-epoch / mini-batch / obs-norm knobs. */
    public static PPOAgent create(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType,
                                  float lr, float clipEps, float gamma, float gaeLambda,
                                  float entropyCoeff, float valueCoeff, float maxGradNorm,
                                  int ppoEpochs, int miniBatchSize, boolean normalizeObs) {
        return new PPOAgent(build(stateDim, actionDim, actionSpaceType, lr),
                clipEps, gamma, gaeLambda, entropyCoeff, valueCoeff, maxGradNorm,
                ppoEpochs, miniBatchSize, normalizeObs);
    }

    private PPOAgent(Object[] built, float clipEps, float gamma, float gaeLambda,
                     float entropyCoeff, float valueCoeff, float maxGradNorm) {
        this(built, clipEps, gamma, gaeLambda, entropyCoeff, valueCoeff, maxGradNorm, 4, 64, true);
    }

    private PPOAgent(Object[] built, float clipEps, float gamma, float gaeLambda,
                     float entropyCoeff, float valueCoeff, float maxGradNorm,
                     int ppoEpochs, int miniBatchSize, boolean normalizeObs) {
        super((AbstractActorCritic) built[0], (Optimizer) built[1], new ReplayBuffer());
        this.clipEps = clipEps;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.entropyCoeff = entropyCoeff;
        this.valueCoeff = valueCoeff;
        this.maxGradNorm = maxGradNorm;
        this.ppoEpochs = Math.max(1, ppoEpochs);
        this.miniBatchSize = Math.max(0, miniBatchSize);
        this.normalizeObs = normalizeObs;
        this.obsNormalizer = new RunningMeanStd(guessStateDim((AbstractActorCritic) built[0]));
    }

    private static long guessStateDim(AbstractActorCritic model) {
        try {
            return model.getStateDim();
        } catch (Throwable t) {
            return 4;
        }
    }

    private static Object[] build(long stateDim, long actionDim,
                                  AbstractActorCritic.ActionSpaceType actionSpaceType, float lr) {
        AbstractActorCritic model = actionSpaceType == AbstractActorCritic.ActionSpaceType.CONTINUOUS
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
            throw new IllegalStateException("PPO replay buffer is empty");
        }
        // Prefer precomputed advantages/returns (push / build); otherwise require caller to update(...)
        Tensor advantages = buffer.getAdvantages();
        Tensor returns = buffer.getReturns();
        if (advantages == null || returns == null) {
            throw new IllegalStateException(
                    "PPO trainStep() needs precomputed advantages/returns. "
                            + "Call buffer.build(...) or buffer.push(...), or use update(...)");
        }
        return update(buffer.getStates(), buffer.getActions(), buffer.getLogProbs(), returns, advantages);
    }

    /**
     * Multi-epoch PPO update on a batch already stacked as tensors.
     * Runs {@link #ppoEpochs} passes; each pass optionally splits into mini-batches.
     *
     * @return detached total loss from the last mini-batch step
     */
    public Tensor update(Tensor states, Tensor actions, Tensor oldLogProbs,
                         Tensor returns, Tensor advantages) {
        Tensor acts = flattenBatch(actions);
        Tensor oldLp = flattenBatch(oldLogProbs);
        Tensor adv = flattenBatch(advantages);
        Tensor ret = flattenBatch(returns);
        // Normalize advantages once for the whole rollout (stable across epochs)
        Tensor advNormFull = normalizeAdvantages(adv);

        long n = advNormFull.numel();
        int mb = miniBatchSize > 0 ? miniBatchSize : (int) n;
        if (mb > n) mb = (int) n;
        if (mb < 1) mb = 1;

        Tensor lastLoss = null;
        for (int epoch = 0; epoch < ppoEpochs; epoch++) {
            // Shuffle indices each epoch
            Tensor perm = randperm(n);
            for (long start = 0; start < n; start += mb) {
                long end = Math.min(start + mb, n);
                long len = end - start;
                Tensor idx = perm.narrow(0, start, len);

                Tensor sMb = indexSelectStates(states, idx);
                Tensor aMb = acts.index_select(0, idx);
                Tensor oldMb = oldLp.index_select(0, idx);
                Tensor advMb = advNormFull.index_select(0, idx);
                Tensor retMb = ret.index_select(0, idx);

                lastLoss = singleUpdate(sMb, aMb, oldMb, retMb, advMb);
            }
        }
        return lastLoss;
    }

    /** One gradient step on a (mini)batch — used by multi-epoch {@link #update}. */
    private Tensor singleUpdate(Tensor states, Tensor actions, Tensor oldLogProbs,
                                Tensor returns, Tensor advNorm) {
        AbstractActorCritic m = super.getModel();
        Distribution dist = m.getDistribution(states);
        Tensor currentLogProbs = flattenBatch(sumActionLogProb(dist.log_prob(actions)));
        Tensor entropy = dist.entropy().mean();

        Tensor ratio = exp(currentLogProbs.sub(oldLogProbs));
        Tensor surr1 = ratio.mul(advNorm);
        Tensor surr2 = clamp(ratio,
                new ScalarOptional(new Scalar(1.0 - clipEps)),
                new ScalarOptional(new Scalar(1.0 + clipEps))).mul(advNorm);
        Tensor actorLoss = min(surr1, surr2).mean().neg();

        Tensor values = flattenBatch(m.getValue(states));
        Tensor criticLoss = mse_loss(values, returns);

        Tensor totalLoss = actorLoss
                .add(criticLoss.mul(new Scalar(valueCoeff)))
                .sub(entropy.mul(new Scalar(entropyCoeff)));

        super.optimizer.zero_grad();
        totalLoss.backward();
        clip_grad_norm_(m.parameters(), maxGradNorm);
        super.optimizer.step();
        return totalLoss.detach();
    }

    /** Index-select along the leading batch dim of states ([T,…] or [T]). */
    private static Tensor indexSelectStates(Tensor states, Tensor idx) {
        if (states.dim() == 0) return states;
        return states.index_select(0, idx);
    }

    /** Mean/std normalize; if variance ~0 or numel&lt;2, just center (or pass through). */
    static Tensor normalizeAdvantages(Tensor advantages) {
        Tensor flat = flattenBatch(advantages);
        if (flat.numel() < 2) {
            return flat.sub(flat.mean());
        }
        Tensor mean = flat.mean();
        // population-style variance to avoid df<=0 warnings on tiny batches
        Tensor var = flat.sub(mean).pow(new Scalar(2.0)).mean();
        Tensor std = var.sqrt().add(new Scalar(1e-8));
        return flat.sub(mean).div(std);
    }

    /** Collapse trailing size-1 dims and ensure at least 1D: [T,1]→[T], 0D→[1]. */
    static Tensor flattenBatch(Tensor t) {
        Tensor x = t;
        while (x.dim() > 1 && x.size(x.dim() - 1) == 1) {
            x = x.squeeze(x.dim() - 1);
        }
        if (x.dim() == 0) {
            x = x.reshape(1);
        }
        return x;
    }

    @Override
    public Tensor[] sample(Tensor state) {
        AbstractActorCritic m = super.getModel();
        m.train(true);
        Tensor st = maybeNormalizeObs(state, /*updateStats*/true);
        Distribution dist = m.getDistribution(st);
        Tensor action = dist.sample();
        Tensor logProb = sumActionLogProb(dist.log_prob(action));
        Tensor value = m.getValue(st);
        return new Tensor[]{
                action.detach().clone(),
                logProb.detach().clone(),
                value.detach().clone()
        };
    }

    /**
     * Normalize a raw observation with the running mean/std (and optionally update stats).
     * Call this before pushing states into the buffer so train-time states match sample-time.
     */
    public Tensor normalizeObs(Tensor state) {
        return maybeNormalizeObs(state, /*updateStats*/false);
    }

    /** Update running stats from a raw observation batch without returning normalized values. */
    public void updateObsStats(Tensor state) {
        if (normalizeObs && obsNormalizer != null) {
            obsNormalizer.update(state);
        }
    }

    private Tensor maybeNormalizeObs(Tensor state, boolean updateStats) {
        if (!normalizeObs || obsNormalizer == null) return state;
        if (updateStats) obsNormalizer.update(state);
        return obsNormalizer.normalize(state);
    }

    public void setNormalizeObs(boolean enabled) { this.normalizeObs = enabled; }
    public boolean isNormalizeObs() { return normalizeObs; }
    public RunningMeanStd getObsNormalizer() { return obsNormalizer; }
    public int getPpoEpochs() { return ppoEpochs; }
    public int getMiniBatchSize() { return miniBatchSize; }

    /**
     * GAE helper — delegates to {@link GAE#compute} (supports values length T or T+1).
     *
     * @param masks 1 = continue, 0 = done
     */
    public static Tensor[] computeGAE(Tensor rewards, Tensor values, Tensor masks,
                                      float gamma, float tau) {
        return GAE.compute(rewards, values, masks, gamma, tau);
    }

    public static Tensor computePPOLoss(Distribution dist, Tensor actions, Tensor oldLogProbs,
                                        Tensor advantages, float clipEps) {
        Tensor logProb = sumActionLogProb(dist.log_prob(actions));
        Tensor ratio = exp(logProb.sub(oldLogProbs));
        Tensor surr1 = ratio.mul(advantages);
        Tensor surr2 = clamp(ratio,
                new ScalarOptional(new Scalar(1.0 - clipEps)),
                new ScalarOptional(new Scalar(1.0 + clipEps))).mul(advantages);
        return min(surr1, surr2).mean().neg();
    }

    /**
     * Sum log-probs over action dimensions only.
     * Discrete Categorical log_prob is already per-sample (no action dim) — do not sum.
     * Continuous / multi-dim actions keep a trailing action axis that must be reduced.
     */
    static Tensor sumActionLogProb(Tensor logProb) {
        if (logProb.dim() <= 1) {
            return logProb;
        }
        return logProb.sum(-1);
    }

    public float getClipEps() { return clipEps; }
    public float getGamma() { return gamma; }
    public float getGaeLambda() { return gaeLambda; }
    public float getEntropyCoeff() { return entropyCoeff; }
    public Optimizer getOptimizer() { return optimizer; }

    @Override
    public void close() {
        super.close();
    }
}
