package org.bytedeco.pytorch.rl.trainer;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.ActorCriticNetwork;

/**
 * Classic (non-LLM) DPO trainer operating on actor-critic policy distributions.
 *
 * <p>For LLM preference tuning prefer
 * {@link org.bytedeco.pytorch.llm.trl.DPOTrainer}.
 */
public class DPOTrainer implements RLTrainer {
    private final float beta;
    private Optimizer optimizer;

    public DPOTrainer(float beta) {
        this.beta = beta;
    }

    public DPOTrainer(float beta, Optimizer optimizer) {
        this.beta = beta;
        this.optimizer = optimizer;
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public float beta() {
        return beta;
    }

    /**
     * Canonical DPO loss — delegates to the shared TRL implementation.
     */
    public Tensor computeLoss(Tensor pLpC, Tensor pLpR, Tensor rLpC, Tensor rLpR) {
        return DPOLoss.compute(pLpC, pLpR, rLpC, rLpR, beta, "sigmoid");
    }

    public Tensor computeLoss(
            Tensor pLpC, Tensor pLpR, Tensor rLpC, Tensor rLpR, String lossType) {
        return DPOLoss.compute(pLpC, pLpR, rLpC, rLpR, beta, lossType);
    }

    /** @deprecated use {@link #computeLoss} */
    @Deprecated
    public Tensor loss(Tensor pLpC, Tensor pLpR, Tensor rLpC, Tensor rLpR, float beta) {
        return DPOLoss.compute(pLpC, pLpR, rLpC, rLpR, beta, "sigmoid");
    }

    /**
     * One preference step with policy + frozen reference actor-critics.
     */
    public Tensor trainPreferenceStep(
            Tensor state,
            Tensor actionChosen,
            Tensor actionRejected,
            ActorCriticNetwork policy,
            ActorCriticNetwork reference) {
        try (PointerScope scope = new PointerScope()) {
            Distribution pi = policy.forward_policy(state);
            Tensor lpChosen = pi.log_prob(actionChosen).sum(new long[]{-1L});
            Tensor lpRejected = pi.log_prob(actionRejected).sum(new long[]{-1L});

            Tensor refLpChosen;
            Tensor refLpRejected;
            try (NoGradGuard guard = new NoGradGuard()) {
                Distribution ref = reference.forward_policy(state);
                refLpChosen = ref.log_prob(actionChosen).sum(new long[]{-1L}).detach();
                refLpRejected = ref.log_prob(actionRejected).sum(new long[]{-1L}).detach();
            }

            Tensor lossT = computeLoss(lpChosen, lpRejected, refLpChosen, refLpRejected);
            if (optimizer != null) {
                optimizer.zero_grad();
                lossT.backward();
                optimizer.step();
            }
            return lossT.detach();
        }
    }

    @Override
    public void trainBatch(ReplayBuffer buffer) {
        // Preference pairs are not stored in the classic ReplayBuffer layout;
        // callers should use trainPreferenceStep or the LLM DPOTrainer.
        if (buffer == null || buffer.size() == 0) {
            return;
        }
    }

    @Override
    public String algorithm() {
        return "dpo";
    }
}
