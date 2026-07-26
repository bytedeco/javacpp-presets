package org.bytedeco.pytorch.rl.trainer;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.rl.ReplayBuffer;

/**
 * Classic (non-LLM) RL trainer contract.
 *
 * <p>Implementations should reuse shared loss math from
 * {@code org.bytedeco.pytorch.llm.trl.loss} (PPOLoss / DPOLoss / GRPOLoss)
 * rather than re-deriving clip / preference / group-relative formulae.
 *
 * <p>For full LLM trainers prefer the {@code org.bytedeco.pytorch.llm.trl.*}
 * hierarchy ({@code PPOTrainer}, {@code DPOTrainer}, {@code GRPOTrainer}, …).
 */
public interface RLTrainer {

    /** One optimization pass over the contents of {@code buffer}. */
    void trainBatch(ReplayBuffer buffer);

    /**
     * Optional: direct loss evaluation without an optimizer step.
     * Default throws — override in concrete trainers.
     */
    default Tensor computeLoss(ReplayBuffer buffer) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not implement computeLoss(ReplayBuffer)");
    }

    /** Human-readable algorithm id, e.g. {@code "ppo"}, {@code "grpo-group-relative"}. */
    default String algorithm() {
        return getClass().getSimpleName();
    }
}
