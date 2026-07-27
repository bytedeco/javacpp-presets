package org.bytedeco.pytorch.rl.critic;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.transformers.CausalLM;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;

import static org.bytedeco.pytorch.global.torch.kFloat;
import static org.bytedeco.pytorch.global.torch.relu;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * Actor-critic over a real {@link CausalLM} for LM-RL (LMPPO / preference / GRPO).
 *
 * <p><b>Policy</b> — Categorical over next-token logits {@code [B,T,V]} from
 * {@link CausalLM#forward(Tensor)}.
 * <p><b>Value</b> — per-token scalar from a small head on a learned
 * {@code V→H} projection of the logits (keeps the critic end-to-end differentiable
 * without needing internal residual-stream hooks).
 *
 * <p><b>State contract</b>
 * <ul>
 *   <li>{@code [B, T]} Long token ids — real LM path (preferred)</li>
 *   <li>{@code [B, T, H]} float hiddens — legacy linear head (compat)</li>
 * </ul>
 */
public class CausalLMActorCritic extends AbstractActorCritic {
    private final CausalLM causalLM;
    private final LinearImpl logitsToHidden; // V → H (critic feature)
    private final LinearImpl valueHead;      // H → 1
    private final LinearImpl legacyPolicy;   // H → V (hidden-state fallback)
    private final LinearImpl legacyValue;    // H → 1
    private final long vocabSize;
    private final long hiddenSize;
    private final int padTokenId;
    private final boolean ownsCausalLM;

    public CausalLMActorCritic(CausalLM causalLM) {
        this(causalLM, true);
    }

    public CausalLMActorCritic(CausalLM causalLM, boolean ownsCausalLM) {
        super(causalLM.hiddenSize(), causalLM.vocabSize(), ActionSpaceType.DISCRETE);
        this.causalLM = causalLM;
        this.ownsCausalLM = ownsCausalLM;
        this.vocabSize = causalLM.vocabSize();
        this.hiddenSize = causalLM.hiddenSize();
        this.padTokenId = causalLM.config().padTokenId();

        this.logitsToHidden = register_module("logits_to_hidden",
                new LinearImpl(vocabSize, hiddenSize));
        this.valueHead = register_module("value_head",
                new LinearImpl(hiddenSize, 1));
        this.legacyPolicy = register_module("legacy_policy",
                new LinearImpl(hiddenSize, vocabSize));
        this.legacyValue = register_module("legacy_value",
                new LinearImpl(hiddenSize, 1));

        try {
            register_module("causal_lm", causalLM);
        } catch (Exception ignored) {
            // already parented elsewhere — still reachable via field
        }
    }

    public static CausalLMActorCritic tinyGpt2() {
        return new CausalLMActorCritic(CausalLM.fromConfig(PretrainedConfig.tinyGpt2()));
    }

    public static CausalLMActorCritic tinyLlama() {
        return new CausalLMActorCritic(CausalLM.fromConfig(PretrainedConfig.tinyLlama()));
    }

    public static CausalLMActorCritic tinyQwen() {
        return new CausalLMActorCritic(CausalLM.fromConfig(PretrainedConfig.tinyQwen()));
    }

    public CausalLM causalLM() { return causalLM; }
    public long vocabSize() { return vocabSize; }
    public long hiddenSize() { return hiddenSize; }
    public int padTokenId() { return padTokenId; }

    public Tensor forwardLogits(Tensor inputIds) {
        return causalLM.forward(inputIds);
    }

    /** Float mask {@code [B,T]} with 1 where token != pad. */
    public Tensor attentionMaskFromIds(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        return ids.ne(new Scalar(padTokenId)).to(kFloat());
    }

    @Override
    public Distribution getDistribution(Tensor state) {
        Tensor logits;
        if (state.dim() == 2) {
            logits = causalLM.forward(state);
        } else if (state.dim() == 3) {
            logits = legacyPolicy.forward(relu(state));
        } else {
            throw new IllegalArgumentException(
                    "CausalLMActorCritic expects [B,T] ids or [B,T,H] hiddens, got dim="
                            + state.dim());
        }
        Tensor stable = logits.sub(logits.amax(new long[]{-1}, true))
                .clamp(new ScalarOptional(new Scalar(-20.0)),
                        new ScalarOptional(new Scalar(20.0)));
        return new Categorical(softmax(stable, -1));
    }

    @Override
    public Tensor getValue(Tensor state) {
        if (state.dim() == 2) {
            return getValueFromIds(state, null); // [B, T]
        }
        if (state.dim() == 3) {
            return legacyValue.forward(relu(state)); // [B, T, 1]
        }
        throw new IllegalArgumentException(
                "CausalLMActorCritic expects [B,T] or [B,T,H], got dim=" + state.dim());
    }

    /**
     * Per-token values {@code [B, T]} from token ids.
     * Optional {@code attentionMask [B,T]} zeros pad positions.
     */
    public Tensor getValueFromIds(Tensor inputIds, Tensor attentionMask) {
        Tensor logits = causalLM.forward(inputIds);          // [B, T, V]
        Tensor h = relu(logitsToHidden.forward(logits));     // [B, T, H]
        Tensor values = valueHead.forward(h).squeeze(-1);    // [B, T]
        if (attentionMask != null && attentionMask.defined()) {
            values = values.mul(attentionMask);
        }
        return values;
    }

    /** Attach LoRA on the underlying CausalLM; returns adapter count. */
    public int attachLora(LoraConfig cfg) {
        return causalLM.attachLora(cfg);
    }

    @Override
    public void close() {
        try { if (logitsToHidden != null) logitsToHidden.close(); } catch (Throwable ignored) {}
        try { if (valueHead != null) valueHead.close(); } catch (Throwable ignored) {}
        try { if (legacyPolicy != null) legacyPolicy.close(); } catch (Throwable ignored) {}
        try { if (legacyValue != null) legacyValue.close(); } catch (Throwable ignored) {}
        if (ownsCausalLM) {
            try { if (causalLM != null) causalLM.close(); } catch (Throwable ignored) {}
        }
    }
}
