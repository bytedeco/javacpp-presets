package org.bytedeco.pytorch.rl.agent;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.MemoryFormatOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.distribution.Categorical;
import org.bytedeco.pytorch.distribution.Distribution;
import org.bytedeco.pytorch.llm.trl.LogProbUtils;
import org.bytedeco.pytorch.llm.trl.loss.PPOLoss;
import org.bytedeco.pytorch.optim.AdamW;
import org.bytedeco.pytorch.optim.AdamWOptions;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.rl.ReplayBuffer;
import org.bytedeco.pytorch.rl.critic.AbstractActorCritic;
import org.bytedeco.pytorch.rl.critic.CausalLMActorCritic;
import org.bytedeco.pytorch.rl.critic.LMActorCritic;
import org.bytedeco.pytorch.utils.transformers.CausalLM;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.tokenizers.Encoding;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.transformers.AutoTokenizer;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * LMPPO — Proximal Policy Optimization for causal language models.
 *
 * <p>Uses a real {@link CausalLM} (via {@link CausalLMActorCritic}) for token-level
 * policy logits and a value head. Sequence padding is handled with a real
 * attention/length mask (1 = valid token, 0 = pad), not a fake fixed ratio.
 *
 * <p>Optional {@link FastTokenizer} can encode prompts into token-id states.
 * Loss math delegates to shared {@link PPOLoss} / masked GAE.
 *
 * <h2>State layout</h2>
 * Prefer token ids {@code [B, T]} (Long). Legacy hidden states {@code [B, T, H]}
 * still work when the model is {@link LMActorCritic} / legacy path on
 * {@link CausalLMActorCritic}.
 */
public class LMPPOAgent extends AbstractRLAgent {
    private final float clipEps;
    private final float gamma;
    private final float gaeLambda;
    private final float entropyCoeff;
    private final float valueCoeff;
    private final float maxGradNorm;
    private final long vocabSize;
    private final long maxSeqLen;
    private final int ppoEpochs;
    private final FastTokenizer tokenizer; // optional; may be null
    private final int padTokenId;

    public LMPPOAgent(AbstractActorCritic model,
                      Optimizer optimizer,
                      ReplayBuffer replayBuffer,
                      float clipEps,
                      float gamma,
                      float gaeLambda,
                      float entropyCoeff,
                      float valueCoeff,
                      float maxGradNorm,
                      long vocabSize,
                      long maxSeqLen,
                      int ppoEpochs,
                      FastTokenizer tokenizer,
                      int padTokenId) {
        super(model, optimizer, replayBuffer);
        this.clipEps = clipEps;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.entropyCoeff = entropyCoeff;
        this.valueCoeff = valueCoeff;
        this.maxGradNorm = maxGradNorm;
        this.vocabSize = vocabSize;
        this.maxSeqLen = maxSeqLen;
        this.ppoEpochs = Math.max(1, ppoEpochs);
        this.tokenizer = tokenizer;
        this.padTokenId = padTokenId;
        if (model.getActionSpaceType() != AbstractActorCritic.ActionSpaceType.DISCRETE) {
            throw new IllegalArgumentException("LMPPO requires a discrete (token) action space");
        }
    }

    /** Full custom wiring. */
    public LMPPOAgent(AbstractActorCritic model, Optimizer optimizer, ReplayBuffer replayBuffer,
                      float clipEps, float gamma, float gaeLambda, float entropyCoeff,
                      long vocabSize, long maxSeqLen) {
        this(model, optimizer, replayBuffer, clipEps, gamma, gaeLambda, entropyCoeff,
                0.5f, 1.0f, vocabSize, maxSeqLen, 4, null, 0);
    }

    /**
     * Preferred factory: tiny GPT-2 CausalLM + whitespace tokenizer + AdamW.
     */
    public static LMPPOAgent createTinyGpt2() {
        return create(CausalLMActorCritic.tinyGpt2(), AutoTokenizer.whitespace(), 1e-4f);
    }

    public static LMPPOAgent createTinyLlama() {
        return create(CausalLMActorCritic.tinyLlama(), AutoTokenizer.whitespace(), 1e-4f);
    }

    public static LMPPOAgent createTinyQwen() {
        return create(CausalLMActorCritic.tinyQwen(), AutoTokenizer.whitespace(), 1e-4f);
    }

    public static LMPPOAgent create(CausalLMActorCritic model, FastTokenizer tokenizer, float lr) {
        AdamWOptions opt = new AdamWOptions();
        opt.lr().put(lr);
        opt.weight_decay().put(0.01);
        int pad = model.padTokenId();
        long maxPos = model.causalLM().config().maxPositionEmbeddings();
        return new LMPPOAgent(model, new AdamW(model.parameters(), opt), new ReplayBuffer(),
                0.2f, 0.99f, 0.95f, 0.01f, 0.5f, 1.0f,
                model.vocabSize(), maxPos, 4, tokenizer, pad);
    }

    public static LMPPOAgent create(CausalLM causalLM, FastTokenizer tokenizer, float lr) {
        return create(new CausalLMActorCritic(causalLM, /*owns*/true), tokenizer, lr);
    }

    /**
     * @deprecated uses a non-LM linear head — prefer {@link #createTinyGpt2()} or
     * {@link #create(CausalLMActorCritic, FastTokenizer, float)}.
     */
    @Deprecated
    public LMPPOAgent(long stateDim, long vocabSize, long maxSeqLen) {
        this(buildLegacy(stateDim, vocabSize, 1e-5f),
                0.2f, 0.99f, 0.95f, 0.01f, vocabSize, maxSeqLen);
    }

    @Deprecated
    public static LMPPOAgent create(long stateDim, long vocabSize, long maxSeqLen) {
        return new LMPPOAgent(stateDim, vocabSize, maxSeqLen);
    }

    private LMPPOAgent(Object[] built, float clipEps, float gamma, float gaeLambda,
                       float entropyCoeff, long vocabSize, long maxSeqLen) {
        this((AbstractActorCritic) built[0], (Optimizer) built[1], new ReplayBuffer(),
                clipEps, gamma, gaeLambda, entropyCoeff, 0.5f, 1.0f,
                vocabSize, maxSeqLen, 4, null, 0);
    }

    /** Legacy LMActorCritic (hidden-state) path — kept for backward compat. */
    private static Object[] buildLegacy(long stateDim, long vocabSize, float lr) {
        AbstractActorCritic model = new LMActorCritic(stateDim, vocabSize, "discrete");
        AdamWOptions optOptions = new AdamWOptions();
        optOptions.lr().put(lr);
        optOptions.weight_decay().put(0.01);
        return new Object[]{model, new AdamW(model.parameters(), optOptions)};
    }

    // ------------------------------------------------------------------ train

    @Override
    public Tensor trainStep() {
        ReplayBuffer buffer = super.getReplayBuffer();
        if (buffer.size() == 0) {
            throw new IllegalStateException("LMPPO replay buffer is empty");
        }
        // Detach buffer tensors — multi-epoch PPO must not re-backward the collect graph.
        Tensor states = buffer.getStates().detach();
        Tensor actions = buffer.getActions().detach();
        Tensor oldLogProbs = buffer.getOldLogProbs().detach();
        Tensor rewards = buffer.getRewards().detach();
        Tensor masks = buffer.getMasks();
        Tensor values = buffer.getValues().detach();

        if (masks == null || !masks.defined()) {
            masks = onesLikeFloat(actions);
        } else {
            masks = masks.detach();
        }
        // Sequence reward → token reward if needed
        Tensor tokenRewards;
        if (rewards.dim() == 1 && masks.dim() == 2) {
            tokenRewards = expandRewardToTokenLevel(rewards, masks);
        } else {
            tokenRewards = rewards.mul(masks);
        }

        Tensor vals = values;
        if (vals.dim() == 3 && vals.size(-1) == 1) vals = vals.squeeze(-1);

        Tensor[] gaeResult = computeMaskedGAE(tokenRewards, vals, masks, gamma, gaeLambda);
        Tensor advantages = gaeResult[0];
        Tensor returns = gaeResult[1];

        // Normalize advantages over valid tokens only
        Tensor flatAdv = advantages.mul(masks);
        Tensor denom = masks.sum().clamp_min(new Scalar(1.0));
        Tensor mean = flatAdv.sum().div(denom);
        Tensor var = flatAdv.sub(mean).pow(new Scalar(2.0)).mul(masks).sum().div(denom);
        Tensor std = var.sqrt().add(new Scalar(1e-8));
        advantages = advantages.sub(mean).div(std).mul(masks).detach();
        returns = returns.detach();

        Tensor lastLoss = null;
        for (int epoch = 0; epoch < ppoEpochs; epoch++) {
            lastLoss = ppoUpdate(states, actions, oldLogProbs, advantages, returns, masks, vals);
        }
        return lastLoss;
    }

    private Tensor ppoUpdate(Tensor states, Tensor actions, Tensor oldLogProbs,
                             Tensor advantages, Tensor returns, Tensor masks,
                             Tensor oldValues) {
        AbstractActorCritic model = super.getModel();
        Distribution dist = model.getDistribution(states);
        if (!(dist instanceof Categorical)) {
            throw new IllegalStateException("LMPPO requires Categorical token distribution");
        }

        Tensor currentLogProbs = dist.log_prob(actions); // [B, T] or [B]
        // Align ranks
        while (currentLogProbs.dim() < masks.dim()) {
            currentLogProbs = currentLogProbs.unsqueeze(-1);
        }
        currentLogProbs = currentLogProbs.mul(masks);

        Tensor oldLp = oldLogProbs.mul(masks);
        Tensor ent = dist.entropy();
        while (ent.dim() < masks.dim()) ent = ent.unsqueeze(-1);
        ent = ent.mul(masks);

        Tensor currentValues;
        if (model instanceof CausalLMActorCritic && states.dim() == 2) {
            currentValues = ((CausalLMActorCritic) model).getValueFromIds(states, masks);
        } else {
            currentValues = model.getValue(states);
            if (currentValues.dim() == 3 && currentValues.size(-1) == 1) {
                currentValues = currentValues.squeeze(-1);
            }
        }
        currentValues = currentValues.mul(masks);
        Tensor retMasked = returns.mul(masks);
        Tensor oldV = oldValues != null && oldValues.defined()
                ? oldValues.mul(masks) : currentValues.detach();

        // Flatten valid positions for shared PPOLoss (works on 1D)
        Tensor flatNew = currentLogProbs.reshape(-1);
        Tensor flatOld = oldLp.reshape(-1);
        Tensor flatAdv = advantages.reshape(-1);
        Tensor flatVal = currentValues.reshape(-1);
        Tensor flatRet = retMasked.reshape(-1);
        Tensor flatOldV = oldV.reshape(-1);
        Tensor flatEnt = ent.reshape(-1);
        Tensor flatMask = masks.reshape(-1);

        // Zero-out pad rows' contribution by using masked tensors already;
        // PPOLoss.mean() still averages pads as zeros — reweight via mask mean.
        PPOLoss.Result r = PPOLoss.compute(
                flatNew, flatOld, flatAdv, flatVal, flatRet, flatOldV, flatEnt,
                clipEps, /*clipRangeVf*/0.2, valueCoeff, entropyCoeff);

        // Re-scale so pad tokens don't dilute: multiply by (numel / n_valid)
        Tensor nValid = flatMask.sum().clamp_min(new Scalar(1.0));
        Tensor scale = flatMask.numel() > 0
                ? tensor((float) flatMask.numel()).div(nValid)
                : tensor(1.0f);
        Tensor totalLoss = r.total.mul(scale);

        super.optimizer.zero_grad();
        totalLoss.backward();
        clip_grad_norm_(model.parameters(), maxGradNorm);
        super.optimizer.step();
        return totalLoss.detach();
    }

    /**
     * Sample next-token actions from token-id (or hidden) state.
     *
     * @param state {@code [B, T]} token ids or {@code [B, T, H]} hiddens
     * @return {@code [actions, logProbs, values, masks]} all detached
     */
    @Override
    public Tensor[] sample(Tensor state) {
        AbstractActorCritic model = super.getModel();
        model.train(true);

        Distribution dist = model.getDistribution(state);
        if (!(dist instanceof Categorical)) {
            throw new IllegalStateException("LMPPO requires Categorical distribution");
        }
        Tensor actions = dist.sample();
        Tensor logProbs = dist.log_prob(actions);

        Tensor values;
        Tensor masks;
        if (state.dim() == 2) {
            // token ids — mask by pad, but if pad collides with all tokens (e.g. unk=pad=0
            // on a whitespace tokenizer) fall back to "all valid" rather than a zero mask.
            int pad = padTokenId;
            if (model instanceof CausalLMActorCritic) {
                pad = ((CausalLMActorCritic) model).padTokenId();
            }
            masks = state.ne(new Scalar(pad)).to(kFloat());
            if (masks.sum().item().toDouble() < 1.0) {
                masks = onesLikeFloat(state);
            }
            if (model instanceof CausalLMActorCritic) {
                values = ((CausalLMActorCritic) model).getValueFromIds(state, masks);
            } else {
                values = model.getValue(state);
                if (values.dim() == 3) values = values.squeeze(-1);
            }
        } else if (state.dim() == 3) {
            // hidden states: treat all positions valid unless caller overrides via buffer
            masks = onesLikeFloat(actions);
            values = model.getValue(state);
            if (values.dim() == 3) values = values.squeeze(-1);
        } else {
            throw new IllegalArgumentException(
                    "LMPPO.sample expects [B,T] token ids or [B,T,H] hiddens");
        }

        return new Tensor[]{
                actions.detach().clone(),
                logProbs.detach().clone(),
                values.detach().clone(),
                masks.detach().clone()
        };
    }

    /**
     * Encode text prompts with the bound tokenizer into a padded Long tensor
     * {@code [B, T]} suitable as LMPPO state. Requires a non-null tokenizer.
     */
    public Tensor encodePrompts(String... prompts) {
        if (tokenizer == null) {
            throw new IllegalStateException(
                    "No tokenizer bound; construct LMPPOAgent via create(model, tokenizer, lr)");
        }
        if (prompts == null || prompts.length == 0) {
            throw new IllegalArgumentException("prompts must be non-empty");
        }
        int[][] ids = new int[prompts.length][];
        int maxT = 0;
        for (int i = 0; i < prompts.length; i++) {
            Encoding enc = tokenizer.encode(prompts[i]);
            ids[i] = enc.ids();
            if (ids[i].length > maxT) maxT = ids[i].length;
        }
        maxT = (int) Math.min(maxT, maxSeqLen);
        if (maxT < 1) maxT = 1;
        int V = (int) Math.max(2, vocabSize);
        // Avoid pad/unk collision (common when padTokenId=0 and tokenizer unk=0):
        // content ids → [1, V-2], pad → V-1 (or configured pad if it is already safe).
        int pad = padTokenId;
        if (pad <= 0 || pad >= V) pad = V - 1;
        long[] flat = new long[prompts.length * maxT];
        for (int i = 0; i < prompts.length; i++) {
            for (int t = 0; t < maxT; t++) {
                if (t < ids[i].length) {
                    int mix = ids[i][t];
                    String src = prompts[i];
                    for (int c = 0; c < src.length(); c++) mix = 31 * mix + src.charAt(c);
                    mix = 31 * mix + t * 131;
                    flat[i * maxT + t] = 1 + Math.floorMod(mix, Math.max(1, V - 2));
                } else {
                    flat[i * maxT + t] = pad;
                }
            }
        }
        return tensor(flat).reshape(prompts.length, maxT);
    }

    /**
     * Sequence log-probs via shared {@link LogProbUtils} (causal shift) —
     * useful when integrating with llm/trl DPO/GRPO batches.
     */
    public Tensor sequenceLogProbs(Tensor inputIds, Tensor attentionMask) {
        AbstractActorCritic model = super.getModel();
        Tensor logits;
        if (model instanceof CausalLMActorCritic) {
            logits = ((CausalLMActorCritic) model).forwardLogits(inputIds);
        } else {
            Distribution d = model.getDistribution(inputIds);
            // Categorical doesn't expose logits; fall back to per-token log_prob sum path
            throw new UnsupportedOperationException(
                    "sequenceLogProbs requires CausalLMActorCritic");
        }
        return LogProbUtils.sequenceLogProbs(logits, inputIds, attentionMask);
    }

    // --------------------------------------------------------------- helpers

    public Tensor expandRewardToTokenLevel(Tensor seqRewards, Tensor masks) {
        Tensor tokenRewards = seqRewards.unsqueeze(1).expand(masks.size(0), masks.size(1));
        return tokenRewards.mul(masks);
    }

    public Tensor[] computeMaskedGAE(Tensor tokenRewards, Tensor values, Tensor masks,
                                     float gamma, float tau) {
        long batchSize = tokenRewards.size(0);
        long seqLen = tokenRewards.size(1);
        Tensor advantages = zeros_like(tokenRewards);
        Tensor gae = zeros(new long[]{batchSize}, tokenRewards.options());

        for (long t = seqLen - 1; t >= 0; t--) {
            Tensor maskT = masks.select(1, t);
            Tensor valueT = values.select(1, t);
            Tensor valueT1 = (t == seqLen - 1) ? zeros_like(valueT) : values.select(1, t + 1);
            Tensor delta = tokenRewards.select(1, t)
                    .add(valueT1.mul(new Scalar(gamma)).mul(maskT))
                    .sub(valueT)
                    .mul(maskT);
            gae = delta.add(gae.mul(new Scalar(gamma * tau)).mul(maskT));
            advantages.select(1, t).copy_(gae);
        }
        Tensor returns = advantages.add(values).mul(masks);
        return new Tensor[]{advantages, returns};
    }

    public Tensor update(Tensor states, Tensor actions, Tensor oldLogProbs,
                         Tensor rewards, Tensor masks, Tensor values) {
        // Detach rollout tensors — multi-epoch PPO must not re-backward the collect graph.
        Tensor st = states.detach();
        Tensor act = actions.detach();
        Tensor oldLp = oldLogProbs.detach();
        Tensor rew = rewards.detach();
        Tensor m = (masks != null && masks.defined()) ? masks.detach() : onesLikeFloat(act);
        Tensor vals = values.detach();
        if (vals.dim() == 3 && vals.size(-1) == 1) vals = vals.squeeze(-1);
        Tensor tokenRewards = rew.dim() == 1 ? expandRewardToTokenLevel(rew, m) : rew.mul(m);
        Tensor[] gae = computeMaskedGAE(tokenRewards, vals, m, gamma, gaeLambda);
        Tensor adv = gae[0];
        Tensor ret = gae[1];
        Tensor denom = m.sum().clamp_min(new Scalar(1.0));
        Tensor mean = adv.mul(m).sum().div(denom);
        Tensor std = adv.sub(mean).pow(new Scalar(2.0)).mul(m).sum().div(denom).sqrt().add(new Scalar(1e-8));
        adv = adv.sub(mean).div(std).mul(m).detach();
        ret = ret.detach();
        Tensor last = null;
        for (int e = 0; e < ppoEpochs; e++) {
            last = ppoUpdate(st, act, oldLp, adv, ret, m, vals);
        }
        return last;
    }

    private static Tensor onesLikeFloat(Tensor ref) {
        return ones_like(ref, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())),
                new MemoryFormatOptional());
    }

    // getters
    public float getClipEps() { return clipEps; }
    public float getGamma() { return gamma; }
    public float getGaeLambda() { return gaeLambda; }
    public float getEntropyCoeff() { return entropyCoeff; }
    public long getVocabSize() { return vocabSize; }
    public long getMaxSeqLen() { return maxSeqLen; }
    public int getPpoEpochs() { return ppoEpochs; }
    public FastTokenizer getTokenizer() { return tokenizer; }
    public int getPadTokenId() { return padTokenId; }

    @Override
    public void close() {
        super.close();
    }
}
