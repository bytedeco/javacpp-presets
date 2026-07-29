/*
 * OpenOneRec — industrial generative recommendation foundation (Kuaishou-aligned).
 *
 * Reference:
 *   - OpenOneRec https://github.com/Kuaishou-OneRec/OpenOneRec  (arXiv:2512.24762)
 *   - OneRec series: end-to-end generative rec with Itemic Tokens
 *   - MiniOneRec SFT + alignment tasks
 *
 * What "industrial" means in this pure-Module port (no HF/Qwen weights required):
 *   1) Itemic token vocabulary = SemanticID layout (PAD/BOS/EOS + L×K codebooks)
 *      matching OpenOneRec's <|sid_begin|> s_a_* s_b_* s_c_* <|sid_end|> hierarchy
 *      compressed into dense code-token ids for efficient training.
 *   2) Session-level autoregressive generation (history → next-item SID stream).
 *   3) Multi-task SFT heads (optional):
 *        - next-item SID NTP (primary)
 *        - item-understanding auxiliary CE on a pooled hist representation
 *          (stand-in for OpenOneRec Layer-0 semantic alignment without an LLM)
 *   4) Scaling knobs: dModel / nLayers / nHeads / maxSeqLen for 1.7B-class
 *      experiments when coupled with larger codebooks + longer contexts.
 *   5) Hooks for post-training: expose generateItem + logprobs for GRPO/IPA.
 *
 * This is NOT a weight-compatible Qwen3 loader; it is an industrial *architecture*
 * and training contract you can scale, then distill into / align with OpenOneRec
 * Foundation checkpoints when available.
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.enumtype.TransformerActivation;
import org.bytedeco.pytorch.enumtype.kReLU;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.TransformerEncoderImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.TransformerEncoderLayerOptions;
import org.bytedeco.pytorch.nn.options.TransformerEncoderOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class OpenOneRec extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Special itemic markers (beyond SemanticID.PAD/BOS/EOS) reserved in extended vocab. */
    public static final int SID_BEGIN = 3; // shift SemanticID codes by +2 when useItemicMarkers
    public static final int SID_END = 4;
    public static final int ITEMIC_SPECIAL = 5; // first code token if markers enabled

    private final int vocabSize;
    private final int numLevels;
    private final int codebookSize;
    private final long dModel;
    private final int maxSeqLen;
    private final int nLayers;
    private final boolean tieEmbeddings;
    private final boolean useItemicMarkers;
    private final boolean useAuxHead;
    private final String device;
    private final int codeOffset; // SPECIAL base for encode

    private final EmbeddingImpl tokenEmbedding;
    private final EmbeddingImpl positionEmbedding;
    private final DropoutImpl dropout;
    private final TransformerEncoderImpl decoder;
    private final LayerNormImpl finalNorm;
    private final LinearImpl outputProjection;
    private final LinearImpl auxHead; // optional item-understand / domain head

    public OpenOneRec(int numLevels, int codebookSize) {
        this(numLevels, codebookSize, 512, 8, 8, 1024, 0.1,
                true, true, true, DeviceSupport.backend());
    }

    public OpenOneRec(
            int numLevels,
            int codebookSize,
            long dModel,
            long nHeads,
            int nLayers,
            int maxSeqLen,
            double dropout,
            boolean tieEmbeddings,
            boolean useItemicMarkers,
            boolean useAuxHead,
            String device) {
        super("OpenOneRec");
        if (numLevels <= 0 || codebookSize <= 0) {
            throw new IllegalArgumentException("numLevels/codebookSize must be positive");
        }
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        }
        this.numLevels = numLevels;
        this.codebookSize = codebookSize;
        this.useItemicMarkers = useItemicMarkers;
        this.useAuxHead = useAuxHead;
        this.codeOffset = useItemicMarkers ? ITEMIC_SPECIAL : SemanticID.SPECIAL;
        this.vocabSize = codeOffset + numLevels * codebookSize;
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.nLayers = nLayers;
        this.tieEmbeddings = tieEmbeddings;
        this.device = device != null ? device : DeviceSupport.backend();

        EmbeddingOptions tokOpts = new EmbeddingOptions(vocabSize, dModel);
        tokOpts.padding_idx(new LongOptional((long) SemanticID.PAD));
        this.tokenEmbedding = new EmbeddingImpl(tokOpts);
        register_module("token_emb", tokenEmbedding);

        this.positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxSeqLen, dModel));
        register_module("pos_emb", positionEmbedding);

        this.dropout = new DropoutImpl(dropout);
        register_module("drop", this.dropout);

        TransformerEncoderLayerOptions layerOpts =
                new TransformerEncoderLayerOptions(dModel, nHeads)
                        .dim_feedforward(dModel * 4)
                        .dropout(dropout)
                        .activation(new TransformerActivation(new kReLU()));
        this.decoder = new TransformerEncoderImpl(
                new TransformerEncoderOptions(layerOpts, nLayers));
        register_module("decoder", decoder);

        LongVector lnShape = new LongVector(1);
        lnShape.put(0, dModel);
        this.finalNorm = new LayerNormImpl(lnShape);
        register_module("final_norm", finalNorm);

        if (tieEmbeddings) {
            this.outputProjection = null;
        } else {
            this.outputProjection = new LinearImpl(dModel, vocabSize);
            register_module("lm_head", outputProjection);
        }

        if (useAuxHead) {
            // aux: predict first-level code from pooled history (semantic alignment proxy)
            this.auxHead = new LinearImpl(dModel, codebookSize);
            register_module("aux_head", auxHead);
        } else {
            this.auxHead = null;
        }

        initWeights();
        if (!"cpu".equals(this.device)) {
            this.to(new Device(this.device), false);
        }
    }

    private void initWeights() {
        try {
            torch.xavier_uniform_(tokenEmbedding.weight());
            torch.xavier_uniform_(positionEmbedding.weight());
            tokenEmbedding.weight().narrow(0, SemanticID.PAD, 1).fill_(new Scalar(0.0f));
        } catch (Throwable ignored) {
        }
    }

    public int vocabSize() { return vocabSize; }
    public int numLevels() { return numLevels; }
    public int codebookSize() { return codebookSize; }
    public long dModel() { return dModel; }
    public int maxSeqLen() { return maxSeqLen; }
    public int nLayers() { return nLayers; }
    public boolean useItemicMarkers() { return useItemicMarkers; }
    public String device() { return device; }
    public int codeOffset() { return codeOffset; }

    /** Encode raw (level, code) into OpenOneRec token id (respects itemic offset). */
    public int encode(int level, int code) {
        return codeOffset + level * codebookSize + code;
    }

    public int[] encodeSid(int[] codes) {
        int[] out = new int[codes.length];
        for (int i = 0; i < codes.length; i++) out[i] = encode(i, codes[i]);
        return out;
    }

    /**
     * Build industrial itemic span: [SID_BEGIN] + codes + [SID_END] when markers on,
     * else plain SemanticID.encode path.
     */
    public int[] itemicSpan(int[] rawCodes) {
        int[] codes = encodeSid(rawCodes);
        if (!useItemicMarkers) return codes;
        int[] span = new int[codes.length + 2];
        span[0] = SID_BEGIN;
        System.arraycopy(codes, 0, span, 1, codes.length);
        span[span.length - 1] = SID_END;
        return span;
    }

    public Tensor forward(Tensor tokens) {
        long B = tokens.size(0);
        long T = tokens.size(1);
        if (T > maxSeqLen) {
            throw new IllegalArgumentException("seq len " + T + " > maxSeqLen " + maxSeqLen);
        }
        Device dev = new Device(device);
        Tensor tok = tokens.toType(ScalarType.Long);
        try { tok = tok.to(dev, ScalarType.Long); } catch (Throwable ignored) {}

        Tensor positions = torch.arange(new Scalar(0), new Scalar((double) T), new Scalar(1),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        try { positions = positions.to(dev, ScalarType.Long); } catch (Throwable ignored) {}
        positions = positions.unsqueeze(0).expand(new long[]{B, T});

        Tensor x = dropout.forward(
                tokenEmbedding.forward(tok).add(positionEmbedding.forward(positions)));
        Tensor xTB = x.transpose(0, 1);
        Tensor attnMask = OneRec.causalMask(T, device);
        Tensor padMask = tok.eq(new Scalar((long) SemanticID.PAD));

        Tensor encOut;
        try {
            encOut = decoder.forward(xTB, attnMask, padMask);
        } catch (Throwable t1) {
            try {
                encOut = decoder.forward(xTB, attnMask);
            } catch (Throwable t2) {
                encOut = decoder.forward(xTB);
            }
        }
        Tensor h = finalNorm.forward(encOut.transpose(0, 1));
        if (tieEmbeddings) {
            return torch.matmul(h, tokenEmbedding.weight().t());
        }
        return outputProjection.forward(h);
    }

    /** Primary NTP loss (+ optional aux). */
    public Tensor computeLoss(Tensor tokens) {
        return computeLoss(tokens, 0.1f);
    }

    public Tensor computeLoss(Tensor tokens, float auxWeight) {
        Tensor input = tokens.narrow(1, 0, tokens.size(1) - 1);
        Tensor target = tokens.narrow(1, 1, tokens.size(1) - 1);
        Tensor logits = forward(input);
        long B = logits.size(0);
        long T = logits.size(1);
        long V = logits.size(2);
        Tensor flatLogits = logits.reshape(B * T, V);
        Tensor flatTarget = target.reshape(B * T).toType(ScalarType.Long);
        Tensor logProb = torch.log_softmax(flatLogits, 1L);
        Tensor logp = logProb.gather(1, flatTarget.view(-1L, 1L)).squeeze(1).neg();
        Tensor mask = flatTarget.ne(new Scalar((long) SemanticID.PAD)).toType(ScalarType.Float);
        // also ignore SID_BEGIN/END as targets if markers used? keep them supervised
        Tensor denom = mask.sum().clamp_min(new Scalar(1.0f));
        Tensor ntp = logp.mul(mask).sum().div(denom);

        if (!useAuxHead || auxHead == null || auxWeight <= 0f) {
            return ntp;
        }
        // aux: from last non-pad hidden of input, predict level-0 code of last target code token
        try {
            Tensor h = forwardFeatures(input); // [B,T,D]
            // take last position
            Tensor lastH = h.select(1, h.size(1) - 1); // [B,D]
            Tensor auxLogits = auxHead.forward(lastH); // [B,K]
            // target level-0: scan target row for first code token at level 0
            // fallback: 0
            Tensor auxTarget = torch.zeros(new long[]{B},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
            try { auxTarget = auxTarget.to(new Device(device), ScalarType.Long); } catch (Throwable ignored) {}
            // simplified: use target's first non-pad mod codebook as weak label
            Tensor t0 = target.select(1, 0).toType(ScalarType.Long);
            Tensor weak = t0.sub(new Scalar((long) codeOffset)).remainder(new Scalar((long) codebookSize));
            weak = weak.clamp_min(new Scalar(0L)).clamp_max(new Scalar((long) codebookSize - 1));
            Tensor auxLogProb = torch.log_softmax(auxLogits, 1L);
            Tensor auxNll = auxLogProb.gather(1, weak.view(-1L, 1L)).squeeze(1).neg().mean();
            return ntp.add(auxNll.mul(new Scalar(auxWeight)));
        } catch (Throwable t) {
            return ntp;
        }
    }

    /** Hidden states before LM head [B,T,D]. */
    public Tensor forwardFeatures(Tensor tokens) {
        long B = tokens.size(0);
        long T = tokens.size(1);
        Device dev = new Device(device);
        Tensor tok = tokens.toType(ScalarType.Long);
        try { tok = tok.to(dev, ScalarType.Long); } catch (Throwable ignored) {}
        Tensor positions = torch.arange(new Scalar(0), new Scalar((double) T), new Scalar(1),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        try { positions = positions.to(dev, ScalarType.Long); } catch (Throwable ignored) {}
        positions = positions.unsqueeze(0).expand(new long[]{B, T});
        Tensor x = dropout.forward(
                tokenEmbedding.forward(tok).add(positionEmbedding.forward(positions)));
        Tensor xTB = x.transpose(0, 1);
        Tensor attnMask = OneRec.causalMask(T, device);
        Tensor padMask = tok.eq(new Scalar((long) SemanticID.PAD));
        Tensor encOut;
        try {
            encOut = decoder.forward(xTB, attnMask, padMask);
        } catch (Throwable t1) {
            try { encOut = decoder.forward(xTB, attnMask); }
            catch (Throwable t2) { encOut = decoder.forward(xTB); }
        }
        return finalNorm.forward(encOut.transpose(0, 1));
    }

    /**
     * Token log-probs for a full sequence under teacher forcing — for GRPO.
     * @return [B, T-1] log p(token_t | prefix) for t=1..T-1
     */
    public Tensor sequenceLogProbs(Tensor tokens) {
        Tensor input = tokens.narrow(1, 0, tokens.size(1) - 1);
        Tensor target = tokens.narrow(1, 1, tokens.size(1) - 1);
        Tensor logits = forward(input); // [B,T-1,V]
        Tensor logProb = torch.log_softmax(logits, 2L);
        return logProb.gather(2, target.toType(ScalarType.Long).unsqueeze(2)).squeeze(2);
    }

    public Tensor generateItem(Tensor prefixTokens, SemanticID.ConstrainedDecoder[] constrained) {
        // Reuse same greedy loop as OneRec but with this.forward
        int B = (int) prefixTokens.size(0);
        int[][] generated = new int[B][numLevels];
        Tensor cur = prefixTokens.toType(ScalarType.Long);
        for (int step = 0; step < numLevels; step++) {
            Tensor logits = forward(cur);
            Tensor last = logits.select(1, logits.size(1) - 1);
            int[] nextTok = new int[B];
            for (int b = 0; b < B; b++) {
                float[] scores = TensorHelpers.toFloatArray(
                        last.select(0, b).contiguous().cpu().toType(ScalarType.Float));
                if (constrained != null && constrained[b] != null) {
                    // ConstrainedDecoder uses SemanticID.encode offsets — when itemic markers
                    // shift codes, user should build trie with matching encode via encode()
                    constrained[b].maskLogits(scores);
                }
                int best = 0;
                float bestScore = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < scores.length; i++) {
                    if (scores[i] > bestScore) { bestScore = scores[i]; best = i; }
                }
                generated[b][step] = best;
                nextTok[b] = best;
                if (constrained != null && constrained[b] != null) constrained[b].accept(best);
            }
            Tensor next = TensorHelpers.tensor(nextTok, (long) B, 1L).toType(ScalarType.Long);
            try { next = next.to(new Device(device), ScalarType.Long); } catch (Throwable ignored) {}
            cur = TensorHelpers.cat(new Tensor[]{cur, next}, 1);
        }
        float[] flat = new float[B * numLevels];
        int p = 0;
        for (int b = 0; b < B; b++)
            for (int l = 0; l < numLevels; l++) flat[p++] = generated[b][l];
        Tensor out = TensorHelpers.tensor(flat, (long) B, (long) numLevels).toType(ScalarType.Long);
        try { return out.to(new Device(device), ScalarType.Long); }
        catch (Throwable t) { return out; }
    }

    public void summary() {
        System.out.println("=== OpenOneRec (industrial foundation-style) ===");
        System.out.println("  Itemic markers : " + useItemicMarkers
                + "  codeOffset=" + codeOffset);
        System.out.println("  SID L×K        : " + numLevels + "×" + codebookSize
                + "  vocab=" + vocabSize);
        System.out.println("  dModel/layers  : " + dModel + " / " + nLayers);
        System.out.println("  maxSeqLen      : " + maxSeqLen);
        System.out.println("  aux head       : " + useAuxHead);
        System.out.println("  device         : " + device);
        System.out.println("  Note: architecture/training contract for OpenOneRec;");
        System.out.println("        not a Qwen3 weight loader.");
    }
}
