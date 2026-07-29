/*
 * OneRec-V2 style Lazy Decoder-Only / encoder-light generative recommender.
 *
 * Reference:
 *   - OneRec-V2 Technical Report https://arxiv.org/abs/2508.20900
 *     Key insight: V1 spent ~97.66% FLOPs on sequence encoding rather than generation.
 *     V2 uses a lazy / encoder-light design so compute focuses on generating SID tokens.
 *
 * Design (faithful lightweight industrial approximation in pure Module form):
 *   History tokens  → shallow encoder (1 layer, optional mean-pool summary)
 *   Generation slots → deep causal decoder cross-attending to history memory
 *
 * For SFT we still do full-sequence NTP, but the architecture separates:
 *   - histEncoder: few layers over history prefix
 *   - genDecoder: deeper causal stack; gen positions attend to hist memory via
 *     gated residual fusion (simulates cross-attn without full MHA cross module)
 *
 * API mirrors {@link OneRec}: forward(tokens), computeLoss(tokens), generateItem(...).
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
public class OneRecV2 extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int vocabSize;
    private final int numLevels;
    private final int codebookSize;
    private final long dModel;
    private final int maxSeqLen;
    private final int histLayers;
    private final int genLayers;
    private final boolean tieEmbeddings;
    private final String device;

    private final EmbeddingImpl tokenEmbedding;
    private final EmbeddingImpl positionEmbedding;
    private final DropoutImpl dropout;
    private final TransformerEncoderImpl histEncoder; // shallow
    private final TransformerEncoderImpl genDecoder;  // deeper causal
    private final LinearImpl fuseGate; // sigmoid gate: mix hist summary into gen states
    private final LinearImpl histProj;
    private final LayerNormImpl finalNorm;
    private final LinearImpl outputProjection;

    public OneRecV2(int numLevels, int codebookSize) {
        this(numLevels, codebookSize, 256, 4, 1, 4, 512, 0.1, true, DeviceSupport.backend());
    }

    /**
     * @param histLayers shallow history encoder depth (OneRec-V2: light)
     * @param genLayers  deeper generator depth
     */
    public OneRecV2(
            int numLevels,
            int codebookSize,
            long dModel,
            long nHeads,
            int histLayers,
            int genLayers,
            int maxSeqLen,
            double dropout,
            boolean tieEmbeddings,
            String device) {
        super("OneRecV2");
        if (numLevels <= 0 || codebookSize <= 0) {
            throw new IllegalArgumentException("numLevels/codebookSize must be positive");
        }
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel must be divisible by nHeads");
        }
        this.numLevels = numLevels;
        this.codebookSize = codebookSize;
        this.vocabSize = SemanticID.vocabSize(numLevels, codebookSize);
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.histLayers = Math.max(1, histLayers);
        this.genLayers = Math.max(1, genLayers);
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

        this.histEncoder = buildEncoder(dModel, nHeads, this.histLayers, dropout);
        register_module("hist_enc", histEncoder);

        this.genDecoder = buildEncoder(dModel, nHeads, this.genLayers, dropout);
        register_module("gen_dec", genDecoder);

        this.histProj = new LinearImpl(dModel, dModel);
        register_module("hist_proj", histProj);

        this.fuseGate = new LinearImpl(dModel * 2, dModel);
        register_module("fuse_gate", fuseGate);

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

        initWeights();
        if (!"cpu".equals(this.device)) {
            this.to(new Device(this.device), false);
        }
    }

    private static TransformerEncoderImpl buildEncoder(
            long dModel, long nHeads, int nLayers, double dropout) {
        TransformerEncoderLayerOptions layerOpts =
                new TransformerEncoderLayerOptions(dModel, nHeads)
                        .dim_feedforward(dModel * 4)
                        .dropout(dropout)
                        .activation(new TransformerActivation(new kReLU()));
        return new TransformerEncoderImpl(new TransformerEncoderOptions(layerOpts, nLayers));
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
    public int histLayers() { return histLayers; }
    public int genLayers() { return genLayers; }
    public String device() { return device; }

    public static Tensor causalMask(long seqLen, String device) {
        return OneRec.causalMask(seqLen, device);
    }

    /**
     * Lazy forward:
     *   1) embed all tokens
     *   2) hist = encoder over full seq with causal mask but only 1 light layer
     *      (cheap contextualisation of history)
     *   3) gen = deeper causal decoder on same embeddings
     *   4) fuse: h = gen + σ(W[gen; hist_summary]) * hist_proj(hist)
     *      where hist_summary is broadcast mean of non-pad hist states
     */
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

        Tensor padMask = tok.eq(new Scalar((long) SemanticID.PAD));
        Tensor attnMask = causalMask(T, device);
        Tensor xTB = x.transpose(0, 1); // [T,B,D]

        Tensor histOut = runEncoder(histEncoder, xTB, attnMask, padMask);
        Tensor genOut = runEncoder(genDecoder, xTB, attnMask, padMask);
        // back to [B,T,D]
        Tensor hist = histOut.transpose(0, 1);
        Tensor gen = genOut.transpose(0, 1);

        // masked mean history summary [B,1,D]
        Tensor valid = padMask.logical_not().toType(ScalarType.Float).unsqueeze(2); // [B,T,1]
        Tensor histSum = hist.mul(valid).sum(new long[]{1L}); // [B,D]
        Tensor histCnt = valid.sum(new long[]{1L}).clamp_min(new Scalar(1.0f)); // [B,1]
        Tensor histSummary = histSum.div(histCnt).unsqueeze(1); // [B,1,D]
        Tensor histBroadcast = histSummary.expand(new long[]{B, T, dModel});
        Tensor histCtx = histProj.forward(hist.mul(new Scalar(0.5f)).add(histBroadcast.mul(new Scalar(0.5f))));

        Tensor gateIn = TensorHelpers.cat(new Tensor[]{gen, histCtx}, 2);
        // gate over last dim chunks — use linear then sigmoid
        Tensor gate = fuseGate.forward(gateIn).sigmoid();
        Tensor fused = gen.add(gate.mul(histCtx));
        Tensor h = finalNorm.forward(fused);

        if (tieEmbeddings) {
            return torch.matmul(h, tokenEmbedding.weight().t());
        }
        return outputProjection.forward(h);
    }

    private Tensor runEncoder(TransformerEncoderImpl enc, Tensor xTB, Tensor attnMask, Tensor padMask) {
        try {
            return enc.forward(xTB, attnMask, padMask);
        } catch (Throwable t1) {
            try {
                return enc.forward(xTB, attnMask);
            } catch (Throwable t2) {
                return enc.forward(xTB);
            }
        }
    }

    public Tensor computeLoss(Tensor tokens) {
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
        Tensor denom = mask.sum().clamp_min(new Scalar(1.0f));
        return logp.mul(mask).sum().div(denom);
    }

    /** Greedy constrained generate — same contract as OneRec. */
    public Tensor generateItem(Tensor prefixTokens, SemanticID.ConstrainedDecoder[] constrained) {
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
                    constrained[b].maskLogits(scores);
                }
                int best = 0;
                float bestScore = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < scores.length; i++) {
                    if (scores[i] > bestScore) {
                        bestScore = scores[i];
                        best = i;
                    }
                }
                generated[b][step] = best;
                nextTok[b] = best;
                if (constrained != null && constrained[b] != null) {
                    constrained[b].accept(best);
                }
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

    public Tensor generateItem(Tensor prefixTokens) {
        return generateItem(prefixTokens, null);
    }

    public void summary() {
        System.out.println("=== OneRec-V2 (Lazy Decoder-Only) ===");
        System.out.println("  SID levels     : " + numLevels + "  K=" + codebookSize
                + "  vocab=" + vocabSize);
        System.out.println("  dModel         : " + dModel);
        System.out.println("  histLayers     : " + histLayers + " (light encoder)");
        System.out.println("  genLayers      : " + genLayers + " (generator)");
        System.out.println("  maxSeqLen      : " + maxSeqLen);
        System.out.println("  device         : " + device);
    }
}
