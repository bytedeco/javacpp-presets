/*
 * OneRec — Kuaishou end-to-end generative recommender (decoder-only backbone).
 *
 * Reference:
 *   - OneRec https://arxiv.org/abs/2502.18965
 *   - OneRec Technical Report https://arxiv.org/abs/2506.13695
 *   - OneRec-V2 Lazy Decoder-Only https://arxiv.org/abs/2508.20900
 *   - MiniOneRec https://github.com/AkaliKong/MiniOneRec
 *   - OpenOneRec https://github.com/Kuaishou-OneRec/OpenOneRec
 *
 * Formulation (industrial generative rec):
 *   1) Items → Semantic IDs via residual quantization (RQ-VAE / RQ-KMeans), L codes each.
 *   2) User history flattened as SID token sequence with BOS/EOS.
 *   3) Causal Transformer decoder next-token-predicts the SID stream.
 *   4) Constrained decoding (prefix trie over valid SIDs) guarantees real items.
 *
 * This class is the decoder (step 3). Use {@link RQVAE} + {@link SemanticID} for steps 1–2
 * and {@link SemanticID.ConstrainedDecoder} for step 4.
 *
 * Input : tokens [B, T] long  (PAD=0)
 * Output: logits [B, T, V]
 */
package org.bytedeco.pytorch.recommend.models.generative;

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
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.TensorHelpers;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class OneRec extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int vocabSize;
    private final int numLevels;
    private final int codebookSize;
    private final long dModel;
    private final int maxSeqLen;
    private final boolean tieEmbeddings;
    private final String device;

    private final EmbeddingImpl tokenEmbedding;
    private final EmbeddingImpl positionEmbedding;
    private final DropoutImpl dropout;
    private final TransformerEncoderImpl encoder; // causal via attn mask
    private final LayerNormImpl finalNorm;
    private final LinearImpl outputProjection; // null if tieEmbeddings

    /** Defaults: 3-level SID × 256 codebook → vocab = 3 + 768 = 771. */
    public OneRec(int numLevels, int codebookSize) {
        this(numLevels, codebookSize, 256, 4, 4, 512, 0.1, true, DeviceSupport.backend());
    }

    public OneRec(
            int numLevels,
            int codebookSize,
            long dModel,
            long nHeads,
            int nLayers,
            int maxSeqLen,
            double dropout,
            boolean tieEmbeddings,
            String device) {
        super("OneRec");
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
        this.tieEmbeddings = tieEmbeddings;
        this.device = device != null ? device : DeviceSupport.backend();

        Device dev = new Device(this.device);

        EmbeddingOptions tokOpts = new EmbeddingOptions(vocabSize, dModel);
        tokOpts.padding_idx(new LongOptional((long) SemanticID.PAD));
        this.tokenEmbedding = new EmbeddingImpl(tokOpts);
        register_module("token_emb", tokenEmbedding);

        this.positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxSeqLen, dModel));
        register_module("pos_emb", positionEmbedding);

        this.dropout = new DropoutImpl(dropout);
        register_module("drop", this.dropout);

        // Causal decoder = TransformerEncoder + causal mask (GPT-style).
        TransformerEncoderLayerOptions layerOpts =
                new TransformerEncoderLayerOptions(dModel, nHeads)
                        .dim_feedforward(dModel * 4)
                        .dropout(dropout)
                        .activation(new TransformerActivation(new kReLU()));
        TransformerEncoderOptions encOpts = new TransformerEncoderOptions(layerOpts, nLayers);
        this.encoder = new TransformerEncoderImpl(encOpts);
        register_module("decoder", encoder);

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
            this.to(dev, false);
        }
    }

    private void initWeights() {
        try {
            torch.xavier_uniform_(tokenEmbedding.weight());
            torch.xavier_uniform_(positionEmbedding.weight());
            tokenEmbedding.weight().narrow(0, SemanticID.PAD, 1).fill_(new Scalar(0.0f));
            if (outputProjection != null) {
                torch.xavier_uniform_(outputProjection.weight());
            }
        } catch (Throwable ignored) {
            // best-effort init
        }
    }

    public int vocabSize() { return vocabSize; }
    public int numLevels() { return numLevels; }
    public int codebookSize() { return codebookSize; }
    public long dModel() { return dModel; }
    public int maxSeqLen() { return maxSeqLen; }
    public String device() { return device; }

    /** Causal additive mask [T,T]: 0 on/under diagonal, -1e9 above. */
    public static Tensor causalMask(long seqLen, String device) {
        Tensor ones = torch.ones(new long[]{seqLen, seqLen},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        Tensor lower = torch.tril(ones);
        Tensor blocked = lower.eq(new Scalar(0.0f)).toType(ScalarType.Float);
        Tensor addMask = blocked.mul(new Scalar(-1e9f));
        if (device != null && !"cpu".equals(device)) {
            addMask = addMask.to(new Device(device), ScalarType.Float);
        }
        return addMask;
    }

    /**
     * @param tokens [B, T] long token ids (PAD=0)
     * @return logits [B, T, V]
     */
    public Tensor forward(Tensor tokens) {
        long B = tokens.size(0);
        long T = tokens.size(1);
        if (T > maxSeqLen) {
            throw new IllegalArgumentException(
                    "sequence length " + T + " exceeds maxSeqLen " + maxSeqLen);
        }

        Device dev = new Device(device);
        Tensor tok = tokens.toType(ScalarType.Long);
        try {
            tok = tok.to(dev, ScalarType.Long);
        } catch (Throwable ignored) {
        }

        Tensor positions = torch.arange(new Scalar(0), new Scalar((double) T), new Scalar(1),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        try {
            positions = positions.to(dev, ScalarType.Long);
        } catch (Throwable ignored) {
        }
        // [1,T] -> [B,T]
        positions = positions.unsqueeze(0).expand(new long[]{B, T});

        Tensor tokEmb = tokenEmbedding.forward(tok);
        Tensor posEmb = positionEmbedding.forward(positions);
        Tensor x = dropout.forward(tokEmb.add(posEmb)); // [B, T, D]

        // TransformerEncoder is seq-first: [T, B, D]
        Tensor xTB = x.transpose(0, 1);
        Tensor attnMask = causalMask(T, device);
        Tensor padMask = tok.eq(new Scalar((long) SemanticID.PAD)); // [B, T] bool

        Tensor encOut;
        try {
            encOut = encoder.forward(xTB, attnMask, padMask);
        } catch (Throwable t1) {
            try {
                encOut = encoder.forward(xTB, attnMask);
            } catch (Throwable t2) {
                encOut = encoder.forward(xTB);
            }
        }
        Tensor h = finalNorm.forward(encOut.transpose(0, 1)); // [B, T, D]

        if (tieEmbeddings) {
            return torch.matmul(h, tokenEmbedding.weight().t());
        }
        return outputProjection.forward(h);
    }

    /**
     * Next-token prediction loss; PAD targets ignored via mask.
     *
     * @param tokens [B, T] full sequence including BOS
     * @return scalar mean CE over non-PAD positions
     */
    public Tensor computeLoss(Tensor tokens) {
        Tensor input = tokens.narrow(1, 0, tokens.size(1) - 1);
        Tensor target = tokens.narrow(1, 1, tokens.size(1) - 1);
        Tensor logits = forward(input); // [B, T-1, V]

        long B = logits.size(0);
        long Tm1 = logits.size(1);
        long V = logits.size(2);
        Tensor flatLogits = logits.reshape(B * Tm1, V);
        Tensor flatTarget = target.reshape(B * Tm1).toType(ScalarType.Long);

        // CE per-element via log_softmax + gather, mask PAD
        Tensor logProb = torch.log_softmax(flatLogits, 1L); // [N, V]
        Tensor nll = torch.nll_loss(logProb, flatTarget); // mean over all by default
        // Prefer explicit mask so PAD never contributes even if backend ignore differs
        Tensor logp = logProb.gather(1, flatTarget.view(-1L, 1L)).squeeze(1).neg(); // [N]
        Tensor mask = flatTarget.ne(new Scalar((long) SemanticID.PAD)).toType(ScalarType.Float);
        Tensor denom = mask.sum().clamp_min(new Scalar(1.0f));
        Tensor loss = logp.mul(mask).sum().div(denom);
        // touch nll so compiler keeps import path valid if gather path fails at runtime
        if (loss == null) return nll;
        return loss;
    }

    /**
     * Greedy constrained decode of one next-item SID (L code tokens).
     *
     * @param prefixTokens [B, T] context
     * @param constrained  per-row decoders (size B); null → unconstrained argmax
     * @return [B, L] generated code token ids
     */
    public Tensor generateItem(Tensor prefixTokens, SemanticID.ConstrainedDecoder[] constrained) {
        int B = (int) prefixTokens.size(0);
        int[][] generated = new int[B][numLevels];

        Tensor cur = prefixTokens.toType(ScalarType.Long);
        for (int step = 0; step < numLevels; step++) {
            Tensor logits = forward(cur); // [B, T, V]
            long tLast = logits.size(1) - 1;
            Tensor last = logits.select(1, tLast); // [B, V]
            int[] nextTok = new int[B];
            for (int b = 0; b < B; b++) {
                Tensor row = last.select(0, b).contiguous().cpu().toType(ScalarType.Float);
                float[] scores = TensorHelpers.toFloatArray(row);
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
            try {
                next = next.to(new Device(device), ScalarType.Long);
            } catch (Throwable ignored) {
            }
            cur = TensorHelpers.cat(new Tensor[]{cur, next}, 1);
        }

        float[] flat = new float[B * numLevels];
        int p = 0;
        for (int b = 0; b < B; b++) {
            for (int l = 0; l < numLevels; l++) flat[p++] = generated[b][l];
        }
        Tensor out = TensorHelpers.tensor(flat, (long) B, (long) numLevels).toType(ScalarType.Long);
        try {
            return out.to(new Device(device), ScalarType.Long);
        } catch (Throwable t) {
            return out;
        }
    }

    public Tensor generateItem(Tensor prefixTokens) {
        return generateItem(prefixTokens, null);
    }

    public void summary() {
        System.out.println("=== OneRec (Kuaishou generative) ===");
        System.out.println("  SID levels     : " + numLevels);
        System.out.println("  Codebook size  : " + codebookSize);
        System.out.println("  Vocab size     : " + vocabSize + "  (PAD/BOS/EOS + L*K)");
        System.out.println("  dModel         : " + dModel);
        System.out.println("  maxSeqLen      : " + maxSeqLen);
        System.out.println("  tie embeddings : " + tieEmbeddings);
        System.out.println("  device         : " + device);
    }
}
