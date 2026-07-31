/*
 * ProteinSeqEncoder — lightweight protein sequence encoder for bioinformatics.
 *
 * Context:
 *   Full ESM-2 / ProtBERT models are multi-GB foundation models; this module
 *   provides a *library-native* encoder with the same *interface pattern*
 *   (residue tokens -> contextual embedding -> pooled protein vector) used by:
 *     - Rives et al., "Biological structure and function emerge from scaling
 *       unsupervised learning to 250 million protein sequences", PNAS 2021 (ESM)
 *     - Elnaggar et al., "ProtTrans", IEEE TPAMI 2021
 *
 *   Use cases inside this recommend/pharma stack:
 *     - freeze / fine-tune as protein tower in DeepDTA / MolTrans / DrugBAN
 *     - protein–protein interaction (PPI) twin-tower scoring
 *     - sequence-level representation for gene-protein linkage features
 *
 * Architecture:
 *   residue embedding + sinusoidal / learned positional embedding
 *   -> N × (MultiHeadSelfAttention + FFN) Pre-Norm blocks
 *   -> AdditiveAttention pool (or CLS / mean)
 *
 * Amino-acid vocab convention (common):
 *   0=pad, 1=unk, 2-21 = ACDEFGHIKLMNPQRSTVWY (20 std AA), optional specials.
 */
package org.bytedeco.pytorch.recommend.models.bio;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.industry.AdditiveAttention;
import org.bytedeco.pytorch.recommend.basic.layers.industry.MultiHeadSelfAttention;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ProteinSeqEncoder extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Standard 20 amino acids + pad + unk (minimal vocab size suggestion). */
    public static final int DEFAULT_AA_VOCAB = 22;

    private final EmbeddingImpl residueEmbedding;
    private final EmbeddingImpl positionEmbedding;
    private final List<MultiHeadSelfAttention> attns = new ArrayList<>();
    private final List<LayerNormImpl> norm1 = new ArrayList<>();
    private final List<LayerNormImpl> norm2 = new ArrayList<>();
    private final List<LinearImpl> ffn1 = new ArrayList<>();
    private final List<LinearImpl> ffn2 = new ArrayList<>();
    private final LayerNormImpl finalNorm;
    private final AdditiveAttention pool;
    private final int embedDim;
    private final int maxLen;
    private final int numLayers;
    private final float dropoutProb;

    public ProteinSeqEncoder(int vocabSize, int maxLen) {
        this(vocabSize, maxLen, 128, 4, 2, 256, 0.1f, DeviceSupport.backend());
    }

    public ProteinSeqEncoder(int vocabSize, int maxLen, int embedDim, int numHeads,
                             int numLayers, int ffnDim, float dropout, String device) {
        super("ProteinSeqEncoder");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.maxLen = maxLen;
        this.numLayers = numLayers;
        this.dropoutProb = dropout;

        EmbeddingOptions rOpts = new EmbeddingOptions(Math.max(vocabSize, DEFAULT_AA_VOCAB), embedDim);
        rOpts.padding_idx().put(new LongOptional(0L));
        this.residueEmbedding = new EmbeddingImpl(rOpts);
        register_module("residue_embedding", residueEmbedding);

        this.positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxLen, embedDim));
        register_module("position_embedding", positionEmbedding);

        for (int i = 0; i < numLayers; i++) {
            LongVector s1 = new LongVector(1); s1.put(0, embedDim);
            LongVector s2 = new LongVector(1); s2.put(0, embedDim);
            LayerNormImpl n1 = new LayerNormImpl(s1);
            LayerNormImpl n2 = new LayerNormImpl(s2);
            MultiHeadSelfAttention attn = new MultiHeadSelfAttention(embedDim, numHeads, dropout, device);
            LinearImpl f1 = new LinearImpl(embedDim, ffnDim);
            LinearImpl f2 = new LinearImpl(ffnDim, embedDim);
            register_module("norm1_" + i, n1);
            register_module("attn_" + i, attn);
            register_module("norm2_" + i, n2);
            register_module("ffn1_" + i, f1);
            register_module("ffn2_" + i, f2);
            norm1.add(n1);
            attns.add(attn);
            norm2.add(n2);
            ffn1.add(f1);
            ffn2.add(f2);
        }

        LongVector fs = new LongVector(1); fs.put(0, embedDim);
        this.finalNorm = new LayerNormImpl(fs);
        register_module("final_norm", finalNorm);

        this.pool = new AdditiveAttention(embedDim, embedDim, device);
        register_module("pool", pool);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            residueEmbedding.to(dev, false);
            positionEmbedding.to(dev, false);
            for (LayerNormImpl n : norm1) n.to(dev, false);
            for (LayerNormImpl n : norm2) n.to(dev, false);
            for (LinearImpl f : ffn1) f.to(dev, false);
            for (LinearImpl f : ffn2) f.to(dev, false);
            finalNorm.to(dev, false);
        }
    }

    /**
     * @param residueIds [B, L] long amino-acid ids (0=pad), L <= maxLen
     * @return protein vector [B, embedDim]
     */
    public Tensor forward(Tensor residueIds) {
        long batch = residueIds.size(0);
        long len = residueIds.size(1);
        if (len > maxLen) {
            throw new IllegalArgumentException(
                    "sequence length " + len + " exceeds maxLen " + maxLen);
        }
        Tensor mask = residueIds.ne(new Scalar(0L)).toType(ScalarType.Float);
        Tensor pos = torch.arange(new Scalar(len),
                new TensorOptions().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Long)));
        pos = pos.unsqueeze(0L).expand(batch, len);

        Tensor h = residueEmbedding.forward(residueIds.toType(ScalarType.Long))
                .add(positionEmbedding.forward(pos));
        h = torch.dropout(h, dropoutProb, false);

        for (int i = 0; i < numLayers; i++) {
            h = h.add(attns.get(i).forward(norm1.get(i).forward(h), mask));
            Tensor f = ffn2.get(i).forward(
                    torch.dropout(ffn1.get(i).forward(norm2.get(i).forward(h)).relu(),
                            dropoutProb, false));
            h = h.add(f);
        }
        h = finalNorm.forward(h);
        return pool.forward(h, mask);
    }

    /** Token-level contextual embeddings [B, L, D] (no pooling). */
    public Tensor forwardTokens(Tensor residueIds) {
        long batch = residueIds.size(0);
        long len = residueIds.size(1);
        Tensor mask = residueIds.ne(new Scalar(0L)).toType(ScalarType.Float);
        Tensor pos = torch.arange(new Scalar(len),
                new TensorOptions().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Long)));
        pos = pos.unsqueeze(0L).expand(batch, len);
        Tensor h = residueEmbedding.forward(residueIds.toType(ScalarType.Long))
                .add(positionEmbedding.forward(pos));
        for (int i = 0; i < numLayers; i++) {
            h = h.add(attns.get(i).forward(norm1.get(i).forward(h), mask));
            Tensor f = ffn2.get(i).forward(
                    torch.dropout(ffn1.get(i).forward(norm2.get(i).forward(h)).relu(),
                            dropoutProb, false));
            h = h.add(f);
        }
        return finalNorm.forward(h);
    }

    public int embedDim() {
        return embedDim;
    }

    public int maxLen() {
        return maxLen;
    }
}
