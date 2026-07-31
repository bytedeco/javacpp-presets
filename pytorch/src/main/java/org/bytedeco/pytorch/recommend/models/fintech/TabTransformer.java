/*
 * TabTransformer — Transformer for tabular data (fintech / credit / fraud).
 *
 * Reference:
 *   Huang et al., "TabTransformer: Tabular Data Modeling Using Contextual
 *   Embeddings", arXiv 2020 (Amazon AWS AI).
 *   https://arxiv.org/abs/2012.06678
 *
 * Industrial use:
 *   Credit scoring, card fraud, anti-money-laundering tabular features where
 *   categorical columns dominate. Contextual embeddings via Transformer on
 *   categorical column tokens; continuous features pass through a separate LN+MLP
 *   and concatenate before the final MLP head.
 *
 * Input:
 *   categoricalIds: [B, numCat] long  (each column has its own vocab; packed via offsets)
 *   continuous:     [B, numCont] float (may be empty)
 * Output:
 *   probability [B] (binary classification default)
 */
package org.bytedeco.pytorch.recommend.models.fintech;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.industry.MultiHeadSelfAttention;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TabTransformer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<EmbeddingImpl> catEmbeddings = new ArrayList<>();
    private final List<MultiHeadSelfAttention> encoderLayers = new ArrayList<>();
    private final List<LayerNormImpl> encoderNorms = new ArrayList<>();
    private final LayerNormImpl contNorm;
    private final MLP head;
    private final int embedDim;
    private final int numCat;
    private final int numCont;
    private final int numLayers;

    /**
     * @param catVocabSizes vocabulary size per categorical column
     * @param numContinuous number of continuous columns
     */
    public TabTransformer(int[] catVocabSizes, int numContinuous) {
        this(catVocabSizes, numContinuous, 32, 8, 4, new long[]{128L, 64L},
                0.1f, DeviceSupport.backend());
    }

    public TabTransformer(int[] catVocabSizes, int numContinuous, int embedDim,
                          int numHeads, int numLayers, long[] mlpHidden,
                          float dropout, String device) {
        super("TabTransformer");
        if (catVocabSizes == null || catVocabSizes.length == 0) {
            throw new IllegalArgumentException("At least one categorical column required");
        }
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numCat = catVocabSizes.length;
        this.numCont = Math.max(numContinuous, 0);
        this.numLayers = numLayers;

        for (int i = 0; i < numCat; i++) {
            EmbeddingOptions opts = new EmbeddingOptions(Math.max(catVocabSizes[i], 2), embedDim);
            opts.padding_idx().put(new LongOptional(0L));
            EmbeddingImpl emb = new EmbeddingImpl(opts);
            register_module("cat_emb_" + i, emb);
            catEmbeddings.add(emb);
        }

        for (int i = 0; i < numLayers; i++) {
            MultiHeadSelfAttention attn = new MultiHeadSelfAttention(embedDim, numHeads, dropout, device);
            register_module("encoder_attn_" + i, attn);
            encoderLayers.add(attn);

            LongVector shape = new LongVector(1);
            shape.put(0, embedDim);
            LayerNormImpl ln = new LayerNormImpl(shape);
            register_module("encoder_norm_" + i, ln);
            encoderNorms.add(ln);
        }

        if (numCont > 0) {
            LongVector cShape = new LongVector(1);
            cShape.put(0, numCont);
            this.contNorm = new LayerNormImpl(cShape);
            register_module("cont_norm", contNorm);
        } else {
            this.contNorm = null;
        }

        long headIn = (long) numCat * embedDim + numCont;
        this.head = new MLP(headIn, mlpHidden, 1L, "relu", dropout, false, false, true, device);
        register_module("head", head);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            for (EmbeddingImpl e : catEmbeddings) e.to(dev, false);
            for (LayerNormImpl n : encoderNorms) n.to(dev, false);
            if (contNorm != null) contNorm.to(dev, false);
        }
    }

    /**
     * @param categoricalIds [B, numCat] long
     * @param continuous     [B, numCont] float or null
     * @return probability [B]
     */
    public Tensor forward(Tensor categoricalIds, Tensor continuous) {
        long batch = categoricalIds.size(0);
        // Embed each column -> stack as sequence [B, numCat, D]
        TensorVector cols = new TensorVector();
        Tensor ids = categoricalIds.toType(org.bytedeco.pytorch.global.torch.ScalarType.Long);
        for (int i = 0; i < numCat; i++) {
            Tensor col = ids.select(1L, i); // [B]
            cols.push_back(catEmbeddings.get(i).forward(col).unsqueeze(1L));
        }
        Tensor seq = torch.cat(cols, 1L); // [B, numCat, D]

        // Transformer encoder (Pre-LN residual)
        Tensor h = seq;
        for (int i = 0; i < numLayers; i++) {
            Tensor normed = encoderNorms.get(i).forward(h);
            Tensor attn = encoderLayers.get(i).forward(normed);
            h = h.add(attn);
        }

        // Flatten contextual cat embeddings
        Tensor catFlat = h.contiguous().view(batch, (long) numCat * embedDim);

        Tensor fused;
        if (numCont > 0 && continuous != null && !continuous.isNull()) {
            Tensor c = contNorm.forward(continuous);
            TensorVector cat = new TensorVector();
            cat.push_back(catFlat);
            cat.push_back(c);
            fused = torch.cat(cat, 1L);
        } else {
            fused = catFlat;
        }
        return head.forward(fused).squeeze(1L).sigmoid();
    }

    /** Logits without sigmoid (for BCE-with-logits). */
    public Tensor forwardLogits(Tensor categoricalIds, Tensor continuous) {
        long batch = categoricalIds.size(0);
        TensorVector cols = new TensorVector();
        Tensor ids = categoricalIds.toType(org.bytedeco.pytorch.global.torch.ScalarType.Long);
        for (int i = 0; i < numCat; i++) {
            cols.push_back(catEmbeddings.get(i).forward(ids.select(1L, i)).unsqueeze(1L));
        }
        Tensor h = torch.cat(cols, 1L);
        for (int i = 0; i < numLayers; i++) {
            h = h.add(encoderLayers.get(i).forward(encoderNorms.get(i).forward(h)));
        }
        Tensor catFlat = h.contiguous().view(batch, (long) numCat * embedDim);
        Tensor fused = catFlat;
        if (numCont > 0 && continuous != null && !continuous.isNull()) {
            TensorVector cat = new TensorVector();
            cat.push_back(catFlat);
            cat.push_back(contNorm.forward(continuous));
            fused = torch.cat(cat, 1L);
        }
        return head.forward(fused).squeeze(1L);
    }

    public int numCat() { return numCat; }
    public int numCont() { return numCont; }
    public int embedDim() { return embedDim; }
}
