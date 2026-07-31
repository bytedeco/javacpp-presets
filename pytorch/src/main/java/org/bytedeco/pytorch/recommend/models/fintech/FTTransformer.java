/*
 * FT-Transformer — Feature Tokenizer + Transformer for tabular data.
 *
 * Reference:
 *   Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
 *   NeurIPS 2021. https://arxiv.org/abs/2106.11959
 *   Official: https://github.com/Yura52/tabular-dl-revisiting-models
 *
 * Key difference vs TabTransformer:
 *   - BOTH categorical AND continuous features are tokenized to the same
 *     embedding space (continuous: x * W + b per feature; categorical: embedding).
 *   - A [CLS] token is prepended; final representation is the [CLS] output
 *     after stacked Transformer blocks (Pre-Norm).
 *
 * Strong baseline for fintech risk / fraud tabular modeling when feature
 * count is moderate (tens to low hundreds).
 */
package org.bytedeco.pytorch.recommend.models.fintech;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.industry.MultiHeadSelfAttention;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FTTransformer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<EmbeddingImpl> catEmbeddings = new ArrayList<>();
    private final List<LinearImpl> contTokenizers = new ArrayList<>(); // each: 1 -> embedDim
    private final EmbeddingImpl clsToken; // learned [CLS]
    private final List<MultiHeadSelfAttention> attns = new ArrayList<>();
    private final List<LayerNormImpl> norm1 = new ArrayList<>();
    private final List<LayerNormImpl> norm2 = new ArrayList<>();
    private final List<LinearImpl> ffn1 = new ArrayList<>();
    private final List<LinearImpl> ffn2 = new ArrayList<>();
    private final LayerNormImpl finalNorm;
    private final LinearImpl head;
    private final int embedDim;
    private final int numCat;
    private final int numCont;
    private final int numLayers;
    private final float dropoutProb;
    private final String deviceName;

    public FTTransformer(int[] catVocabSizes, int numContinuous) {
        this(catVocabSizes, numContinuous, 64, 8, 3, 128, 0.1f, DeviceSupport.backend());
    }

    public FTTransformer(int[] catVocabSizes, int numContinuous, int embedDim,
                         int numHeads, int numLayers, int ffnDim, float dropout, String device) {
        super("FTTransformer");
        this.embedDim = embedDim;
        this.numCat = catVocabSizes != null ? catVocabSizes.length : 0;
        this.numCont = Math.max(numContinuous, 0);
        this.numLayers = numLayers;
        this.dropoutProb = dropout;
        this.deviceName = device != null ? device : "cpu";

        if (numCat == 0 && numCont == 0) {
            throw new IllegalArgumentException("FTTransformer needs at least one feature");
        }
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        for (int i = 0; i < numCat; i++) {
            EmbeddingOptions opts = new EmbeddingOptions(Math.max(catVocabSizes[i], 2), embedDim);
            opts.padding_idx().put(new LongOptional(0L));
            EmbeddingImpl emb = new EmbeddingImpl(opts);
            register_module("cat_tok_" + i, emb);
            catEmbeddings.add(emb);
        }
        for (int i = 0; i < numCont; i++) {
            LinearImpl tok = new LinearImpl(1L, embedDim);
            register_module("cont_tok_" + i, tok);
            contTokenizers.add(tok);
        }

        // CLS token as embedding of a single index
        this.clsToken = new EmbeddingImpl(new EmbeddingOptions(1, embedDim));
        register_module("cls_token", clsToken);

        for (int i = 0; i < numLayers; i++) {
            LongVector s1 = new LongVector(1); s1.put(0, embedDim);
            LongVector s2 = new LongVector(1); s2.put(0, embedDim);
            LayerNormImpl n1 = new LayerNormImpl(s1);
            LayerNormImpl n2 = new LayerNormImpl(s2);
            MultiHeadSelfAttention attn = new MultiHeadSelfAttention(embedDim, numHeads, dropout, device);
            // Position-wise FFN without built-in residual (residual applied in forward)
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
        this.head = new LinearImpl(embedDim, 1L);
        register_module("head", head);

        if (!"cpu".equals(this.deviceName)) {
            Device dev = new Device(this.deviceName);
            for (EmbeddingImpl e : catEmbeddings) e.to(dev, false);
            for (LinearImpl t : contTokenizers) t.to(dev, false);
            clsToken.to(dev, false);
            for (LayerNormImpl n : norm1) n.to(dev, false);
            for (LayerNormImpl n : norm2) n.to(dev, false);
            for (LinearImpl f : ffn1) f.to(dev, false);
            for (LinearImpl f : ffn2) f.to(dev, false);
            finalNorm.to(dev, false);
            head.to(dev, false);
        }
    }

    /**
     * @param categoricalIds [B, numCat] or null if numCat==0
     * @param continuous     [B, numCont] or null if numCont==0
     * @return probability [B]
     */
    public Tensor forward(Tensor categoricalIds, Tensor continuous) {
        return forwardLogits(categoricalIds, continuous).sigmoid();
    }

    public Tensor forwardLogits(Tensor categoricalIds, Tensor continuous) {
        long batch;
        if (numCat > 0) {
            batch = categoricalIds.size(0);
        } else {
            batch = continuous.size(0);
        }

        List<Tensor> tokens = new ArrayList<>();
        // CLS
        Tensor clsIdx = torch.zeros(new long[]{batch},
                new TensorOptions().dtype(new org.bytedeco.pytorch.ScalarTypeOptional(ScalarType.Long)));
        if (!"cpu".equals(deviceName)) {
            clsIdx = clsIdx.to(new Device(deviceName), ScalarType.Long);
        }
        tokens.add(clsToken.forward(clsIdx).unsqueeze(1L)); // [B,1,D]

        if (numCat > 0 && categoricalIds != null) {
            Tensor ids = categoricalIds.toType(ScalarType.Long);
            for (int i = 0; i < numCat; i++) {
                tokens.add(catEmbeddings.get(i).forward(ids.select(1L, i)).unsqueeze(1L));
            }
        }
        if (numCont > 0 && continuous != null) {
            for (int i = 0; i < numCont; i++) {
                Tensor col = continuous.select(1L, i).unsqueeze(1L); // [B,1]
                tokens.add(contTokenizers.get(i).forward(col).unsqueeze(1L)); // [B,1,D]
            }
        }

        TensorVector vec = new TensorVector();
        for (Tensor t : tokens) vec.push_back(t);
        Tensor h = torch.cat(vec, 1L); // [B, 1+F, D]

        for (int i = 0; i < numLayers; i++) {
            Tensor a = attns.get(i).forward(norm1.get(i).forward(h));
            h = h.add(a);
            Tensor f = ffn2.get(i).forward(
                    torch.dropout(ffn1.get(i).forward(norm2.get(i).forward(h)).relu(),
                            dropoutProb, false));
            h = h.add(f);
        }
        h = finalNorm.forward(h);
        Tensor cls = h.select(1L, 0L); // [B, D]
        return head.forward(cls).squeeze(1L);
    }
}
