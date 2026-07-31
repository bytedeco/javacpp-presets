/*
 * Ported from torch-rechub-scala: torchrec/models/matching/SASRec.scala
 *
 * SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, ICDM'2018).
 * Reference: https://arxiv.org/pdf/1808.09781.pdf
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
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
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SASRec extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String seqFeatureName;
    private final int embedDim;
    private final int numHeads;
    private final int numLayers;
    private final int headDim;
    private final float dropout;

    private final EmbeddingImpl itemEmbedding;
    private final EmbeddingImpl positionEmbedding;
    private final List<LayerNormImpl> attnLayerNorms = new ArrayList<>();
    private final List<LinearImpl> attnQProjs = new ArrayList<>();
    private final List<LinearImpl> attnKProjs = new ArrayList<>();
    private final List<LinearImpl> attnVProjs = new ArrayList<>();
    private final List<LinearImpl> attnOProjs = new ArrayList<>();
    private final List<LayerNormImpl> fwdLayerNorms = new ArrayList<>();
    private final List<PointWiseFeedForward> fwdLayers = new ArrayList<>();
    private final LayerNormImpl lastLayerNorm;
    private final LinearImpl outputProj;

    public SASRec(List<? extends Feature> sequenceFeatures) {
        this(sequenceFeatures, 8, 2, 2, 128, 0.2f, DeviceSupport.backend());
    }

    public SASRec(List<? extends Feature> sequenceFeatures, int embedDim, int numHeads,
                  int numLayers, int ffnDim, float dropout, String device) {
        super("SASRec");
        if (sequenceFeatures == null || sequenceFeatures.isEmpty()) {
            throw new IllegalArgumentException("sequenceFeatures cannot be empty");
        }
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException(
                    "embedDim (" + embedDim + ") must be divisible by numHeads (" + numHeads + ")");
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be > 0, got " + numLayers);
        }

        Feature head = sequenceFeatures.get(0);
        if (!(head instanceof SequenceFeature)) {
            throw new IllegalArgumentException(
                    "SASRec expects a SequenceFeature, got: " + head.getClass().getSimpleName());
        }
        SequenceFeature seqFeature = (SequenceFeature) head;
        this.seqFeatureName = seqFeature.name();
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.headDim = embedDim / numHeads;
        this.dropout = dropout;

        long vocabSize = seqFeature.vocabSize();
        int maxLen = seqFeature.maxLen();

        EmbeddingOptions itemOpts = new EmbeddingOptions(vocabSize, embedDim);
        itemOpts.padding_idx().put(new LongOptional(0L));
        this.itemEmbedding = new EmbeddingImpl(itemOpts);
        register_module("item_embedding", itemEmbedding);

        this.positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxLen, embedDim));
        register_module("position_embedding", positionEmbedding);

        for (int i = 0; i < numLayers; i++) {
            LongVector normShape = new LongVector(1);
            normShape.put(0, embedDim);

            LayerNormImpl attnLn = new LayerNormImpl(normShape);
            register_module("attn_layer_norm_" + i, attnLn);
            attnLayerNorms.add(attnLn);

            LinearImpl q = new LinearImpl(embedDim, embedDim);
            register_module("q_proj_" + i, q);
            attnQProjs.add(q);
            LinearImpl k = new LinearImpl(embedDim, embedDim);
            register_module("k_proj_" + i, k);
            attnKProjs.add(k);
            LinearImpl v = new LinearImpl(embedDim, embedDim);
            register_module("v_proj_" + i, v);
            attnVProjs.add(v);
            LinearImpl o = new LinearImpl(embedDim, embedDim);
            register_module("o_proj_" + i, o);
            attnOProjs.add(o);

            LongVector fwdNormShape = new LongVector(1);
            fwdNormShape.put(0, embedDim);
            LayerNormImpl fwdLn = new LayerNormImpl(fwdNormShape);
            register_module("fwd_layer_norm_" + i, fwdLn);
            fwdLayerNorms.add(fwdLn);

            PointWiseFeedForward ffn = new PointWiseFeedForward(embedDim, ffnDim, dropout, device);
            register_module("fwd_layer_" + i, ffn);
            fwdLayers.add(ffn);
        }

        LongVector lastNormShape = new LongVector(1);
        lastNormShape.put(0, embedDim);
        this.lastLayerNorm = new LayerNormImpl(lastNormShape);
        register_module("last_layer_norm", lastLayerNorm);

        this.outputProj = new LinearImpl(embedDim, 1L);
        register_module("output", outputProj);
    }

    public String seqFeatureName() {
        return seqFeatureName;
    }

    @Override
    public Tensor forward(Tensor sequence) {
        Tensor seq = sequence.toType(ScalarType.Long);
        long batch = seq.size(0);
        long len = seq.size(1);

        Tensor itemEmb = itemEmbedding.forward(seq);

        Tensor posIds = torch.arange(new Scalar(len),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        Tensor posEmb = positionEmbedding.forward(posIds).unsqueeze(0L);

        float scale = (float) Math.sqrt(embedDim);
        Tensor hidden = itemEmb.mul(new Scalar(scale)).add(posEmb);
        hidden = torch.dropout(hidden, dropout, false);

        Tensor padScalar = torch.zeros(new long[]{1L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        Tensor keyMask = seq.ne(padScalar);
        Tensor keyMaskF = keyMask.toType(ScalarType.Float);

        // Causal mask: True where allowed (lower triangle including diagonal)
        Tensor causalMask = torch.triu(
                torch.ones(new long[]{len, len},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))), 1L)
                .eq(new Scalar(0.0));

        for (int layer = 0; layer < numLayers; layer++) {
            hidden = attentionBlock(hidden, keyMaskF, causalMask, batch, len, layer);
        }

        hidden = lastLayerNorm.forward(hidden);
        Tensor maskExpanded = keyMaskF.unsqueeze(2L);
        Tensor summed = hidden.mul(maskExpanded).sum(1L);
        Tensor counts = keyMaskF.sum(1L).clamp_min(new Scalar(1.0)).unsqueeze(1L);
        Tensor pooled = summed.div(counts);

        return outputProj.forward(pooled);
    }

    private Tensor attentionBlock(Tensor input, Tensor keyMaskF, Tensor causalMask,
                                  long batch, long len, int layer) {
        Tensor qNorm = attnLayerNorms.get(layer).forward(input);
        Tensor q = attnQProjs.get(layer).forward(qNorm);
        Tensor k = attnKProjs.get(layer).forward(input);
        Tensor v = attnVProjs.get(layer).forward(input);

        Tensor qh = q.reshape(batch, len, numHeads, headDim).transpose(1L, 2L);
        Tensor kh = k.reshape(batch, len, numHeads, headDim).transpose(1L, 2L);
        Tensor vh = v.reshape(batch, len, numHeads, headDim).transpose(1L, 2L);

        Scalar scaleAttn = new Scalar((float) (1.0 / Math.sqrt(headDim)));
        Tensor scores = torch.matmul(qh, kh.transpose(-2L, -1L)).mul(scaleAttn);

        Tensor causalBroad = causalMask.reshape(1L, 1L, len, len).toType(ScalarType.Float);
        Tensor keyBroad = keyMaskF.reshape(batch, 1L, 1L, len);
        Tensor combined = causalBroad.mul(keyBroad);

        Tensor negInf = torch.full(new long[]{1L}, new Scalar(-1e9))
                .to(input.device(), ScalarType.Float);
        Tensor maskedScores = scores.mul(combined).add(
                combined.mul(new Scalar(-1.0)).add(new Scalar(1.0)).mul(negInf));

        Tensor attn = torch.softmax(maskedScores, -1L);
        Tensor context = torch.matmul(attn, vh);

        Tensor merged = context.transpose(1L, 2L).contiguous().reshape(batch, len, embedDim);
        Tensor attnOut = attnOProjs.get(layer).forward(merged);

        Tensor res1 = input.add(attnOut);
        Tensor res1Norm = fwdLayerNorms.get(layer).forward(res1);
        Tensor ffOut = fwdLayers.get(layer).forward(res1Norm);
        return res1.add(ffOut);
    }
}
