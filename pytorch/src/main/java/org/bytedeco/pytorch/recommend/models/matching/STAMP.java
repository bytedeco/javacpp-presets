/*
 * Ported from torch-rechub-scala: torchrec/models/matching/STAMP.scala
 *
 * STAMP — Short-Term Attention/Memory Priority Model (CIKM'2018).
 * Reference: https://dl.acm.org/doi/10.1145/3219819.3219950
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.TanhImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;

import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class STAMP extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String seqFeatureName;
    private final int embedDim;
    private final EmbeddingImpl itemEmbedding;
    private final Tensor w0, w1T, w2T, w3T, bA;
    private final SequentialImpl fS;
    private final SequentialImpl fT;

    public STAMP(List<? extends Feature> features) {
        this(features, 8, 0.1f, 0.05f, -1, DeviceSupport.backend());
    }

    public STAMP(List<? extends Feature> features, int embedDim, int attentionDim, String device) {
        this(features, embedDim, 0.1f, 0.05f, attentionDim, device);
    }

    public STAMP(List<? extends Feature> features, int embedDim, float weightStd, float embStd,
                 int attentionDim, String device) {
        super("STAMP");
        // attentionDim kept for API parity
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("STAMP: features cannot be empty");
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("STAMP: embedDim must be > 0, got " + embedDim);
        }
        this.embedDim = embedDim;

        Feature head = features.get(0);
        if (!(head instanceof SequenceFeature)) {
            throw new IllegalArgumentException(
                    "STAMP expects a SequenceFeature as the first feature, got: "
                            + head.getClass().getSimpleName());
        }
        SequenceFeature seqFeature = (SequenceFeature) head;
        this.seqFeatureName = seqFeature.name();
        long vocabSize = seqFeature.vocabSize();

        EmbeddingOptions opts = new EmbeddingOptions(vocabSize, embedDim);
        opts.padding_idx().put(new LongOptional(0L));
        this.itemEmbedding = new EmbeddingImpl(opts);
        torch.normal_(itemEmbedding.weight(), 0.0, embStd);
        register_module("item_emb", itemEmbedding);

        TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));

        this.w0 = torch.zeros(new long[]{embedDim, 1L}, fOpts);
        torch.normal_(w0, 0.0, weightStd);
        register_parameter("w_0", w0);

        this.w1T = torch.zeros(new long[]{embedDim, embedDim}, fOpts);
        torch.normal_(w1T, 0.0, weightStd);
        register_parameter("w_1_t", w1T);

        this.w2T = torch.zeros(new long[]{embedDim, embedDim}, fOpts);
        torch.normal_(w2T, 0.0, weightStd);
        register_parameter("w_2_t", w2T);

        this.w3T = torch.zeros(new long[]{embedDim, embedDim}, fOpts);
        torch.normal_(w3T, 0.0, weightStd);
        register_parameter("w_3_t", w3T);

        this.bA = torch.zeros(new long[]{embedDim}, fOpts);
        register_parameter("b_a", bA);

        this.fS = new SequentialImpl();
        fS.push_back("tanh_s", new TanhImpl());
        fS.push_back("linear_s", new LinearImpl(embedDim, embedDim));
        register_module("f_s", fS);

        this.fT = new SequentialImpl();
        fT.push_back("tanh_t", new TanhImpl());
        fT.push_back("linear_t", new LinearImpl(embedDim, embedDim));
        register_module("f_t", fT);
    }

    public String seqFeatureName() {
        return seqFeatureName;
    }

    @Override
    public Tensor forward(Tensor sequence) {
        Tensor input = sequence.toType(ScalarType.Long);

        Tensor valueMask = input.ne(new Scalar(0L)).unsqueeze(-1L);
        Tensor valueMaskF = valueMask.toType(ScalarType.Float);
        Tensor valueCounts = valueMaskF.sum(1L).unsqueeze(-1L);
        Tensor itemEmbBatch = itemEmbedding.forward(input).mul(valueMaskF);

        // x_t: embedding of the last valid token per row
        Tensor lastIdx = valueCounts.squeeze(-1L).sub(new Scalar(1.0f)).toType(ScalarType.Long);
        Tensor xT = itemEmbedding.forward(torch.gather(input, 1L, lastIdx)).squeeze(1L);

        // m_s: mean-pooled, masked history
        Tensor mS = itemEmbBatch.sum(new long[]{1L}).div(valueCounts.squeeze(1L)).unsqueeze(1L);

        // a = normalize(sigmoid(... ) @ w_0) * value_mask
        Tensor preSigmoid = torch.matmul(itemEmbBatch, w1T)
                .add(torch.matmul(xT.unsqueeze(1L), w2T))
                .add(torch.matmul(mS, w3T))
                .add(bA);
        Tensor preAttn = preSigmoid.sigmoid().matmul(w0).mul(valueMaskF);
        Tensor a = preAttn.div(
                preAttn.sum(new long[]{1L}).clamp_min(new Scalar(1e-9)).unsqueeze(-1L));

        // m_a = (a * item_emb).sum(1) + m_s
        Tensor mA = torch.mul(a, itemEmbBatch).sum(1L).add(mS.squeeze(1L));

        Tensor hS = fS.forward(mA);
        Tensor hT = fT.forward(xT);
        return hS.mul(hT);
    }
}
