/*
 * Multi-Head Self-Attention block used by Microsoft NRMS news encoder / user encoder.
 *
 * Reference:
 *   Wu et al., "Neural News Recommendation with Multi-Head Self-Attention", EMNLP 2019
 *   (Microsoft MIND dataset companion model)
 *
 * Differs slightly from generic MHA by using the additive-attention pooling pattern
 * common in the news-rec literature after self-attention.
 */
package org.bytedeco.pytorch.recommend.basic.layers.industry;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MultiHeadSelfAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final LinearImpl qProj;
    private final LinearImpl kProj;
    private final LinearImpl vProj;
    private final LinearImpl outProj;
    private final DropoutImpl dropout;

    public MultiHeadSelfAttention(int embedDim, int numHeads) {
        this(embedDim, numHeads, 0.1f, DeviceSupport.backend());
    }

    public MultiHeadSelfAttention(int embedDim, int numHeads, float dropoutProb, String device) {
        super("MultiHeadSelfAttention");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException(
                    "embedDim (" + embedDim + ") must be divisible by numHeads (" + numHeads + ")");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        this.qProj = new LinearImpl(embedDim, embedDim);
        this.kProj = new LinearImpl(embedDim, embedDim);
        this.vProj = new LinearImpl(embedDim, embedDim);
        this.outProj = new LinearImpl(embedDim, embedDim);
        this.dropout = new DropoutImpl(dropoutProb);

        register_module("q_proj", qProj);
        register_module("k_proj", kProj);
        register_module("v_proj", vProj);
        register_module("out_proj", outProj);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            qProj.to(dev, false);
            kProj.to(dev, false);
            vProj.to(dev, false);
            outProj.to(dev, false);
        }
    }

    /**
     * @param x    [B, L, D]
     * @param mask optional [B, L] 1=valid 0=pad; applied as additive -inf on scores
     * @return [B, L, D]
     */
    public Tensor forward(Tensor x, Tensor mask) {
        long batch = x.size(0);
        long len = x.size(1);

        Tensor q = qProj.forward(x).view(batch, len, numHeads, headDim).transpose(1, 2);
        Tensor k = kProj.forward(x).view(batch, len, numHeads, headDim).transpose(1, 2);
        Tensor v = vProj.forward(x).view(batch, len, numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(q, k.transpose(2, 3)).div(scale);

        if (mask != null && !mask.isNull() && mask.numel() > 0) {
            // mask [B, L] -> [B, 1, 1, L]
            Tensor m = mask.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float)
                    .unsqueeze(1L).unsqueeze(2L);
            Tensor neg = torch.full_like(scores, new Scalar(-1e9f));
            scores = scores.mul(m).add(neg.mul(torch.sub(torch.ones_like(m), m)));
        }

        Tensor attn = dropout.forward(scores.softmax(-1));
        Tensor ctx = torch.matmul(attn, v); // [B, H, L, Dh]
        Tensor merged = ctx.transpose(1, 2).contiguous().view(batch, len, embedDim);
        return outProj.forward(merged);
    }

    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }

    public int embedDim() {
        return embedDim;
    }

    public int numHeads() {
        return numHeads;
    }
}
