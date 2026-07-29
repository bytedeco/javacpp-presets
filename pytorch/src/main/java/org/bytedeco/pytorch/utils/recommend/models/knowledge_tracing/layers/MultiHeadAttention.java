/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Attention.scala
 *
 * Standard Multi-head self-attention.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MultiHeadAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final LinearImpl qLinear;
    private final LinearImpl kLinear;
    private final LinearImpl vLinear;
    private final LinearImpl outLinear;
    private final DropoutImpl dropoutLayer;

    public MultiHeadAttention(int embedDim, int numHeads) {
        this(embedDim, numHeads, 0.1f, DeviceSupport.backend());
    }

    public MultiHeadAttention(int embedDim, int numHeads, float dropout, String device) {
        super("MultiHeadAttention");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        this.qLinear = new LinearImpl(embedDim, embedDim);
        this.kLinear = new LinearImpl(embedDim, embedDim);
        this.vLinear = new LinearImpl(embedDim, embedDim);
        this.outLinear = new LinearImpl(embedDim, embedDim);
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("q_linear", qLinear);
        register_module("k_linear", kLinear);
        register_module("v_linear", vLinear);
        register_module("out_linear", outLinear);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qLinear.to(dev, false);
            kLinear.to(dev, false);
            vLinear.to(dev, false);
            outLinear.to(dev, false);
        }
    }

    public Tensor forward(Tensor q, Tensor k, Tensor v, Tensor mask) {
        int batchSize = (int) q.size(0);
        int seqLen = (int) q.size(1);

        Tensor qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor kProj = kLinear.forward(k).view(batchSize, (int) k.size(1), numHeads, headDim).transpose(1, 2);
        Tensor vProj = vLinear.forward(v).view(batchSize, (int) v.size(1), numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale);

        if (mask != null && !mask.isNull() && mask.numel() > 0) {
            scores = scores.add(mask);
        }

        Tensor attnWeights = scores.softmax(-1);
        Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj);
        Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
        return outLinear.forward(reshaped);
    }

    public Tensor forward(Tensor q, Tensor k, Tensor v) {
        return forward(q, k, v, null);
    }
}
