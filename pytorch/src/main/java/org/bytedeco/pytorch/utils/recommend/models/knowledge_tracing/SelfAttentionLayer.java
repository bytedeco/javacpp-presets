/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/ATDKT.scala
 * (SelfAttentionLayer)
 *
 * Self-Attention layer for capturing important historical patterns in AT-DKT.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SelfAttentionLayer extends Module {

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
    private final LayerNormImpl ln;

    public SelfAttentionLayer(int embedDim, int numHeads) {
        this(embedDim, numHeads, 0.1f, DeviceSupport.backend());
    }

    public SelfAttentionLayer(int embedDim, int numHeads, float dropout, String device) {
        super("SelfAttentionLayer");
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

        LongVector shape = new LongVector(1);
        shape.put(0, embedDim);
        this.ln = new LayerNormImpl(new LayerNormOptions(shape));

        register_module("q_linear", qLinear);
        register_module("k_linear", kLinear);
        register_module("v_linear", vLinear);
        register_module("out_linear", outLinear);
        register_module("ln", ln);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qLinear.to(dev, false);
            kLinear.to(dev, false);
            vLinear.to(dev, false);
            outLinear.to(dev, false);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);

        Tensor q = qLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor k = kLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor v = vLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(q, k.transpose(2, 3)).div(scale);

        if (seqLen > 1) {
            Tensor causalMask = torch.triu(
                    torch.ones(new long[]{seqLen, seqLen},
                            new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))),
                    1).unsqueeze(0).unsqueeze(0);
            scores = scores.add(causalMask.mul(new Scalar(1e9)));
        }

        Tensor attnWeights = scores.softmax(-1);
        Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), v);
        Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
        Tensor out = outLinear.forward(reshaped);
        return ln.forward(x.add(out));
    }
}
