/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Attention.scala
 *
 * Cone-shaped attention for CSKT.
 * Uses geometric distance penalty in attention scores.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ConeAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final float r;
    private final float gamma;
    private final String device;
    private final LinearImpl qLinear;
    private final LinearImpl kLinear;
    private final LinearImpl vLinear;
    private final LinearImpl outLinear;
    private final DropoutImpl dropoutLayer;

    public ConeAttention(int embedDim, int numHeads) {
        this(embedDim, numHeads, 1.0f, 1.0f, 0.1f, DeviceSupport.backend());
    }

    public ConeAttention(
            int embedDim,
            int numHeads,
            float r,
            float gamma,
            float dropout,
            String device) {
        super("ConeAttention");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.r = r;
        this.gamma = gamma;
        this.device = device;

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

    /**
     * @param q    query (batch, seq, embedDim)
     * @param k    key/value source (batch, seq, embedDim)
     * @param mask 0 = causal, 1 = no causal
     */
    public Tensor forward(Tensor q, Tensor k, int mask) {
        int batchSize = (int) q.size(0);
        int seqLen = (int) q.size(1);

        Tensor qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor kProj = kLinear.forward(k).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor vProj = vLinear.forward(k).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale);

        if (seqLen > 1) {
            Tensor posIds = torch.arange(
                    new Scalar(0),
                    new Scalar(seqLen),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            if (!"cpu".equals(device)) {
                posIds = posIds.to(new Device(device), ScalarType.Float);
            }
            Tensor posDiff = posIds.view(seqLen, 1L).sub(posIds.view(1L, seqLen)).abs();
            // Cone penalty: tanh(r * d_ij) * gamma
            Tensor conePenalty = torch.tanh(posDiff.mul(new Scalar((float) r))).mul(new Scalar((float) gamma));
            Tensor coneBias = conePenalty.view(1L, 1L, (long) seqLen, (long) seqLen);
            scores = scores.add(coneBias);

            if (mask == 0) {
                Tensor causalMask = torch.triu(
                        torch.ones(new long[]{seqLen, seqLen},
                                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))),
                        1).unsqueeze(0).unsqueeze(0);
                scores = scores.add(causalMask.mul(new Scalar(1e9)));
            }
        }

        Tensor attnWeights = scores.softmax(-1);
        Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj);
        Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
        return outLinear.forward(reshaped);
    }

    public Tensor forward(Tensor q, Tensor k) {
        return forward(q, k, 0);
    }
}
