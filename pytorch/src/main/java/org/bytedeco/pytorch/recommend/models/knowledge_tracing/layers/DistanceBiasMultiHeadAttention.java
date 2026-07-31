/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Attention.scala
 *
 * Multi-head attention with learnable distance bias.
 * Used by AKT, SimpleKT, CSKT.
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
public class DistanceBiasMultiHeadAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final String device;
    private final LinearImpl query;
    private final LinearImpl key;
    private final LinearImpl value;
    private final LinearImpl output;
    private final DropoutImpl dropoutLayer;
    private final Tensor gamma;

    public DistanceBiasMultiHeadAttention(int embedDim, int numHeads) {
        this(embedDim, numHeads, 0.1f, DeviceSupport.backend());
    }

    public DistanceBiasMultiHeadAttention(int embedDim, int numHeads, float dropout, String device) {
        super("DistanceBiasMultiHeadAttention");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.device = device;

        this.query = new LinearImpl(embedDim, embedDim);
        this.key = new LinearImpl(embedDim, embedDim);
        this.value = new LinearImpl(embedDim, embedDim);
        this.output = new LinearImpl(embedDim, embedDim);
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("query", query);
        register_module("key", key);
        register_module("value", value);
        register_module("output", output);

        Tensor gammaInit = torch.zeros(
                new long[]{1L, 1L, 1L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        gammaInit.fill_(new Scalar(0.9f));
        this.gamma = gammaInit;
        register_parameter("gamma", gamma);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            query.to(dev, false);
            key.to(dev, false);
            value.to(dev, false);
            output.to(dev, false);
        }
    }

    /**
     * @param x    (batch, seq, embedDim)
     * @param mask 1 = no causal mask, 0 = causal mask
     */
    public Tensor forward(Tensor x, int mask) {
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);

        Tensor q = query.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor k = key.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor v = value.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(q, k.transpose(2, 3)).div(scale);

        if (seqLen > 1) {
            Tensor g = gamma.sigmoid();
            Tensor posSeq = torch.arange(
                    new Scalar(0),
                    new Scalar(seqLen),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            if (!"cpu".equals(device)) {
                posSeq = posSeq.to(new Device(device), ScalarType.Float);
            }
            Tensor posIds = posSeq.view(seqLen, 1L);
            Tensor posIdsT = posSeq.view(1L, seqLen);
            Tensor distMat = posIds.sub(posIdsT).abs().toType(ScalarType.Float);
            Tensor distBias = torch.pow(g, distMat.view(1L, 1L, (long) seqLen, (long) seqLen));
            scores = scores.add(distBias);

            if (mask == 0) {
                Tensor causalMask = torch.triu(
                        torch.ones(new long[]{seqLen, seqLen},
                                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))),
                        1).unsqueeze(0).unsqueeze(0);
                scores = scores.add(causalMask.mul(new Scalar(1e9)));
            }
        }

        Tensor attnWeights = scores.softmax(-1);
        Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), v);
        Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
        return output.forward(reshaped);
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, 1);
    }
}
