/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/StableKT.scala
 * (StableTransformerBlock)
 *
 * StableTransformerBlock with ALiBi bias and penumbral attention.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.TensorHelpers;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StableTransformerBlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final int halfHeads;
    private final float r;
    private final float gamma;
    private final LinearImpl qLinear;
    private final LinearImpl kLinear;
    private final LinearImpl vLinear;
    private final LinearImpl outLinear;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;
    private final LinearImpl ff1;
    private final LinearImpl ff2;
    private final DropoutImpl dropoutLayer;
    private final Tensor slopes;

    public StableTransformerBlock(int embedDim, int numHeads) {
        this(embedDim, numHeads, 1.0f, 1.0f, 0.1f, DeviceSupport.backend());
    }

    public StableTransformerBlock(
            int embedDim,
            int numHeads,
            float r,
            float gamma,
            float dropout,
            String device) {
        super("StableTransformerBlock");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.halfHeads = numHeads / 2;
        this.r = r;
        this.gamma = gamma;

        this.qLinear = new LinearImpl(embedDim, embedDim);
        this.kLinear = new LinearImpl(embedDim, embedDim);
        this.vLinear = new LinearImpl(embedDim, embedDim);
        register_module("q_linear", qLinear);
        register_module("k_linear", kLinear);
        register_module("v_linear", vLinear);

        this.outLinear = new LinearImpl(embedDim, embedDim);
        register_module("out_linear", outLinear);

        this.ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        register_module("ln1", ln1);
        register_module("ln2", ln2);

        this.ff1 = new LinearImpl(embedDim, embedDim * 4L);
        this.ff2 = new LinearImpl(embedDim * 4L, embedDim);
        register_module("ff1", ff1);
        register_module("ff2", ff2);

        this.dropoutLayer = new DropoutImpl(dropout);

        // ALiBi slopes
        Tensor s = torch.zeros(
                new long[]{numHeads},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        for (int h = 0; h < numHeads; h++) {
            float slope = (float) Math.pow(2, -8.0 * (h + 1) / numHeads);
            s.select(0, h).fill_(new Scalar(-slope));
        }
        this.slopes = s;
        register_parameter("slopes", slopes);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qLinear.to(dev, false);
            kLinear.to(dev, false);
            vLinear.to(dev, false);
            outLinear.to(dev, false);
            ff1.to(dev, false);
            ff2.to(dev, false);
        }
    }

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    public T_TensorTensor_T forwardPair(Tensor x, Tensor y) {
        return forwardPair(x, y, false);
    }

    public T_TensorTensor_T forwardPair(Tensor x, Tensor y, boolean useReg) {
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);

        Tensor q = qLinear.forward(x).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor k = kLinear.forward(y).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor v = vLinear.forward(y).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);

        float scaleVal = (float) Math.sqrt(headDim);
        Scalar scale = new Scalar(scaleVal);
        Tensor stdScores = torch.matmul(
                q.narrow(1, 0, halfHeads),
                k.narrow(1, 0, halfHeads).transpose(2, 3)).div(scale);

        Tensor penumbralScores = penumbralAttention(
                q.narrow(1, halfHeads, halfHeads),
                k.narrow(1, halfHeads, halfHeads),
                r, gamma);

        Tensor scores = torch.cat(new TensorVector(stdScores, penumbralScores), 1);

        Tensor alibiBias = computeAliBi(seqLen, numHeads);
        Tensor biasedScores = scores.add(alibiBias);

        Tensor finalOut;
        if (seqLen > 1) {
            Tensor causalMask = torch.triu(
                    torch.ones(new long[]{seqLen, seqLen},
                            new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float))),
                    1)
                    .mul(new Scalar(-1e9))
                    .unsqueeze(0).unsqueeze(0);
            Tensor maskedScores = biasedScores.add(causalMask);
            Tensor attnWeights = maskedScores.softmax(-1);
            Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), v);

            Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
            Tensor outProj = outLinear.forward(reshaped);

            Tensor withRes = x.add(outProj);
            Tensor normed1 = ln1.forward(withRes);

            Tensor ffOut = torch.relu(ff1.forward(normed1));
            Tensor ffDropped = dropoutLayer.forward(ffOut);
            Tensor ffResult = ff2.forward(ffDropped);

            finalOut = ln2.forward(normed1.add(ffResult));
        } else {
            Tensor attnWeights = biasedScores.softmax(-1);
            Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), v);

            Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
            Tensor outProj = outLinear.forward(reshaped);

            Tensor withRes = x.add(outProj);
            Tensor normed1 = ln1.forward(withRes);

            Tensor ffOut = torch.relu(ff1.forward(normed1));
            Tensor ffDropped = dropoutLayer.forward(ffOut);
            Tensor ffResult = ff2.forward(ffDropped);

            finalOut = ln2.forward(normed1.add(ffResult));
        }
        return new T_TensorTensor_T(finalOut, y);
    }

    private Tensor computeAliBi(int seqLen, int numHeads) {
        int total = 1 * numHeads * seqLen * seqLen;
        float[] flat = new float[total];
        for (int h = 0; h < numHeads; h++) {
            float slope = slopes.select(0, h).item().toFloat();
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    int distance = j - i;
                    float bias = slope * distance;
                    int idx = ((0 * numHeads + h) * seqLen + i) * seqLen + j;
                    flat[idx] = bias;
                }
            }
        }
        Tensor t = TensorHelpers.tensor(flat, new long[]{1L, numHeads, seqLen, seqLen});
        return t;
    }

    private Tensor penumbralAttention(Tensor q, Tensor k, float r, float gamma) {
        int headDimLocal = (int) q.size(3);

        Tensor qExpand = q.unsqueeze(3);
        Tensor kExpand = k.unsqueeze(2);
        Tensor diff = qExpand.sub(kExpand);
        Tensor pairwiseDist = torch.norm(diff, new ScalarOptional(new Scalar(2)), -1);

        Tensor penumbralScores = torch.exp(pairwiseDist.mul(new Scalar((double) (-gamma))));

        float scaleVal = (float) Math.sqrt(headDimLocal);
        return penumbralScores.div(new Scalar((double) scaleVal));
    }
}
