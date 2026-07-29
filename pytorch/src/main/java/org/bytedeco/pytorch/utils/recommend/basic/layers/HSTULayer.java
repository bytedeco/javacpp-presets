/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/HSTULayer.scala
 *
 * Hierarchical Sequential Transduction Unit layer (Meta HSTU).
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

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
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class HSTULayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int dModel;
    private final int nHeads;
    private final int dqk;
    private final int dv;
    private final int maxSeqLen;
    private final float attnAlpha;

    private final LayerNormImpl normIn;
    private final LinearImpl proj1;
    private final RelativeBucketedTimeAndPositionBias rab;
    private final LayerNormImpl normAttn;
    private final LinearImpl proj2;
    private final DropoutImpl dropoutLayer;

    public HSTULayer() {
        this(512, 8, 64, 64, 0.1f, 200, 128, "sqrt", 1.0f, "minutes", DeviceSupport.backend());
    }

    public HSTULayer(int dModel, int nHeads, int dqk, int dv, float dropout, int maxSeqLen,
                     int numTimeBuckets, String timeBucketFn, float timeBucketDivisor,
                     String timeBucketUnit, String device) {
        super("HSTULayer");
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException(
                    "d_model (" + dModel + ") must be divisible by n_heads (" + nHeads + ")");
        }
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.dqk = dqk;
        this.dv = dv;
        this.maxSeqLen = maxSeqLen;
        this.attnAlpha = 1.0f / (float) Math.sqrt(dqk);

        LongVector normInShape = new LongVector(1);
        normInShape.put(0, dModel);
        this.normIn = new LayerNormImpl(normInShape);
        register_module("norm_in", normIn);
        normIn.to(new Device(device), false);

        this.proj1 = new LinearImpl(dModel, 2L * nHeads * dqk + 2L * nHeads * dv);
        register_module("proj1", proj1);
        proj1.to(new Device(device), false);

        this.rab = new RelativeBucketedTimeAndPositionBias(
                nHeads, maxSeqLen, numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit, device);
        register_module("rab", rab);

        LongVector normAttnShape = new LongVector(1);
        normAttnShape.put(0, (long) nHeads * dv);
        this.normAttn = new LayerNormImpl(normAttnShape);
        register_module("norm_attn", normAttn);
        normAttn.to(new Device(device), false);

        this.proj2 = new LinearImpl((long) nHeads * dv, dModel);
        register_module("proj2", proj2);
        proj2.to(new Device(device), false);

        this.dropoutLayer = new DropoutImpl(dropout);
    }

    public Tensor forward(Tensor x, Tensor paddingMask, Tensor timeDiffs) {
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);
        int H = nHeads;

        Tensor xNormed = normIn.forward(x);
        Tensor projOut = torch.silu(proj1.forward(xNormed));

        Tensor q = projOut.narrow(2, 0, H * dqk).reshape(batchSize, seqLen, H, dqk).transpose(1, 2);
        Tensor k = projOut.narrow(2, H * dqk, H * dqk).reshape(batchSize, seqLen, H, dqk).transpose(1, 2);
        Tensor u = projOut.narrow(2, 2 * H * dqk, H * dv).reshape(batchSize, seqLen, H, dv);
        Tensor v = projOut.narrow(2, 2 * H * dqk + H * dv, H * dv).reshape(batchSize, seqLen, H, dv).transpose(1, 2);

        Tensor scores = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(attnAlpha));

        Tensor rabBias = rab.forward(timeDiffs, seqLen);
        Tensor scoresWithBias = scores.add(rabBias);

        Tensor causalMask = torch.tril(torch.ones(new long[]{seqLen, seqLen},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .to(x.device(), ScalarType.Long));
        Tensor validMask = causalMask.unsqueeze(0).unsqueeze(0);

        Tensor finalMask;
        if (paddingMask != null) {
            Tensor keyMask = paddingMask.unsqueeze(1).unsqueeze(1);
            finalMask = torch.logical_and(validMask, keyMask);
        } else {
            finalMask = validMask;
        }

        Tensor maskedScores = scoresWithBias.masked_fill(
                torch.eq(finalMask, new Scalar(0)), new Scalar(-1e4f));

        Tensor attnWeights = torch.silu(maskedScores).div(new Scalar((float) maxSeqLen));

        Tensor attnOutput = torch.matmul(attnWeights, v);
        Tensor attnOutputReshaped = attnOutput.transpose(1, 2).reshape(batchSize, seqLen, H * dv);
        Tensor uFlat = u.reshape(batchSize, seqLen, H * dv);

        Tensor gated = normAttn.forward(attnOutputReshaped).mul(uFlat);
        Tensor dropped = dropoutLayer.forward(gated);
        return proj2.forward(dropped);
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor paddingMask) {
        return forward(x, paddingMask, (Tensor) null);
    }
}
