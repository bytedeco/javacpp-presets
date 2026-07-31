/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/RelativeBucketedTimeAndPositionBias.scala
 *
 * HSTU rab^{p,t}: per-head bias on attention scores from (position-diff, time-diff) pairs.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.GeneratorOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RelativeBucketedTimeAndPositionBias extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nHeads;
    private final int maxSeqLen;
    private final int numTimeBuckets;
    private final String timeBucketFn;
    private final float timeBucketDivisor;
    private final String timeBucketUnit;
    private final Tensor posW;
    private final Tensor tsW;

    public RelativeBucketedTimeAndPositionBias(int nHeads, int maxSeqLen) {
        this(nHeads, maxSeqLen, 128, "sqrt", 1.0f, "minutes", DeviceSupport.backend());
    }

    public RelativeBucketedTimeAndPositionBias(
            int nHeads, int maxSeqLen, int numTimeBuckets,
            String timeBucketFn, float timeBucketDivisor, String timeBucketUnit, String device) {
        super("RelativeBucketedTimeAndPositionBias");
        if (!"sqrt".equals(timeBucketFn) && !"log".equals(timeBucketFn)) {
            throw new IllegalArgumentException("Unsupported time_bucket_fn: " + timeBucketFn);
        }
        if (!"minutes".equals(timeBucketUnit) && !"seconds".equals(timeBucketUnit)) {
            throw new IllegalArgumentException("Unsupported time_bucket_unit: " + timeBucketUnit);
        }
        this.nHeads = nHeads;
        this.maxSeqLen = maxSeqLen;
        this.numTimeBuckets = numTimeBuckets;
        this.timeBucketFn = timeBucketFn;
        this.timeBucketDivisor = timeBucketDivisor;
        this.timeBucketUnit = timeBucketUnit;

        Device targetDevice = new Device(device);
        float boundPos = (float) Math.sqrt(1.0 / (2 * maxSeqLen - 1));
        Tensor pos = torch.rand(new long[]{2L * maxSeqLen - 1, nHeads},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .to(targetDevice, ScalarType.Float);
        pos.uniform_(-boundPos, boundPos, new GeneratorOptional());
        register_parameter("pos_w", pos);
        this.posW = pos;

        float boundTs = (float) Math.sqrt(1.0 / (numTimeBuckets + 1));
        Tensor ts = torch.rand(new long[]{numTimeBuckets + 1L, nHeads},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .to(targetDevice, ScalarType.Float);
        ts.uniform_(-boundTs, boundTs, new GeneratorOptional());
        register_parameter("ts_w", ts);
        this.tsW = ts;
    }

    private Tensor bucketizeTime(Tensor dt) {
        Tensor dtAbs = dt.abs();
        Tensor dtScaled = "minutes".equals(timeBucketUnit)
                ? dtAbs.div(new Scalar(60.0f)) : dtAbs;
        Tensor dtClamped = dtScaled.clamp(
                new ScalarOptional(new Scalar(1e-6f)), new ScalarOptional());

        Tensor buckets = "sqrt".equals(timeBucketFn) ? dtClamped.sqrt() : dtClamped.log();
        return buckets.div(new Scalar(timeBucketDivisor))
                .clamp(new ScalarOptional(new Scalar(0)),
                        new ScalarOptional(new Scalar(numTimeBuckets)))
                .toType(ScalarType.Long);
    }

    public Tensor forward(Tensor timeDiffs, int seqLen) {
        int L;
        if (timeDiffs != null) {
            L = (int) timeDiffs.size(1);
        } else {
            if (seqLen <= 0) {
                throw new IllegalArgumentException("Provide either time_diffs or seq_len.");
            }
            if (seqLen > maxSeqLen) {
                throw new IllegalArgumentException(
                        "seq_len (" + seqLen + ") exceeds max_seq_len (" + maxSeqLen + ")");
            }
            L = seqLen;
        }

        Device paramDevice = posW.device();
        Tensor positions = torch.arange(
                new Scalar(L),
                new TensorOptions()
                        .dtype(new ScalarTypeOptional(ScalarType.Long))
                        .device(new DeviceOptional(paramDevice)));
        Tensor relPosIdx = positions.unsqueeze(0).sub(positions.unsqueeze(1))
                .add(new Scalar(maxSeqLen - 1));
        TensorIndexVector idxVec = new TensorIndexVector();
        idxVec.push_back(new TensorIndex(relPosIdx));
        Tensor posBias = posW.index(idxVec).permute(2, 0, 1);

        if (timeDiffs == null) {
            return posBias.unsqueeze(0);
        }

        Tensor td = timeDiffs.to(paramDevice, ScalarType.Float);
        Tensor dtPairwise = td.unsqueeze(2).sub(td.unsqueeze(1));
        Tensor timeBuckets = bucketizeTime(dtPairwise);
        TensorIndexVector tsIdx = new TensorIndexVector();
        tsIdx.push_back(new TensorIndex(timeBuckets));
        Tensor timeBias = tsW.index(tsIdx).to(paramDevice, ScalarType.Float);
        Tensor timeBiasPermuted = timeBias.permute(0, 3, 1, 2);
        return posBias.unsqueeze(0).add(timeBiasPermuted);
    }

    public Tensor forward(int seqLen) {
        return forward(null, seqLen);
    }
}
