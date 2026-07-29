/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/LNN.scala
 *
 * Logarithmic Neural Network (LNN) for recommendation.
 * y = exp(W * log(|x| + eps) + b) - 1  (via log1p / expm1)
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

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
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LNN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int embedDim;
    private final int lnnDim;
    private final long lnnOutputDim;
    private final Tensor lnnWeight;
    private final Tensor lnnBias;

    public LNN(int numFields, int embedDim) {
        this(numFields, embedDim, 8, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public LNN(int numFields, int embedDim, int lnnDim, long[] mlpDims,
               float dropout, String device) {
        super("LNN");
        // mlpDims/dropout kept for API parity with Scala ctor (unused in LNN forward alone).
        if (numFields < 2) {
            throw new IllegalArgumentException("numFields must be >= 2, got " + numFields);
        }
        if (embedDim <= 0) {
            throw new IllegalArgumentException("embedDim must be positive, got " + embedDim);
        }
        if (lnnDim < 1) {
            throw new IllegalArgumentException("lnnDim must be >= 1, got " + lnnDim);
        }
        this.numFields = numFields;
        this.embedDim = embedDim;
        this.lnnDim = lnnDim;
        this.lnnOutputDim = (long) lnnDim * embedDim;

        // LNN weight: (lnn_dim, num_fields)
        Tensor w = torch.randn(new long[]{lnnDim, numFields})
                .mul(new Scalar((float) Math.sqrt(2.0 / numFields)));
        Tensor pW = new Tensor();
        pW.copy_(w);
        register_parameter("lnn_weight", pW);
        this.lnnWeight = pW;

        // LNN bias: (lnn_dim, embed_dim)
        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor b = torch.zeros(new long[]{lnnDim, embedDim}, opts);
        Tensor pB = new Tensor();
        pB.copy_(b);
        register_parameter("lnn_bias", pB);
        this.lnnBias = pB;

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            lnnWeight.to(dev, ScalarType.Float);
            lnnBias.to(dev, ScalarType.Float);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, num_fields, embed_dim) = (B, F, E)
        int batchSize = (int) x.size(0);

        Tensor absX = x.abs();
        Tensor logX = torch.log1p(absX);

        Tensor w = lnnWeight.to(x.device(), ScalarType.Float);
        // logX: (B, F, E) → (B, E, F)
        Tensor logXT = logX.transpose(1, 2);
        Tensor wT = w.t(); // (F, L)
        // preAct: (B, E, L) via bmm
        Tensor preAct = torch.bmm(logXT, wT);
        // preActT: (B, L, E)
        Tensor preActT = preAct.transpose(1, 2);

        Tensor b = lnnBias.to(x.device(), ScalarType.Float);
        Tensor bBcast = b.unsqueeze(0).expand(batchSize, lnnDim, embedDim);
        Tensor out = preActT.add(bBcast);

        out = torch.expm1(out);
        out = out.relu();
        return out.view(batchSize, lnnOutputDim);
    }
}
