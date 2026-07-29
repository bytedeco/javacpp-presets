/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CEN.scala
 *
 * Cross Embedding Network with attention over field crosses.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CEN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numFieldCrosses;
    private final Device dev;
    private final Tensor u;
    private final MLP mlpAttention;

    public CEN(int embedDim, int numFieldCrosses, int reductionRatio) {
        this(embedDim, numFieldCrosses, reductionRatio, DeviceSupport.backend());
    }

    public CEN(int embedDim, int numFieldCrosses, int reductionRatio, String device) {
        super("CEN");
        this.embedDim = embedDim;
        this.numFieldCrosses = numFieldCrosses;
        this.dev = DeviceSupport.deviceOf(device);

        TensorOptions tensorOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(ScalarType.Float))
                .device(new DeviceOptional(dev));

        this.u = torch.rand(new long[]{numFieldCrosses, embedDim}, tensorOpts);
        register_parameter("u", u);

        this.mlpAttention = new MLP(
                numFieldCrosses,
                new long[]{Math.max(1L, numFieldCrosses / reductionRatio)},
                numFieldCrosses,
                "relu",
                0.0f,
                false,
                false,
                true,
                device);
        register_module("mlp_att", mlpAttention);
    }

    @Override
    public Tensor forward(Tensor em) {
        long b = em.size(0);
        long c = em.size(1);
        long d = em.size(2);

        if (c != numFieldCrosses) {
            throw new IllegalArgumentException(
                    "CEN cross count mismatch: expect " + numFieldCrosses + ", input dim1=" + c);
        }
        if (d != embedDim) {
            throw new IllegalArgumentException(
                    "CEN embed dim mismatch: expect " + embedDim + ", input dim2=" + d);
        }
        if (!em.device().equals(dev)) {
            throw new IllegalArgumentException(
                    "CEN device mismatch: layer=" + dev + ", input=" + em.device());
        }

        Tensor u3d = u.unsqueeze(0);
        Tensor mul = torch.mul(u3d, em);
        Tensor dVec = torch.relu(mul.sum(-1));
        Tensor s = mlpAttention.forward(dVec);
        Tensor sExpand = s.unsqueeze(-1);
        Tensor aem = torch.mul(sExpand, em);
        return aem.reshape(aem.size(0), -1);
    }
}
