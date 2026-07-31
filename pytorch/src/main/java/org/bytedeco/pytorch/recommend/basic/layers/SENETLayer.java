/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/SENETLayer.scala
 *
 * SENet-style feature gating (FiBiNet).
 * Mirrors the Python reference torch-rechub implementation:
 *   z = mean(x, dim=-1)
 *   a = Linear -> ReLU -> Linear -> ReLU (z)
 *   v = x * a.unsqueeze(-1)
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * SENet-style feature gating (FiBiNet).
 *
 * <p>Shape: Input {@code (batch, numFields, embedDim)}, Output same.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SENETLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final int reduction;
    private final SequentialImpl mlp;

    public SENETLayer(int numFields) {
        this(numFields, 3, DeviceSupport.backend());
    }

    public SENETLayer(int numFields, int reduction) {
        this(numFields, reduction, DeviceSupport.backend());
    }

    public SENETLayer(int numFields, int reduction, String device) {
        super("SENETLayer");
        if (numFields <= 0) {
            throw new IllegalArgumentException("SENETLayer: numFields must be > 0, got " + numFields);
        }
        if (reduction <= 0) {
            throw new IllegalArgumentException("SENETLayer: reduction must be > 0, got " + reduction);
        }
        this.numFields = numFields;
        this.reduction = reduction;
        long reducedSize = Math.max(1L, (long) numFields / (long) reduction);

        this.mlp = new SequentialImpl();
        mlp.push_back("fc1", new LinearImpl(numFields, reducedSize));
        mlp.push_back("relu1", new ReLUImpl());
        mlp.push_back("fc2", new LinearImpl(reducedSize, numFields));
        mlp.push_back("relu2", new ReLUImpl());
        register_module("mlp", mlp);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            mlp.to(dev, false);
        }
    }

    public int numFields() {
        return numFields;
    }

    public int reduction() {
        return reduction;
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, numFields, embedDim)
        Tensor z = x.mean(-1L);               // (batch, numFields)
        Tensor a = mlp.forward(z);            // (batch, numFields)
        return x.mul(a.unsqueeze(-1));        // (batch, numFields, embedDim)
    }
}
