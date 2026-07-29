/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CrossLayer.scala
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

/**
 * Cross layer.
 *
 * <p>Parameters: inputDim — input dimension.
 * Shape: Input {@code (B, *)}, Output {@code (B, *)}.
 *
 * <p>Note: multi-arg forward (x0, xi) is an ordinary method, not Module.forward override.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CrossLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl w;
    private final Tensor b;

    public CrossLayer(long inputDim) {
        this(inputDim, DeviceSupport.backend());
    }

    public CrossLayer(long inputDim, String device) {
        super("CrossLayer");
        this.w = new LinearImpl(inputDim, 1);
        register_module("w", w);
        w.to(new Device(device), false);

        this.b = torch.zeros(new long[]{inputDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        register_buffer("b", b);

        if (device != null && !"cpu".equals(device)) {
            this.to(new Device(device), false);
        }
    }

    public Tensor forward(Tensor x0, Tensor xi) {
        Tensor wtx = w.forward(xi); // (batch, 1)
        Tensor x0Wtx = x0.mul(wtx); // (batch, dim)
        return x0Wtx.add(b);
    }
}
