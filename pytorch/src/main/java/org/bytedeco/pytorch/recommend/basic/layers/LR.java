/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/LR.scala
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * Logistic regression module.
 *
 * <p>Parameters
 * <ul>
 *   <li>inputDim — Input dimension.</li>
 *   <li>sigmoid — Apply sigmoid to output when true (default false).</li>
 * </ul>
 *
 * <p>Shape: Input {@code (B, input_dim)}, Output {@code (B, 1)}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LR extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl fc;
    private final boolean sigmoid;

    public LR(long inputDim) {
        this(inputDim, false, DeviceSupport.backend());
    }

    public LR(long inputDim, boolean sigmoid) {
        this(inputDim, sigmoid, DeviceSupport.backend());
    }

    public LR(long inputDim, boolean sigmoid, String device) {
        super("LR");
        this.sigmoid = sigmoid;
        this.fc = new LinearImpl(inputDim, 1);
        register_module("fc", fc);
        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            fc.to(dev, false);
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        if (sigmoid) {
            return torch.sigmoid(fc.forward(x));
        }
        return fc.forward(x);
    }
}
