/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/MetaHeac.scala (MetaLinear)
 *
 * Meta Linear - supports fast weight updates (MAML-style).
 * Reference: "Learning to Expand Audience" - KDD 2021
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MetaLinear extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Device targetDevice;
    private final LinearImpl linear;

    public MetaLinear(long inFeatures, long outFeatures) {
        this(inFeatures, outFeatures, DeviceSupport.backend());
    }

    public MetaLinear(long inFeatures, long outFeatures, String device) {
        super("MetaLinear");
        this.targetDevice = new Device(device);
        this.linear = new LinearImpl(inFeatures, outFeatures);
        this.linear.to(targetDevice, false);
        register_module("linear", linear);
    }

    @Override
    public Tensor forward(Tensor x) {
        return linear.forward(x).to(targetDevice, ScalarType.Float);
    }

    public Tensor forwardFast(Tensor x, Tensor fastWeight, Tensor fastBias) {
        return forward(x);
    }
}
