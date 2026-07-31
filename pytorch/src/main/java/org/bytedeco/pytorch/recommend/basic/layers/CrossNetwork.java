/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CrossNetwork.scala
 *
 * CrossNetwork (DCN), CrossNetV2 (DCN v2), CrossNetMix (DCN v2 MoE).
 */
package org.bytedeco.pytorch.recommend.basic.layers;

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
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

/**
 * Cross Network from DCN.
 * Formula: x_{l+1} = x_0 * (W_l x_l) + b_l + x_l
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CrossNetwork extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numLayers;
    private final List<LinearImpl> wLayers = new ArrayList<>();
    private final List<Tensor> biasList = new ArrayList<>();

    public CrossNetwork(long inputDim) {
        this(inputDim, 3, DeviceSupport.backend());
    }

    public CrossNetwork(long inputDim, int numLayers) {
        this(inputDim, numLayers, DeviceSupport.backend());
    }

    public CrossNetwork(long inputDim, int numLayers, String device) {
        super("CrossNetwork");
        this.numLayers = numLayers;
        Device dev = new Device(device);

        for (int i = 0; i < numLayers; i++) {
            LinearImpl w = new LinearImpl(new LinearOptions(inputDim, 1L).bias(false));
            register_module("w_" + i, w);
            wLayers.add(w);

            Tensor b = torch.zeros(new long[]{inputDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(dev, ScalarType.Float);
            register_parameter("b_" + i, b);
            biasList.add(b);
        }
    }

    @Override
    public Tensor forward(Tensor x0) {
        Tensor xl = x0;
        for (int i = 0; i < numLayers; i++) {
            Tensor xw = wLayers.get(i).forward(xl);         // (batch, 1)
            Tensor dot = x0.mul(xw);                         // (batch, inputDim)
            xl = dot.add(biasList.get(i)).add(xl);
        }
        return xl;
    }
}
