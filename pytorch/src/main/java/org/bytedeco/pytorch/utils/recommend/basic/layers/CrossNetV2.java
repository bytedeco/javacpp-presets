/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/CrossNetwork.scala (CrossNetV2)
 * Aligned with FuxiCTR CrossNetV2:
 *   X_{l+1} = X_l + X_0 * Linear_l(X_l)   (Linear has bias=true)
 *
 * Using LinearImpl(bias=true) keeps W and b as submodules so model.to(device)
 * moves them together — avoids the old free bias Tensor + register_parameter
 * device drift on MPS.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CrossNetV2 extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numLayers;
    private final LinearImpl[] layers;

    public CrossNetV2(long inputDim) {
        this(inputDim, 3, DeviceSupport.backend());
    }

    public CrossNetV2(long inputDim, int numLayers) {
        this(inputDim, numLayers, DeviceSupport.backend());
    }

    public CrossNetV2(long inputDim, int numLayers, String device) {
        super("CrossNetV2");
        this.numLayers = numLayers;
        this.layers = new LinearImpl[numLayers];

        for (int i = 0; i < numLayers; i++) {
            // Prefer (in,out) ctor — matches AFM/DeepFM style; bias defaults to true.
            LinearImpl layer = new LinearImpl(inputDim, inputDim);
            register_module("cross_" + i, layer);
            layers[i] = layer;
        }

        // Always move whole module (no-op on cpu, required on mps/cuda).
        if (device != null) {
            this.to(new Device(device), false);
        }
    }

    @Override
    public Tensor forward(Tensor x0) {
        Tensor xl = x0;
        for (int i = 0; i < numLayers; i++) {
            // X_{l+1} = X_l + X_0 * Linear(X_l)
            Tensor projected = layers[i].forward(xl);
            xl = xl.add(x0.mul(projected));
        }
        return xl;
    }

    public int numLayers() {
        return numLayers;
    }
}
