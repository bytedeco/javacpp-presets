/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/HSTUBlock.scala
 *
 * Stack of HSTULayer modules with external residual: x = x + Layer(x).
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class HSTUBlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nLayers;
    private final HSTULayer[] layerRefs;

    public HSTUBlock() {
        this(512, 8, 4, 64, 64, 0.1f, 200, 128, "sqrt", 1.0f, "minutes", DeviceSupport.backend());
    }

    public HSTUBlock(int dModel, int nHeads, int nLayers, int dqk, int dv, float dropout,
                     int maxSeqLen, int numTimeBuckets, String timeBucketFn,
                     float timeBucketDivisor, String timeBucketUnit, String device) {
        super("HSTUBlock");
        this.nLayers = nLayers;
        this.layerRefs = new HSTULayer[nLayers];

        for (int i = 0; i < nLayers; i++) {
            HSTULayer layer = new HSTULayer(
                    dModel, nHeads, dqk, dv, dropout, maxSeqLen,
                    numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit, device);
            register_module("layer_" + i, layer);
            layerRefs[i] = layer;
        }
    }

    public Tensor forward(Tensor x, Tensor paddingMask, Tensor timeDiffs) {
        Tensor h = x;
        for (HSTULayer layer : layerRefs) {
            h = h.add(layer.forward(h, paddingMask, timeDiffs));
        }
        return h;
    }

    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null, (Tensor) null);
    }

    public Tensor forward(Tensor x, Tensor paddingMask) {
        return forward(x, paddingMask, (Tensor) null);
    }
}
