/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Attention.scala
 *
 * Transformer layer with distance bias attention.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TransformerLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final DistanceBiasMultiHeadAttention attention;
    private final LayerNormImpl norm1;
    private final LayerNormImpl norm2;
    private final LinearImpl linear1;
    private final LinearImpl linear2;
    private final DropoutImpl dropoutLayer;

    public TransformerLayer(int embedDim, int numHeads) {
        this(embedDim, numHeads, 256, 0.1f, DeviceSupport.backend());
    }

    public TransformerLayer(int embedDim, int numHeads, int ffnDim, float dropout, String device) {
        super("TransformerLayer");
        this.attention = new DistanceBiasMultiHeadAttention(embedDim, numHeads, dropout, device);
        this.norm1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.norm2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.linear1 = new LinearImpl(embedDim, ffnDim);
        this.linear2 = new LinearImpl(ffnDim, embedDim);
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("attention", attention);
        register_module("norm1", norm1);
        register_module("norm2", norm2);
        register_module("linear1", linear1);
        register_module("linear2", linear2);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            linear2.to(dev, false);
        }
    }

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    public Tensor forward(Tensor x, int mask) {
        Tensor attended = attention.forward(x, mask);
        Tensor withResidual1 = x.add(dropoutLayer.forward(attended));
        Tensor normed1 = norm1.forward(withResidual1);

        Tensor ffnOut = dropoutLayer.forward(linear2.forward(torch.relu(linear1.forward(normed1))));
        Tensor withResidual2 = normed1.add(ffnOut);
        return norm2.forward(withResidual2);
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, 1);
    }
}
