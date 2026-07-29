/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Attention.scala
 *
 * Decoder block for SAINT.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
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
public class SAINTDecoderBlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final MultiHeadAttention multiDe1;
    private final MultiHeadAttention multiDe2;
    private final LinearImpl ffnDe1;
    private final LinearImpl ffnDe2;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;
    private final LayerNormImpl ln3;
    private final DropoutImpl dropoutLayer;

    public SAINTDecoderBlock(int embedDim, int numHeads) {
        this(embedDim, numHeads, 256, 0.1f, DeviceSupport.backend());
    }

    public SAINTDecoderBlock(int embedDim, int numHeads, int ffnDim, float dropout, String device) {
        super("SAINTDecoderBlock");
        this.multiDe1 = new MultiHeadAttention(embedDim, numHeads, dropout, device);
        this.multiDe2 = new MultiHeadAttention(embedDim, numHeads, dropout, device);
        this.ffnDe1 = new LinearImpl(embedDim, ffnDim);
        this.ffnDe2 = new LinearImpl(ffnDim, embedDim);
        this.ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.ln3 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("multi_de1", multiDe1);
        register_module("multi_de2", multiDe2);
        register_module("ffn_de1", ffnDe1);
        register_module("ffn_de2", ffnDe2);
        register_module("ln1", ln1);
        register_module("ln2", ln2);
        register_module("ln3", ln3);
    }

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    public Tensor forward(Tensor inRes, Tensor inPos, Tensor enOut) {
        Tensor combined = inRes.add(inPos);

        // Cross attention on encoder output
        Tensor crossAttn = multiDe1.forward(combined, enOut, enOut);
        Tensor withResidual1 = combined.add(dropoutLayer.forward(crossAttn));
        Tensor normed1 = ln1.forward(withResidual1);

        // Self attention
        Tensor selfAttn = multiDe2.forward(normed1, normed1, normed1);
        Tensor withResidual2 = normed1.add(dropoutLayer.forward(selfAttn));
        Tensor normed2 = ln2.forward(withResidual2);

        // FFN
        Tensor ffnOut = dropoutLayer.forward(ffnDe2.forward(torch.relu(ffnDe1.forward(normed2))));
        return ln3.forward(normed2.add(ffnOut));
    }
}
