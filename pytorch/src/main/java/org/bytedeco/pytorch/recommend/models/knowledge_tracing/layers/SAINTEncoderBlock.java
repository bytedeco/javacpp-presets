/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Attention.scala
 *
 * Encoder block for SAINT.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers;

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
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SAINTEncoderBlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final MultiHeadAttention multiEn;
    private final LinearImpl ffnEn1;
    private final LinearImpl ffnEn2;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;
    private final DropoutImpl dropoutLayer;

    public SAINTEncoderBlock(int embedDim, int numHeads) {
        this(embedDim, numHeads, 256, 0.1f, DeviceSupport.backend());
    }

    public SAINTEncoderBlock(int embedDim, int numHeads, int ffnDim, float dropout, String device) {
        super("SAINTEncoderBlock");
        this.multiEn = new MultiHeadAttention(embedDim, numHeads, dropout, device);
        this.ffnEn1 = new LinearImpl(embedDim, ffnDim);
        this.ffnEn2 = new LinearImpl(ffnDim, embedDim);
        this.ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("multi_en", multiEn);
        register_module("ffn_en1", ffnEn1);
        register_module("ffn_en2", ffnEn2);
        register_module("ln1", ln1);
        register_module("ln2", ln2);
    }

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    public Tensor forward(Tensor inEx, Tensor inCat, Tensor inPos) {
        Tensor combined = inEx.add(inCat).add(inPos);
        Tensor attended = multiEn.forward(combined, combined, combined);
        Tensor withResidual1 = combined.add(dropoutLayer.forward(attended));
        Tensor normed1 = ln1.forward(withResidual1);

        Tensor ffnOut = dropoutLayer.forward(ffnEn2.forward(torch.relu(ffnEn1.forward(normed1))));
        return ln2.forward(normed1.add(ffnOut));
    }
}
