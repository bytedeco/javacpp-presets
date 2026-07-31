/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/PositionalEmbedding.scala
 *
 * Learnable positional embedding used by SAINT.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LearnablePositionalEmbedding extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String device;
    private final EmbeddingImpl embedding;
    private final DropoutImpl dropoutLayer;

    public LearnablePositionalEmbedding(int maxLen, int embedDim) {
        this(maxLen, embedDim, 0.1f, DeviceSupport.backend());
    }

    public LearnablePositionalEmbedding(int maxLen, int embedDim, float dropout, String device) {
        super("LearnablePositionalEmbedding");
        this.device = device;
        this.embedding = new EmbeddingImpl(new EmbeddingOptions(maxLen, embedDim));
        register_module("embedding", embedding);
        this.dropoutLayer = new DropoutImpl(dropout);
    }

    /** Forward over positions [0, seqLen). */
    public Tensor forward(long seqLen) {
        Device dev = new Device(device);
        Tensor positions = torch.arange(
                new Scalar(0),
                new Scalar(seqLen),
                new TensorOptions()
                        .device(new DeviceOptional(dev))
                        .dtype(new ScalarTypeOptional(ScalarType.Long)));
        return dropoutLayer.forward(embedding.forward(positions));
    }
}
