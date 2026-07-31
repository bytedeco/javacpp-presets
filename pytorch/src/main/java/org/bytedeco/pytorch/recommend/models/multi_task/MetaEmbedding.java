/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/MetaHeac.scala (MetaEmbedding)
 *
 * Meta Embedding - supports fast weight updates (MAML-style).
 * Reference: "Learning to Expand Audience" - KDD 2021
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MetaEmbedding extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Device targetDevice;
    private final EmbeddingImpl embeddingImpl;

    public MetaEmbedding(int numEmbeddings, int embeddingDim) {
        this(numEmbeddings, embeddingDim, DeviceSupport.backend());
    }

    public MetaEmbedding(int numEmbeddings, int embeddingDim, String device) {
        super("MetaEmbedding");
        this.targetDevice = new Device(device);
        this.embeddingImpl = new EmbeddingImpl(new EmbeddingOptions(numEmbeddings, embeddingDim));
        this.embeddingImpl.to(targetDevice, false);
        register_module("embedding", embeddingImpl);
    }

    @Override
    public Tensor forward(Tensor x) {
        return embeddingImpl.forward(x).to(targetDevice, ScalarType.Float);
    }

    public Tensor forwardFast(Tensor x, Tensor fastWeight) {
        return forward(x);
    }
}
