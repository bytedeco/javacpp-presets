/*
 * Ported from torch-rechub-scala: torchrec/models/generative/TIGER.scala
 *
 * Tree-based Indexing with Generative Enhancement (Amazon, SIGIR 2023).
 * Uses tree-based hierarchical indexing for efficient item retrieval
 * with generative enhancement for query understanding.
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TIGER extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final Tensor itemEmb;
    private final MLP encoder;
    private final MLP treeEncoder;

    public TIGER(Tensor itemEmbeddings) {
        this(itemEmbeddings, 8, 128, 2, 0.2f, DeviceSupport.backend());
    }

    public TIGER(
            Tensor itemEmbeddings,
            int embedDim,
            int hiddenDim,
            int numLayers,
            float dropout,
            String device) {
        super("TIGER");
        this.embedDim = embedDim;

        // Frozen item embeddings (no gradient)
        this.itemEmb = itemEmbeddings.clone().detach();
        this.itemEmb.requires_grad_(false);

        this.encoder = new MLP(embedDim, new long[]{(long) hiddenDim}, hiddenDim, "relu", dropout,
                false, false, true, device);
        register_module("encoder", encoder);

        this.treeEncoder = new MLP(hiddenDim, new long[]{hiddenDim / 2L}, hiddenDim, "relu", 0f,
                false, false, true, device);
        register_module("treeEncoder", treeEncoder);
    }

    /**
     * @param sequence (batch, seq_len) item IDs
     * @return tree-encoded representation
     */
    @Override
    public Tensor forward(Tensor sequence) {
        Tensor seqFlat = sequence.view(-1).toType(ScalarType.Long);
        Tensor seqEmb = itemEmb.index_select(0, seqFlat)
                .view(sequence.size(0), sequence.size(1), embedDim);

        Tensor pooled = seqEmb.mean(1);
        Tensor encoded = encoder.forward(pooled);
        return treeEncoder.forward(encoded);
    }

    public Tensor getItemEmbedding(Tensor itemId) {
        return itemEmb.index_select(0, itemId.toType(ScalarType.Long));
    }

    public long getItemCount() {
        return itemEmb.size(0);
    }
}
