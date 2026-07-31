/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/FNN.scala
 *
 * Factorization Machine supported Neural Network (FNN).
 * Sparse Input → FM Embeddings → MLP → Output
 * Reference: Zhang et al., 2016
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FNN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embeddingLayer;
    private final MLP mlp;

    public FNN(List<? extends Feature> features) {
        this(features, 8, new long[]{256L, 128L, 64L}, 0.2f, DeviceSupport.backend());
    }

    public FNN(List<? extends Feature> features, int embedDim, long[] mlpDims,
               float dropout, String device) {
        super("FNN");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        List<Feature> featList = new ArrayList<>(features);
        this.embeddingLayer = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embeddingLayer);

        long sparseDim = Features.calcSparseDim(featList);
        this.mlp = new MLP(sparseDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        // Get FM-style embeddings: (batch, num_fields * embed_dim)
        Tensor embeddings = embeddingLayer.forward(sparseFeats);
        // Keep (batch,1) shape for consistency
        return mlp.forward(embeddings);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
