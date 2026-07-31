/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/LR.scala
 *
 * Logistic Regression Model — simplest CTR baseline.
 * Named RankingLR to avoid clash with basic.layers.LR.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RankingLR extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embeddingLayer;

    public RankingLR(List<? extends Feature> features) {
        this(features, 8, DeviceSupport.backend());
    }

    public RankingLR(List<? extends Feature> features, int embedDim, String device) {
        super("LR");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);
        if (device != null && !"cpu".equals(device)) {
            embeddingLayer.toDevice(device);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats);
        // Sum over fields and embedding dimension → scalar per batch
        return embeddings.sum(1L).sum(1L).unsqueeze(1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
