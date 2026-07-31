/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/NFM.scala
 *
 * Neural Factorization Machine (NFM). Reference: SIGIR 2017.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.BiInteractionPooling;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embeddingLayer;
    private final BiInteractionPooling biInteractionPool;
    private final MLP mlp;

    public NFM(List<? extends Feature> features) {
        this(features, 8, new long[]{128L, 64L}, 0.2f, DeviceSupport.backend());
    }

    public NFM(List<? extends Feature> features, int embedDim, long[] mlpDims,
               float dropout, String device) {
        super("NFM");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);

        this.biInteractionPool = new BiInteractionPooling(device);
        register_module("biInteraction", biInteractionPool);

        this.mlp = new MLP(embedDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats, Collections.emptyMap());
        Tensor firstOrder = embeddings.sum(1);
        Tensor biOut = biInteractionPool.forward(embeddings);
        Tensor combined = firstOrder.add(biOut);
        return mlp.forward(combined).squeeze(1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
