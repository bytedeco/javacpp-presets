/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/WideDeep.scala
 *
 * Wide & Deep Learning. Reference: Google.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class WideDeep extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> features;
    private final EmbeddingLayer embeddingLayer;
    private final MLP mlp;

    public WideDeep(List<? extends Feature> features) {
        this(features, 8, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public WideDeep(List<? extends Feature> features, int embedDim, long[] mlpDims,
                    float dropout, String device) {
        super("WideDeep");
        this.features = new ArrayList<>(features);
        this.embeddingLayer = new EmbeddingLayer(this.features, embedDim, device);
        register_module("embedding", embeddingLayer);

        long sparseDim = Features.calcSparseDim(this.features);
        this.mlp = new MLP(sparseDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward(sparseFeats);
        var dev = embeddings.device();

        // Wide: sum of embeddings per field (FM first-order style)
        List<String> sparseNames = new ArrayList<>();
        for (Feature f : features) {
            if (f instanceof SparseFeature) {
                sparseNames.add(f.name());
            }
        }

        Tensor wideOut = null;
        for (String name : sparseNames) {
            Tensor idx = sparseFeats.get(name);
            if (idx == null) continue;
            Tensor idxOnDev;
            if (idx.device().equals(dev)) {
                idxOnDev = idx.toType(ScalarType.Long);
            } else {
                idxOnDev = idx.toType(ScalarType.Long).to(dev, ScalarType.Long);
            }
            Tensor emb = embeddingLayer.getEmbedding(name, idxOnDev);
            Tensor part = emb.sum(1).unsqueeze(1);
            wideOut = (wideOut == null) ? part : wideOut.add(part);
        }

        if (wideOut == null) {
            wideOut = torch.zeros(new long[]{embeddings.size(0), 1L},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                    .to(dev, ScalarType.Float);
        }

        Tensor deepOut = mlp.forward(embeddings);
        return wideOut.add(deepOut);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
