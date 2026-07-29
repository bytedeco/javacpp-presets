/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DeepFFM.scala
 * (DeepFFM + FatDeepFFM)
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Deep Field-weighted Factorization Machine (DeepFFM).
 * Reference: "Deep Field-Weighted Factorization Machine" Alibaba, IJCAI 2018
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DeepFFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embedding;
    private final DeepFFMFieldFM ffm;
    private final MLP mlp;

    public DeepFFM(List<? extends Feature> features, int fieldNum) {
        this(features, 8, fieldNum, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public DeepFFM(List<? extends Feature> features, int embedDim, int fieldNum,
                   long[] mlpDims, float dropout, String device) {
        super("DeepFFM");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        List<Feature> featList = new ArrayList<>(features);

        this.embedding = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embedding);

        this.ffm = new DeepFFMFieldFM(embedDim, fieldNum, device);
        register_module("ffm", ffm);

        long sparseDim = Features.calcSparseDim(featList);
        this.mlp = new MLP(sparseDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings3d = embedding.forward3D(sparseFeats);
        Tensor embeddings = embedding.forward(sparseFeats);

        Tensor ffmOut = ffm.forward(embeddings3d);
        Tensor mlpOut = mlp.forward(embeddings);

        return ffmOut.add(mlpOut);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
