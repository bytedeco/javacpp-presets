/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/FNFM.scala
 *
 * Field-aware Neural Factorization Machine (FNFM).
 * Sparse Input → Embeddings → Pairwise Element-wise Products → MLP → Output
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FNFM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numFields;
    private final EmbeddingLayer embeddingLayer;
    private final MLP mlp;

    public FNFM(List<? extends Feature> features) {
        this(features, 8, new long[]{256L, 128L, 64L}, 0.2f, DeviceSupport.backend());
    }

    public FNFM(List<? extends Feature> features, int embedDim, long[] mlpDims,
                float dropout, String device) {
        super("FNFM");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        int nf = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) nf++;
        }
        if (nf < 2) {
            throw new IllegalArgumentException("FNFM requires at least 2 sparse features");
        }
        this.numFields = nf;

        this.embeddingLayer = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("embedding", embeddingLayer);

        int numPairs = numFields * (numFields - 1) / 2;
        long ffmOutputDim = (long) numPairs * embedDim;
        // useBatchNorm=true as in Scala
        this.mlp = new MLP(ffmOutputDim, mlpDims, 1L, "relu", dropout, true, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> denseFeats) {
        Tensor embeddings = embeddingLayer.forward3D(sparseFeats);

        List<Tensor> interactions = new ArrayList<>();
        for (int i = 0; i < numFields; i++) {
            for (int j = i + 1; j < numFields; j++) {
                Tensor vi = embeddings.narrow(1, i, 1).squeeze(1);
                Tensor vj = embeddings.narrow(1, j, 1).squeeze(1);
                interactions.add(vi.mul(vj)); // element-wise product (batch, embed_dim)
            }
        }

        Tensor ffmFeatures;
        if (interactions.isEmpty()) {
            TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
            ffmFeatures = torch.zeros(new long[]{embeddings.size(0), 1L}, opts)
                    .to(embeddings.device(), ScalarType.Float);
        } else {
            var targetDev = interactions.get(0).device();
            TensorVector vec = new TensorVector();
            for (Tensor t : interactions) {
                Tensor onDev = t.device().equals(targetDev) ? t : t.to(targetDev, t.dtype());
                vec.push_back(onDev);
            }
            ffmFeatures = torch.cat(vec, 1L);
        }

        return mlp.forward(ffmFeatures).squeeze(1);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats) {
        return forward(sparseFeats, Collections.emptyMap());
    }
}
