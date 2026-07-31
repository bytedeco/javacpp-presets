/*
 * Ported from torch-rechub-scala: torchrec/models/matching/YoutubeSBC.scala
 *
 * YoutubeSBC - Sampling-Bias-Corrected Neural Modeling for Matching.
 * Dual-tower with in-batch softmax sampling and bias correction.
 * Reference: RecSys'2019
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class YoutubeSBC extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> userFeatures;
    private final List<Feature> itemFeatures;
    private final Feature sampleWeightFeature; // nullable
    private final int batchSize;
    private final int nNeg;
    private final float temperature;
    private final int userOutDim;
    private final int itemOutDim;
    private final long[] index0;
    private final long[] index1;
    private final EmbeddingLayer embedding;
    private final MLP userMlp;
    private final MLP itemMlp;
    private String mode;

    public YoutubeSBC(List<? extends Feature> userFeatures, List<? extends Feature> itemFeatures) {
        this(userFeatures, itemFeatures, null, Collections.emptyMap(), Collections.emptyMap(),
                128, 3, 1.0f, DeviceSupport.backend());
    }

    public YoutubeSBC(List<? extends Feature> userFeatures,
                      List<? extends Feature> itemFeatures,
                      Feature sampleWeightFeature,
                      Map<String, Object> userParams,
                      Map<String, Object> itemParams,
                      int batchSize, int nNeg, float temperature, String device) {
        super("YoutubeSBC");
        if (userFeatures == null || userFeatures.isEmpty()) {
            throw new IllegalArgumentException("userFeatures cannot be empty");
        }
        if (itemFeatures == null || itemFeatures.isEmpty()) {
            throw new IllegalArgumentException("itemFeatures cannot be empty");
        }
        if (nNeg <= 0) {
            throw new IllegalArgumentException("nNeg must be positive");
        }
        if (batchSize <= nNeg) {
            throw new IllegalArgumentException("batchSize must be greater than nNeg");
        }
        this.userFeatures = new ArrayList<>(userFeatures);
        this.itemFeatures = new ArrayList<>(itemFeatures);
        this.sampleWeightFeature = sampleWeightFeature;
        this.batchSize = batchSize;
        this.nNeg = nNeg;
        this.temperature = temperature;
        this.mode = null;
        this.userOutDim = this.userFeatures.get(0).embedDim();
        this.itemOutDim = this.itemFeatures.get(0).embedDim();

        int userDims = 0;
        for (Feature f : this.userFeatures) userDims += f.embedDim();
        int itemDims = 0;
        for (Feature f : this.itemFeatures) itemDims += f.embedDim();

        List<Feature> allFeats = new ArrayList<>();
        allFeats.addAll(this.userFeatures);
        allFeats.addAll(this.itemFeatures);
        if (sampleWeightFeature != null) allFeats.add(sampleWeightFeature);
        this.embedding = new EmbeddingLayer(allFeats, userOutDim, device);
        register_module("embedding", embedding);

        if (userParams == null) userParams = Collections.emptyMap();
        if (itemParams == null) itemParams = Collections.emptyMap();
        this.userMlp = buildMlp(userDims, userOutDim, userParams, device);
        register_module("user_mlp", userMlp);
        this.itemMlp = buildMlp(itemDims, itemOutDim, itemParams, device);
        register_module("item_mlp", itemMlp);

        // Precompute in-batch sampling indices
        this.index0 = new long[batchSize * (nNeg + 1)];
        this.index1 = new long[batchSize * (nNeg + 1)];
        int p = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < nNeg + 1; j++) {
                index0[p] = i;
                long idx = i + j;
                if (idx >= batchSize) idx -= batchSize;
                index1[p] = idx;
                p++;
            }
        }
    }

    private static MLP buildMlp(int inputDim, int outputDim, Map<String, Object> params, String device) {
        @SuppressWarnings("unchecked")
        List<Long> dimsList = params.containsKey("dims")
                ? (List<Long>) params.get("dims")
                : Collections.singletonList(128L);
        long[] dims = new long[dimsList.size()];
        for (int i = 0; i < dimsList.size(); i++) dims[i] = dimsList.get(i);
        String activation = params.containsKey("activation")
                ? params.get("activation").toString() : "relu";
        float dropout = params.containsKey("dropout")
                ? ((Number) params.get("dropout")).floatValue() : 0.0f;
        return new MLP(inputDim, dims, outputDim, activation, dropout, false, false, false, device);
    }

    public void setMode(String m) {
        this.mode = m;
    }

    private static Map<String, Tensor> selectFeats(Map<String, Tensor> x, List<? extends Feature> feats) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        if (x == null || feats == null) return out;
        for (Feature f : feats) {
            Tensor t = x.get(f.name());
            if (t != null) out.put(f.name(), t);
        }
        return out;
    }

    public Tensor userTower(Map<String, Tensor> x) {
        if ("item".equals(mode)) {
            return torch.zeros(new long[]{batchSize, userOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }
        // Only user-side features; shared EmbeddingLayer otherwise cats item tables too.
        Tensor inputUser = embedding.forward(selectFeats(x, userFeatures), Collections.emptyMap(), true);
        return userMlp.forward(inputUser);
    }

    public Tensor itemTower(Map<String, Tensor> x) {
        if ("user".equals(mode)) {
            return torch.zeros(new long[]{batchSize, itemOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }
        Tensor inputItem = embedding.forward(selectFeats(x, itemFeatures), Collections.emptyMap(), true);
        return itemMlp.forward(inputItem);
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor userEmbedding = userTower(x);
        Tensor itemEmbedding = itemTower(x);

        if ("user".equals(mode)) return userEmbedding;
        if ("item".equals(mode)) return itemEmbedding;

        Tensor userEmbExpanded = userEmbedding.unsqueeze(1);

        NormalizeFuncOptions userNormOpt = new NormalizeFuncOptions();
        userNormOpt.p(2);
        userNormOpt.dim(-1);
        userNormOpt.eps(1e-8);
        Tensor userNorm = torch.normalize(userEmbExpanded, userNormOpt);

        NormalizeFuncOptions itemNormOpt = new NormalizeFuncOptions();
        itemNormOpt.p(2);
        itemNormOpt.dim(-1);
        itemNormOpt.eps(1e-8);
        Tensor itemNorm = torch.normalize(itemEmbedding, itemNormOpt);
        Tensor cosSim = torch.mul(userNorm, itemNorm.unsqueeze(1)).sum(2); // [B, B]

        Tensor sampleWeight;
        if (sampleWeightFeature != null) {
            Tensor w = x.get(sampleWeightFeature.name());
            Tensor emb = embedding.forward(
                    Collections.singletonMap(sampleWeightFeature.name(), w),
                    Collections.emptyMap(), true);
            sampleWeight = emb.squeeze(1);
        } else {
            int currentBatch = (int) userEmbedding.size(0);
            sampleWeight = torch.ones(new long[]{currentBatch},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }

        Tensor sampleWeightExpanded = sampleWeight.unsqueeze(1);
        Tensor scores = cosSim.sub(torch.log(sampleWeightExpanded));

        int currentBatchSize = (int) userEmbedding.size(0);
        long[] idx0;
        long[] idx1;
        if (currentBatchSize * (nNeg + 1) != index0.length) {
            int n = currentBatchSize * (nNeg + 1);
            idx0 = new long[n];
            idx1 = new long[n];
            System.arraycopy(index0, 0, idx0, 0, n);
            System.arraycopy(index1, 0, idx1, 0, n);
            for (int i = 0; i < n; i++) {
                if (idx0[i] >= currentBatchSize) idx0[i] -= currentBatchSize;
                if (idx1[i] >= currentBatchSize) idx1[i] -= currentBatchSize;
            }
        } else {
            idx0 = index0;
            idx1 = index1;
        }

        Tensor flatScores = scores.view(-1);
        Tensor idxTensor = torch.tensor(idx1,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        Tensor batchIdxTensor = torch.tensor(idx0,
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));

        // Scala: flatScores.index_select(0, batchIdx).index_select(0, idx)
        // This double index_select is odd; mirror literally.
        Tensor gatheredScores = flatScores.index_select(0, batchIdxTensor).index_select(0, idxTensor);
        Tensor finalScores = gatheredScores.div(new Scalar(temperature));
        return finalScores.view(currentBatchSize, nNeg + 1);
    }
}
