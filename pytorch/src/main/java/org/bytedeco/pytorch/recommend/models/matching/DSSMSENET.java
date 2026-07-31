/*
 * Ported from torch-rechub-scala: torchrec/models/matching/DSSMSENET.scala
 *
 * DSSM with SENET adaptive feature weighting on both towers.
 * Cosine similarity returned as temperature-scaled logit (no sigmoid).
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
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.SENETLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DSSMSENET extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> userFeatures;
    private final List<Feature> itemFeatures;
    private final int userOutDim;
    private final int itemOutDim;
    private final int userNumFeatures;
    private final int itemNumFeatures;
    private final float temperature;
    private final Set<String> userFeatureNames;
    private final Set<String> itemFeatureNames;
    private final EmbeddingLayer userEmbeddingLayer;
    private final EmbeddingLayer itemEmbeddingLayer;
    private final SENETLayer userSenet;
    private final SENETLayer itemSenet;
    private final MLP userMlp;
    private final MLP itemMlp;
    private String mode;

    public DSSMSENET(List<? extends Feature> userFeatures, List<? extends Feature> itemFeatures) {
        this(userFeatures, itemFeatures, Collections.emptyMap(), Collections.emptyMap(),
                1.0f, DeviceSupport.backend());
    }

    public DSSMSENET(List<? extends Feature> userFeatures, List<? extends Feature> itemFeatures,
                     Map<String, Object> userParams, Map<String, Object> itemParams,
                     float temperature, String device) {
        super("DSSMSENET");
        if (userFeatures == null || userFeatures.isEmpty()) {
            throw new IllegalArgumentException("userFeatures cannot be empty");
        }
        if (itemFeatures == null || itemFeatures.isEmpty()) {
            throw new IllegalArgumentException("itemFeatures cannot be empty");
        }
        this.userFeatures = new ArrayList<>(userFeatures);
        this.itemFeatures = new ArrayList<>(itemFeatures);
        this.temperature = temperature;
        this.mode = null;

        int userDims = 0;
        for (Feature f : this.userFeatures) userDims += f.embedDim();
        int itemDims = 0;
        for (Feature f : this.itemFeatures) itemDims += f.embedDim();
        this.userOutDim = this.userFeatures.get(0).embedDim();
        this.itemOutDim = this.itemFeatures.get(0).embedDim();

        this.userNumFeatures = countSenetFeatures(this.userFeatures);
        this.itemNumFeatures = countSenetFeatures(this.itemFeatures);

        this.userFeatureNames = new HashSet<>();
        for (Feature f : this.userFeatures) userFeatureNames.add(f.name());
        this.itemFeatureNames = new HashSet<>();
        for (Feature f : this.itemFeatures) itemFeatureNames.add(f.name());

        this.userEmbeddingLayer = new EmbeddingLayer(this.userFeatures, userOutDim, device);
        register_module("user_embedding", userEmbeddingLayer);

        this.itemEmbeddingLayer = new EmbeddingLayer(this.itemFeatures, itemOutDim, device);
        register_module("item_embedding", itemEmbeddingLayer);

        this.userSenet = new SENETLayer(userNumFeatures, 3, device);
        register_module("user_senet", userSenet);

        this.itemSenet = new SENETLayer(itemNumFeatures, 3, device);
        register_module("item_senet", itemSenet);

        if (userParams == null) userParams = Collections.emptyMap();
        if (itemParams == null) itemParams = Collections.emptyMap();

        this.userMlp = buildMlp(userDims, userOutDim, userParams, device);
        register_module("user_mlp", userMlp);

        this.itemMlp = buildMlp(itemDims, itemOutDim, itemParams, device);
        register_module("item_mlp", itemMlp);
    }

    private static int countSenetFeatures(List<Feature> features) {
        int n = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) {
                n++;
            } else if (f instanceof SequenceFeature) {
                SequenceFeature sf = (SequenceFeature) f;
                if (sf.sharedWith() == null || sf.sharedWith().isEmpty()) {
                    n++;
                }
            }
        }
        return n;
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

    public Tensor userTower(Map<String, Tensor> x) {
        if ("item".equals(mode)) {
            return torch.zeros(new long[]{1L, userOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }

        Map<String, Tensor> userFeats = filterKeys(x, userFeatureNames);
        Tensor inputUser = userEmbeddingLayer.forward(userFeats, Collections.emptyMap(), true);

        long batchSize = inputUser.size(0);
        Tensor reshapedUser = inputUser.view(batchSize, userNumFeatures, -1);
        Tensor senetedUser = userSenet.forward(reshapedUser);
        Tensor flattenedUser = senetedUser.view(batchSize, -1);

        Tensor userEmbedding = userMlp.forward(flattenedUser);
        NormalizeFuncOptions normOpt = new NormalizeFuncOptions();
        normOpt.p(2);
        normOpt.dim(1);
        normOpt.eps(1e-8);
        return torch.normalize(userEmbedding, normOpt);
    }

    public Tensor itemTower(Map<String, Tensor> x) {
        if ("user".equals(mode)) {
            return torch.zeros(new long[]{1L, itemOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }

        Map<String, Tensor> itemFeats = filterKeys(x, itemFeatureNames);
        Tensor inputItem = itemEmbeddingLayer.forward(itemFeats, Collections.emptyMap(), true);

        long batchSize = inputItem.size(0);
        Tensor reshapedItem = inputItem.view(batchSize, itemNumFeatures, -1);
        Tensor senetedItem = itemSenet.forward(reshapedItem);
        Tensor flattenedItem = senetedItem.view(batchSize, -1);

        Tensor itemEmbedding = itemMlp.forward(flattenedItem);
        NormalizeFuncOptions normOpt = new NormalizeFuncOptions();
        normOpt.p(2);
        normOpt.dim(1);
        normOpt.eps(1e-8);
        return torch.normalize(itemEmbedding, normOpt);
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor userEmbedding = userTower(x);
        Tensor itemEmbedding = itemTower(x);

        if ("user".equals(mode)) return userEmbedding;
        if ("item".equals(mode)) return itemEmbedding;

        Tensor cosSim = torch.mul(userEmbedding, itemEmbedding).sum(1L);
        return cosSim.div(new Scalar(temperature));
    }

    private static Map<String, Tensor> filterKeys(Map<String, Tensor> x, Set<String> names) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : x.entrySet()) {
            if (names.contains(e.getKey())) {
                out.put(e.getKey(), e.getValue());
            }
        }
        return out;
    }
}
