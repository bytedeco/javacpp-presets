/*
 * Ported from torch-rechub-scala: torchrec/models/matching/FaceBookDSSM.scala
 *
 * FaceBookDSSM - Embedding-based Retrieval in Facebook Search.
 * Dual-tower with positive/negative item scoring.
 * Reference: KDD'2020 - https://arxiv.org/abs/2006.11632
 */
package org.bytedeco.pytorch.utils.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FaceBookDSSM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> userFeatures;
    private final List<Feature> posItemFeatures;
    private final List<Feature> negItemFeatures;
    private final int userOutDim;
    private final int itemOutDim;
    private final float temperature;
    private final EmbeddingLayer userEmbedding;
    private final EmbeddingLayer posItemEmbedding;
    private final EmbeddingLayer negItemEmbedding;
    private final MLP userMlp;
    private final MLP itemMlp;
    private String mode; // nullable

    public FaceBookDSSM(List<? extends Feature> userFeatures,
                        List<? extends Feature> posItemFeatures,
                        List<? extends Feature> negItemFeatures) {
        this(userFeatures, posItemFeatures, negItemFeatures,
                Collections.emptyMap(), Collections.emptyMap(), 1.0f, DeviceSupport.backend());
    }

    public FaceBookDSSM(List<? extends Feature> userFeatures,
                        List<? extends Feature> posItemFeatures,
                        List<? extends Feature> negItemFeatures,
                        Map<String, Object> userParams,
                        Map<String, Object> itemParams,
                        float temperature,
                        String device) {
        super("FaceBookDSSM");
        if (userFeatures == null || userFeatures.isEmpty()) {
            throw new IllegalArgumentException("userFeatures cannot be empty");
        }
        if (posItemFeatures == null || posItemFeatures.isEmpty()) {
            throw new IllegalArgumentException("posItemFeatures cannot be empty");
        }
        if (negItemFeatures == null || negItemFeatures.isEmpty()) {
            throw new IllegalArgumentException("negItemFeatures cannot be empty");
        }
        this.userFeatures = new ArrayList<>(userFeatures);
        this.posItemFeatures = new ArrayList<>(posItemFeatures);
        this.negItemFeatures = new ArrayList<>(negItemFeatures);
        this.temperature = temperature;
        this.mode = null;

        int userDims = 0;
        for (Feature f : this.userFeatures) userDims += f.embedDim();
        int itemDims = 0;
        for (Feature f : this.posItemFeatures) itemDims += f.embedDim();
        this.userOutDim = this.userFeatures.get(0).embedDim();
        this.itemOutDim = this.posItemFeatures.get(0).embedDim();

        this.userEmbedding = new EmbeddingLayer(this.userFeatures, userOutDim, device);
        register_module("user_embedding", userEmbedding);

        this.posItemEmbedding = new EmbeddingLayer(this.posItemFeatures, itemOutDim, device);
        register_module("pos_item_embedding", posItemEmbedding);

        this.negItemEmbedding = new EmbeddingLayer(this.negItemFeatures,
                this.negItemFeatures.get(0).embedDim(), device);
        register_module("neg_item_embedding", negItemEmbedding);

        if (userParams == null) userParams = Collections.emptyMap();
        if (itemParams == null) itemParams = Collections.emptyMap();

        this.userMlp = buildMlp(userDims, userOutDim, userParams, device);
        register_module("user_mlp", userMlp);

        this.itemMlp = buildMlp(itemDims, itemOutDim, itemParams, device);
        register_module("item_mlp", itemMlp);
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
        // outputLayer = false as in Scala
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

        Set<String> userNames = new HashSet<>();
        for (Feature f : userFeatures) userNames.add(f.name());
        Map<String, Tensor> userFeats = filterKeys(x, userNames);

        Tensor inputUser = userEmbedding.forward(userFeats, Collections.emptyMap(), true);
        Tensor userEmb = userMlp.forward(inputUser);

        NormalizeFuncOptions normOpt = new NormalizeFuncOptions();
        normOpt.p(2);
        normOpt.dim(1);
        normOpt.eps(1e-8);
        return torch.normalize(userEmb, normOpt);
    }

    /** Returns [posEmbedding, negEmbedding]. */
    public Tensor[] itemTower(Map<String, Tensor> x) {
        if ("user".equals(mode)) {
            Tensor zero = torch.zeros(new long[]{1L, itemOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            return new Tensor[]{zero, zero.clone()};
        }

        Set<String> posNames = new HashSet<>();
        for (Feature f : posItemFeatures) posNames.add(f.name());
        Set<String> negNames = new HashSet<>();
        for (Feature f : negItemFeatures) negNames.add(f.name());
        Map<String, Tensor> posFeats = filterKeys(x, posNames);
        Map<String, Tensor> negFeats = filterKeys(x, negNames);

        Tensor inputItemPos;
        if (!posFeats.isEmpty()) {
            inputItemPos = posItemEmbedding.forward(posFeats, Collections.emptyMap(), true);
        } else {
            inputItemPos = torch.zeros(new long[]{1L, itemOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }

        if ("item".equals(mode)) {
            Tensor posEmbedding = itemMlp.forward(inputItemPos);
            Tensor zeroNeg = torch.zeros(new long[]{1L, itemOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            return new Tensor[]{posEmbedding, zeroNeg};
        }

        Tensor inputItemNeg;
        if (!negFeats.isEmpty()) {
            inputItemNeg = negItemEmbedding.forward(negFeats, Collections.emptyMap(), true);
        } else {
            inputItemNeg = torch.zeros(new long[]{1L, negItemFeatures.get(0).embedDim()},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }

        Tensor posEmbedding = itemMlp.forward(inputItemPos);
        Tensor negEmbedding = itemMlp.forward(inputItemNeg);

        NormalizeFuncOptions normOpt = new NormalizeFuncOptions();
        normOpt.p(2);
        normOpt.dim(1);
        normOpt.eps(1e-8);
        posEmbedding = torch.normalize(posEmbedding, normOpt);
        negEmbedding = torch.normalize(negEmbedding, normOpt);

        return new Tensor[]{posEmbedding, negEmbedding};
    }

    /** Returns [posScore, negScore] (or tower outputs in mode). */
    public Tensor[] forwardPair(Map<String, Tensor> x) {
        Tensor userEmbedding = userTower(x);
        Tensor[] items = itemTower(x);
        Tensor posItemEmb = items[0];
        Tensor negItemEmb = items[1];

        if ("user".equals(mode)) {
            Tensor zeroNeg = torch.zeros(new long[]{1L, itemOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            return new Tensor[]{userEmbedding, zeroNeg};
        }
        if ("item".equals(mode)) {
            Tensor zeroNeg = torch.zeros(new long[]{1L, itemOutDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            return new Tensor[]{posItemEmb, zeroNeg};
        }

        Tensor posScore = torch.mul(userEmbedding, posItemEmb).sum(1L);
        Tensor negScore = torch.mul(userEmbedding, negItemEmb).sum(1L);
        return new Tensor[]{posScore, negScore};
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
