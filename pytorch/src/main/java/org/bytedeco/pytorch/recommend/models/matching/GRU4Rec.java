/*
 * Ported from torch-rechub-scala: torchrec/models/matching/GRU4Rec.scala
 *
 * GRU4Rec - GRU-based Session-Based Recommender.
 * Dual-tower with GRU over history + MLP user tower, item embedding tower.
 * Reference: Hidasi et al., 2015 - http://arxiv.org/abs/1511.06939
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.GRUImpl;
import org.bytedeco.pytorch.nn.options.GRUOptions;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GRU4Rec extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int historyDim;
    private final int userDims;
    private final int numLayers;
    private final float temperature;
    private final List<String> historyFeatureNames;
    private final Set<String> userFeatureNames;
    private final Set<String> itemFeatureNames;
    private final Feature negItemFeature; // nullable
    private final EmbeddingLayer embedding;
    private final GRUImpl gru;
    private final MLP userMlp;
    private String mode; // nullable: "user" | "item" | null

    public GRU4Rec(List<? extends Feature> userFeatures,
                   List<? extends Feature> historyFeatures,
                   List<? extends Feature> itemFeatures) {
        this(userFeatures, historyFeatures, itemFeatures, null,
                defaultUserParams(), 1.0f, DeviceSupport.backend());
    }

    public GRU4Rec(List<? extends Feature> userFeatures,
                   List<? extends Feature> historyFeatures,
                   List<? extends Feature> itemFeatures,
                   Feature negItemFeature,
                   Map<String, Object> userParams,
                   float temperature,
                   String device) {
        super("GRU4Rec");
        if (userFeatures == null || userFeatures.isEmpty()) {
            throw new IllegalArgumentException("userFeatures cannot be empty");
        }
        if (historyFeatures == null || historyFeatures.isEmpty()) {
            throw new IllegalArgumentException("historyFeatures cannot be empty");
        }
        if (itemFeatures == null || itemFeatures.isEmpty()) {
            throw new IllegalArgumentException("itemFeatures cannot be empty");
        }
        this.temperature = temperature;
        this.negItemFeature = negItemFeature;
        this.mode = null;

        if (userParams == null) userParams = defaultUserParams();
        this.numLayers = userParams.containsKey("num_layers")
                ? ((Number) userParams.get("num_layers")).intValue() : 2;

        List<Feature> userList = new ArrayList<>(userFeatures);
        List<Feature> histList = new ArrayList<>(historyFeatures);
        List<Feature> itemList = new ArrayList<>(itemFeatures);

        int ud = 0;
        for (Feature f : userList) ud += f.embedDim();
        for (Feature f : histList) ud += f.embedDim();
        this.userDims = ud;
        this.historyDim = histList.get(0).embedDim();

        this.historyFeatureNames = new ArrayList<>();
        for (Feature f : histList) historyFeatureNames.add(f.name());
        this.userFeatureNames = new HashSet<>();
        for (Feature f : userList) userFeatureNames.add(f.name());
        this.itemFeatureNames = new HashSet<>();
        for (Feature f : itemList) itemFeatureNames.add(f.name());

        List<Feature> allFeats = new ArrayList<>();
        allFeats.addAll(userList);
        allFeats.addAll(itemList);
        allFeats.addAll(histList);
        if (negItemFeature != null) allFeats.add(negItemFeature);
        this.embedding = new EmbeddingLayer(allFeats, historyDim, device);
        register_module("embedding", embedding);

        Device targetDevice = new Device(device);
        GRUOptions opts = new GRUOptions(historyDim, historyDim);
        opts.num_layers().put(numLayers);
        opts.batch_first().put(true);
        opts.bias().put(false);
        this.gru = new GRUImpl(opts);
        gru.to(targetDevice, false);
        register_module("gru", gru);

        @SuppressWarnings("unchecked")
        List<Long> dimsList = userParams.containsKey("dims")
                ? (List<Long>) userParams.get("dims")
                : Collections.singletonList((long) historyDim * 2);
        long[] dims = new long[dimsList.size()];
        for (int i = 0; i < dimsList.size(); i++) dims[i] = dimsList.get(i);
        String activation = userParams.containsKey("activation")
                ? userParams.get("activation").toString() : "relu";
        float dropout = userParams.containsKey("dropout")
                ? ((Number) userParams.get("dropout")).floatValue() : 0.0f;
        this.userMlp = new MLP(userDims, dims, historyDim, activation, dropout,
                false, false, true, device);
        register_module("user_mlp", userMlp);
    }

    private static Map<String, Object> defaultUserParams() {
        Map<String, Object> m = new HashMap<>();
        m.put("num_layers", 2);
        return m;
    }

    public void setMode(String m) {
        this.mode = m;
    }

    /** Split combined map into sparse + sequence and run user tower. */
    public Tensor userTower(Map<String, Tensor> x) {
        Map<String, Tensor> seqMap = new LinkedHashMap<>();
        Map<String, Tensor> sparseMap = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : x.entrySet()) {
            if (historyFeatureNames.contains(e.getKey())) {
                seqMap.put(e.getKey(), e.getValue());
            } else {
                sparseMap.put(e.getKey(), e.getValue());
            }
        }
        return userTower(sparseMap, seqMap);
    }

    public Tensor userTower(Map<String, Tensor> sparseFeats, Map<String, Tensor> sequenceFeats) {
        if ("item".equals(mode)) {
            return torch.zeros(new long[]{1L, historyDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }

        Map<String, Tensor> filteredSparse = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : sparseFeats.entrySet()) {
            if (userFeatureNames.contains(e.getKey())) {
                filteredSparse.put(e.getKey(), e.getValue());
            }
        }
        Tensor userEmb = embedding.forward(filteredSparse, Collections.emptyMap(), true);

        Map<String, Tensor> seqFiltered = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : sequenceFeats.entrySet()) {
            if (historyFeatureNames.contains(e.getKey())) {
                seqFiltered.put(e.getKey(), e.getValue());
            }
        }
        Tensor rawSeq = embedding.forwardSeqRaw(seqFiltered);
        Tensor historyEmb = rawSeq.dim() == 4L ? rawSeq.squeeze(1L) : rawSeq;

        T_TensorTensor_T gruOutput = gru.forwardT_TensorTensor_T(historyEmb);
        // get1 = hidden state; select last layer
        Tensor gruHidden = gruOutput.get1();
        Tensor lastHidden = gruHidden.select(0, numLayers - 1);

        TensorVector cVec = new TensorVector();
        cVec.push_back(userEmb);
        cVec.push_back(lastHidden);
        Tensor combined = torch.cat(cVec, 1);

        Tensor userEmbedding = userMlp.forward(combined).unsqueeze(1);

        NormalizeFuncOptions normOpt = new NormalizeFuncOptions();
        normOpt.p(2);
        normOpt.dim(-1);
        normOpt.eps(1e-8);
        userEmbedding = torch.normalize(userEmbedding, normOpt);

        if ("user".equals(mode)) {
            return userEmbedding.squeeze(1);
        }
        return userEmbedding;
    }

    private Map<String, Tensor> selectItemFeats(Map<String, Tensor> x) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        if (x == null) return out;
        for (Map.Entry<String, Tensor> e : x.entrySet()) {
            if (itemFeatureNames.contains(e.getKey())) {
                out.put(e.getKey(), e.getValue());
            }
        }
        // Fallback: if caller only passed item ids under another key, keep first non-history sparse.
        if (out.isEmpty()) {
            for (Map.Entry<String, Tensor> e : x.entrySet()) {
                if (!historyFeatureNames.contains(e.getKey()) && !userFeatureNames.contains(e.getKey())) {
                    out.put(e.getKey(), e.getValue());
                }
            }
        }
        return out;
    }

    /** Collapse multi-field item embedding to [B, historyDim] for dot-product matching. */
    private Tensor toItemVector(Tensor emb) {
        if (emb == null) throw new IllegalArgumentException("item emb null");
        if (emb.dim() == 3L) {
            // [B, F, D] → mean over fields, or squeeze single field
            if (emb.size(1) == 1L) return emb.squeeze(1L);
            return emb.mean(1);
        }
        if (emb.dim() == 2L) {
            long d = emb.size(1);
            if (d == historyDim) return emb;
            if (d % historyDim == 0L) {
                long n = d / historyDim;
                return emb.view(emb.size(0), n, historyDim).mean(1);
            }
            // truncate / pad to historyDim
            if (d > historyDim) return emb.narrow(1, 0, historyDim);
            Tensor pad = torch.zeros(new long[]{emb.size(0), historyDim - d},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            TensorVector v = new TensorVector();
            v.push_back(emb);
            v.push_back(pad);
            return torch.cat(v, 1);
        }
        throw new IllegalArgumentException("unexpected item emb rank=" + emb.dim());
    }

    public Tensor itemTower(Map<String, Tensor> x) {
        if ("user".equals(mode)) {
            return torch.zeros(new long[]{1L, historyDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        }

        Map<String, Tensor> itemX = selectItemFeats(x);
        Tensor posEmbedding = toItemVector(embedding.forward(itemX, Collections.emptyMap(), false));

        NormalizeFuncOptions normOpt = new NormalizeFuncOptions();
        normOpt.p(2);
        normOpt.dim(-1);
        normOpt.eps(1e-8);
        Tensor posEmbNormed = torch.normalize(posEmbedding, normOpt);

        if ("item".equals(mode)) {
            return posEmbNormed;
        }

        if (negItemFeature != null) {
            Tensor negIdx = x.get(negItemFeature.name());
            if (negIdx == null) {
                throw new IllegalArgumentException("Missing feature: " + negItemFeature.name());
            }
            Map<String, Tensor> negX = Collections.singletonMap(negItemFeature.name(), negIdx);
            Tensor negEmb = toItemVector(embedding.forward(negX, Collections.emptyMap(), false));
            Tensor negEmbNormed = torch.normalize(negEmb, normOpt);
            // return [B, 2, H] for pos/neg
            return torch.stack(new TensorVector(new Tensor[]{posEmbNormed, negEmbNormed}), 1L);
        }
        return posEmbNormed; // [B, H]
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor userEmb = userTower(x); // often [B, 1, H] or [B, H]
        Tensor itemEmb = itemTower(x); // [B, H] or [B, 2, H]

        if ("user".equals(mode)) return userEmb;
        if ("item".equals(mode)) return itemEmb;

        Tensor u = userEmb.dim() == 3L ? userEmb.squeeze(1L) : userEmb;
        if (itemEmb.dim() == 3L) {
            // pos/neg pair: [B, 2, H] · [B, H] → [B, 2]
            return torch.mul(itemEmb, u.unsqueeze(1L)).sum(2L);
        }
        return torch.mul(u, itemEmb).sum(1L);
    }
}
