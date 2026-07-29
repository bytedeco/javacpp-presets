/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/ESMM.scala
 *
 * Entire Space Multi-Task Model (ESMM).
 * Reference: SIGIR'2018 - https://arxiv.org/abs/1804.07931
 * Returns [cvr_pred, ctr_pred, ctcvr_pred] concatenated.
 */
package org.bytedeco.pytorch.utils.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ESMM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer userEmbedding;
    private final EmbeddingLayer itemEmbedding;
    private final MLP towerCvr;
    private final MLP towerCtr;

    public ESMM(List<? extends Feature> userFeatures, List<? extends Feature> itemFeatures) {
        this(userFeatures, itemFeatures, Collections.emptyMap(), Collections.emptyMap(),
                DeviceSupport.backend());
    }

    public ESMM(List<? extends Feature> userFeatures, List<? extends Feature> itemFeatures,
                Map<String, Object> cvrParams, Map<String, Object> ctrParams, String device) {
        super("ESMM");
        if ((userFeatures == null || userFeatures.isEmpty())
                && (itemFeatures == null || itemFeatures.isEmpty())) {
            throw new IllegalArgumentException("userFeatures or itemFeatures cannot be empty");
        }

        List<Feature> userList = userFeatures != null ? new ArrayList<>(userFeatures) : new ArrayList<>();
        List<Feature> itemList = itemFeatures != null ? new ArrayList<>(itemFeatures) : new ArrayList<>();

        int userEmbedDim = 0;
        for (Feature f : userList) userEmbedDim += f.embedDim();
        int itemEmbedDim = 0;
        for (Feature f : itemList) itemEmbedDim += f.embedDim();
        long towerInputDim = userEmbedDim + itemEmbedDim;

        int userEmbedSize = userList.isEmpty() ? 8 : userList.get(0).embedDim();
        int itemEmbedSize = itemList.isEmpty() ? 8 : itemList.get(0).embedDim();

        this.userEmbedding = new EmbeddingLayer(userList, userEmbedSize, device);
        this.itemEmbedding = new EmbeddingLayer(itemList, itemEmbedSize, device);
        register_module("userEmbedding", userEmbedding);
        register_module("itemEmbedding", itemEmbedding);

        if (cvrParams == null) cvrParams = Collections.emptyMap();
        if (ctrParams == null) ctrParams = Collections.emptyMap();

        this.towerCvr = buildTower(towerInputDim, cvrParams, device);
        register_module("tower_cvr", towerCvr);

        this.towerCtr = buildTower(towerInputDim, ctrParams, device);
        register_module("tower_ctr", towerCtr);
    }

    private static MLP buildTower(long inputDim, Map<String, Object> params, String device) {
        @SuppressWarnings("unchecked")
        List<Long> dimsList = params.containsKey("dims")
                ? (List<Long>) params.get("dims")
                : java.util.Arrays.asList(128L, 64L);
        long[] dims = new long[dimsList.size()];
        for (int i = 0; i < dimsList.size(); i++) dims[i] = dimsList.get(i);
        String activation = params.containsKey("activation")
                ? params.get("activation").toString() : "relu";
        float dropout = params.containsKey("dropout")
                ? ((Number) params.get("dropout")).floatValue() : 0.0f;
        return new MLP(inputDim, dims, 1L, activation, dropout, false, false, true, device);
    }

    /**
     * Backward-compatible factory: split features list into user/item halves.
     */
    public static ESMM fromFeatures(List<? extends Feature> features, long[] towerDims,
                                    float dropout, String device) {
        int half = features.size() / 2;
        List<Feature> user = new ArrayList<>();
        List<Feature> item = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) {
            if (i < half) user.add(features.get(i));
            else item.add(features.get(i));
        }
        Map<String, Object> params = new java.util.HashMap<>();
        List<Long> dims = new ArrayList<>();
        for (long d : towerDims) dims.add(d);
        params.put("dims", dims);
        params.put("activation", "relu");
        params.put("dropout", dropout);
        return new ESMM(user, item, params, params, device);
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor userEmbed = userEmbedding.forward(x, Collections.emptyMap(), true);
        Tensor itemEmbed = itemEmbedding.forward(x, Collections.emptyMap(), true);

        TensorVector cVec = new TensorVector();
        cVec.push_back(userEmbed);
        cVec.push_back(itemEmbed);
        Tensor inputTower = torch.cat(cVec, 1);

        Tensor cvrLogit = towerCvr.forward(inputTower);
        Tensor ctrLogit = towerCtr.forward(inputTower);

        Tensor cvrPred = torch.sigmoid(cvrLogit);
        Tensor ctrPred = torch.sigmoid(ctrLogit);
        Tensor ctcvrPred = torch.mul(ctrPred, cvrPred);

        TensorVector oVec = new TensorVector();
        oVec.push_back(cvrPred);
        oVec.push_back(ctrPred);
        oVec.push_back(ctcvrPred);
        return torch.cat(oVec, 1);
    }
}
