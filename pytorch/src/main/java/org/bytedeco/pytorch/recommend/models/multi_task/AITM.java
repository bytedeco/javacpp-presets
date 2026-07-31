/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/AITM.scala
 *
 * AITM — Adaptive Information Transfer Multi-task framework (KDD'2021).
 * Per-task bottom + tower; info-gate and attention pass info from task i-1 to i.
 * Reference: https://arxiv.org/abs/2105.08489
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AITM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nTask;
    private final EmbeddingLayer embedding;
    private final List<MLP> bottoms = new ArrayList<>();
    private final List<MLP> towers = new ArrayList<>();
    private final List<MLP> infoGates = new ArrayList<>();
    private final List<AttentionLayer> aits = new ArrayList<>();

    public AITM(
            List<Feature> features,
            int nTask,
            Map<String, Object> bottomParams,
            List<Map<String, Object>> towerParamsList) {
        this(features, nTask, bottomParams, towerParamsList, DeviceSupport.backend());
    }

    public AITM(
            List<Feature> features,
            int nTask,
            Map<String, Object> bottomParams,
            List<Map<String, Object>> towerParamsList,
            String device) {
        super("AITM");
        if (nTask <= 0) {
            throw new IllegalArgumentException("AITM: nTask must be > 0");
        }
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("AITM: features cannot be empty");
        }
        if (towerParamsList == null || towerParamsList.size() != nTask) {
            throw new IllegalArgumentException(
                    "AITM: towerParamsList.size must equal nTask (" + nTask + ")");
        }
        this.nTask = nTask;

        @SuppressWarnings("unchecked")
        List<Long> bottomDimsList = bottomParams != null && bottomParams.containsKey("dims")
                ? (List<Long>) bottomParams.get("dims")
                : List.of(128L);
        long[] bottomDims = toLongArray(bottomDimsList);
        long bottomLast = bottomDims[bottomDims.length - 1];
        String bottomActivation = bottomParams != null && bottomParams.containsKey("activation")
                ? String.valueOf(bottomParams.get("activation"))
                : "relu";
        float bottomDropout = bottomParams != null && bottomParams.containsKey("dropout")
                ? ((Number) bottomParams.get("dropout")).floatValue()
                : 0.0f;

        int inputDims = 0;
        for (Feature f : features) {
            inputDims += f.embedDim();
        }

        this.embedding = new EmbeddingLayer(features, features.get(0).embedDim(), device);
        register_module("embedding", embedding);

        for (int i = 0; i < nTask; i++) {
            MLP bottom = new MLP(inputDims, bottomDims, bottomLast, bottomActivation, bottomDropout,
                    false, false, false, device);
            register_module("bottom_" + i, bottom);
            bottoms.add(bottom);
        }

        for (int i = 0; i < nTask; i++) {
            Map<String, Object> params = towerParamsList.get(i);
            @SuppressWarnings("unchecked")
            List<Long> tDims = params != null && params.containsKey("dims")
                    ? (List<Long>) params.get("dims")
                    : List.of(bottomLast);
            String activation = params != null && params.containsKey("activation")
                    ? String.valueOf(params.get("activation"))
                    : bottomActivation;
            float dropout = params != null && params.containsKey("dropout")
                    ? ((Number) params.get("dropout")).floatValue()
                    : bottomDropout;
            MLP tower = new MLP(bottomLast, toLongArray(tDims), 1L, activation, dropout,
                    false, false, true, device);
            register_module("tower_" + i, tower);
            towers.add(tower);
        }

        for (int i = 1; i < nTask; i++) {
            MLP gate = new MLP(bottomLast, new long[]{bottomLast}, bottomLast, "relu", 0.0f,
                    false, false, false, device);
            infoGates.add(gate);
            register_module("infoGate_" + (infoGates.size() - 1), gate);
        }

        for (int i = 1; i < nTask; i++) {
            AttentionLayer ait = new AttentionLayer((int) bottomLast, device);
            aits.add(ait);
            register_module("ait_" + (aits.size() - 1), ait);
        }
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor embedX = embedding.forward(x, Collections.emptyMap(), true);

        List<Tensor> towerInputs = new ArrayList<>();
        for (int i = 0; i < nTask; i++) {
            towerInputs.add(bottoms.get(i).forward(embedX));
        }

        for (int k = 0; k < nTask - 1; k++) {
            Tensor info = infoGates.get(k).forward(towerInputs.get(k)).unsqueeze(1);
            TensorVector aitVec = new TensorVector(
                    towerInputs.get(k + 1).unsqueeze(1),
                    info);
            Tensor aitInput = torch.cat(aitVec, 1);
            towerInputs.set(k + 1, aits.get(k).forward(aitInput));
        }

        List<Tensor> ys = new ArrayList<>();
        for (int j = 0; j < nTask; j++) {
            ys.add(torch.sigmoid(towers.get(j).forward(towerInputs.get(j))));
        }
        TensorVector outVec = new TensorVector(ys.toArray(new Tensor[0]));
        return torch.cat(outVec, 1L);
    }

    private static long[] toLongArray(List<Long> list) {
        long[] arr = new long[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
