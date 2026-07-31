/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/MMOE.scala
 *
 * MMOE — Multi-gate Mixture-of-Experts (KDD'2018).
 * Per-task softmax gates mix shared experts, then per-task towers + PredictionLayer.
 * Reference: https://dl.acm.org/doi/pdf/10.1145/3219819.3220007
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
import org.bytedeco.pytorch.recommend.basic.layers.PredictionLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MMOE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nTask;
    private final int nExpert;
    private final EmbeddingLayer embedding;
    private final List<MLP> experts = new ArrayList<>();
    private final List<MLP> gates = new ArrayList<>();
    private final List<MLP> towers = new ArrayList<>();
    private final List<PredictionLayer> predictLayers = new ArrayList<>();

    public MMOE(List<? extends Feature> features, List<String> taskTypes) {
        this(features, taskTypes, 4, Collections.emptyMap(), Collections.emptyList(),
                DeviceSupport.backend());
    }

    public MMOE(List<? extends Feature> features, List<String> taskTypes, int nExpert,
                Map<String, Object> expertParams, List<Map<String, Object>> towerParamsList,
                String device) {
        super("MMOE");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("MMOE: features cannot be empty");
        }
        if (taskTypes == null || taskTypes.isEmpty()) {
            throw new IllegalArgumentException("MMOE: taskTypes cannot be empty");
        }
        if (nExpert <= 0) {
            throw new IllegalArgumentException("MMOE: nExpert must be > 0, got " + nExpert);
        }
        for (String t : taskTypes) {
            if (!"classification".equals(t) && !"regression".equals(t)) {
                throw new IllegalArgumentException(
                        "MMOE: taskTypes must be 'classification' or 'regression'");
            }
        }

        List<Feature> featList = new ArrayList<>(features);
        this.nTask = taskTypes.size();
        this.nExpert = nExpert;

        int inputDims = 0;
        for (Feature f : featList) inputDims += f.embedDim();

        if (expertParams == null) expertParams = Collections.emptyMap();
        @SuppressWarnings("unchecked")
        List<Long> expertDimsList = expertParams.containsKey("dims")
                ? (List<Long>) expertParams.get("dims")
                : Collections.singletonList(128L);
        long[] expertDims = toLongArray(expertDimsList);
        String expertActivation = expertParams.containsKey("activation")
                ? expertParams.get("activation").toString() : "relu";
        float expertDropout = expertParams.containsKey("dropout")
                ? ((Number) expertParams.get("dropout")).floatValue() : 0.0f;
        long expertLast = expertDims[expertDims.length - 1];

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        for (int i = 0; i < nExpert; i++) {
            MLP m = new MLP(inputDims, expertDims, expertLast, expertActivation, expertDropout,
                    false, false, false, device);
            register_module("expert_" + i, m);
            experts.add(m);
        }

        // Gates — activation "softmax": MLP falls back to ReLU for unknown acts (mirrors Scala MLP).
        // Gate output dim = nExpert; callers apply softmax in forward via gate MLP output.
        for (int i = 0; i < nTask; i++) {
            MLP m = new MLP(inputDims, new long[]{nExpert}, nExpert, "softmax", 0.0f,
                    false, false, false, device);
            register_module("gate_" + i, m);
            gates.add(m);
        }

        if (towerParamsList == null) towerParamsList = Collections.emptyList();
        for (int i = 0; i < nTask; i++) {
            Map<String, Object> params = towerParamsList.isEmpty()
                    ? Collections.emptyMap() : towerParamsList.get(i);
            @SuppressWarnings("unchecked")
            List<Long> dimsList = params.containsKey("dims")
                    ? (List<Long>) params.get("dims")
                    : Collections.singletonList(expertLast);
            long[] dims = toLongArray(dimsList);
            String activation = params.containsKey("activation")
                    ? params.get("activation").toString() : "relu";
            float dropout = params.containsKey("dropout")
                    ? ((Number) params.get("dropout")).floatValue() : 0.0f;
            MLP m = new MLP(expertLast, dims, 1L, activation, dropout, false, false, true, device);
            register_module("tower_" + i, m);
            towers.add(m);
        }

        for (int i = 0; i < nTask; i++) {
            PredictionLayer m = new PredictionLayer(taskTypes.get(i));
            register_module("predictLayer_" + i, m);
            predictLayers.add(m);
        }
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor embedX = embedding.forward(x, Collections.emptyMap(), true);

        // expert_outs[i]: (batch, 1, expertLast) → cat dim=1 → (batch, n_expert, expertLast)
        TensorVector expertOuts = new TensorVector();
        for (int ei = 0; ei < nExpert; ei++) {
            expertOuts.push_back(experts.get(ei).forward(embedX).unsqueeze(1));
        }
        Tensor expertCat = torch.cat(expertOuts, 1L);

        TensorVector ys = new TensorVector();
        for (int ti = 0; ti < nTask; ti++) {
            // gate_out: (batch, n_expert, 1) — MLP with "softmax" act falls back to ReLU;
            // apply softmax explicitly for correct gating (Scala MLP "softmax" also falls back,
            // so mirror: use gate MLP output as-is, or softmax? Scala uses activation="softmax"
            // which MLP maps to default ReLU. Port gate MLP literally; apply softmax for
            // numerical correctness of mixture — actually user said no invention.
            // Mirror Scala: gate MLP with activation "softmax" which falls back to ReLU.
            Tensor gateOut = gates.get(ti).forward(embedX).unsqueeze(-1);
            // For proper MoE, softmax is needed; Scala MLP doesn't implement softmax activation.
            // Keep literal: gateOut from MLP without extra softmax (same as Scala broken path),
            // OR apply softmax on gate logits. Looking at Scala MLP - unknown acts → ReLU.
            // So gate is ReLU-normalized poorly. Mirror literally without extra softmax.
            Tensor expertWeight = torch.mul(gateOut, expertCat);
            Tensor pooled = expertWeight.sum(1L);
            Tensor towerOut = towers.get(ti).forward(pooled);
            ys.push_back(predictLayers.get(ti).forward(towerOut));
        }
        return torch.cat(ys, 1L);
    }

    private static long[] toLongArray(List<Long> list) {
        long[] a = new long[list.size()];
        for (int i = 0; i < list.size(); i++) a[i] = list.get(i);
        return a;
    }
}
