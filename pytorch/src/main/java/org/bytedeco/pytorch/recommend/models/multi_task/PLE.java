/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/PLE.scala
 *
 * PLE — Progressive Layered Extraction (RecSys'2020).
 * Stack of CGC layers then per-task towers + PredictionLayer.
 * Reference: https://dl.acm.org/doi/abs/10.1145/3383313.3412236
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
public class PLE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nTask;
    private final int nLevel;
    private final EmbeddingLayer embedding;
    private final List<CGC> cgcRefs = new ArrayList<>();
    private final List<MLP> towers = new ArrayList<>();
    private final List<PredictionLayer> predictLayers = new ArrayList<>();

    public PLE(List<Feature> features, List<String> taskTypes) {
        this(features, taskTypes, 3, 1, 1, Collections.emptyMap(), Collections.emptyList(), DeviceSupport.backend());
    }

    public PLE(
            List<Feature> features,
            List<String> taskTypes,
            int nLevel,
            int nExpertSpecific,
            int nExpertShared,
            Map<String, Object> expertParams,
            List<Map<String, Object>> towerParamsList,
            String device) {
        super("PLE");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("PLE: features cannot be empty");
        }
        if (taskTypes == null || taskTypes.isEmpty()) {
            throw new IllegalArgumentException("PLE: taskTypes cannot be empty");
        }
        if (nLevel <= 0) {
            throw new IllegalArgumentException("PLE: nLevel must be > 0, got " + nLevel);
        }
        if (nExpertSpecific < 0) {
            throw new IllegalArgumentException("PLE: nExpertSpecific must be >= 0, got " + nExpertSpecific);
        }
        if (nExpertShared < 0) {
            throw new IllegalArgumentException("PLE: nExpertShared must be >= 0, got " + nExpertShared);
        }
        for (String t : taskTypes) {
            if (!"classification".equals(t) && !"regression".equals(t)) {
                throw new IllegalArgumentException("PLE: taskTypes must be 'classification' or 'regression'");
            }
        }

        this.nTask = taskTypes.size();
        this.nLevel = nLevel;

        int inputDims = 0;
        for (Feature f : features) {
            inputDims += f.embedDim();
        }

        @SuppressWarnings("unchecked")
        List<Long> dimsList = expertParams != null && expertParams.containsKey("dims")
                ? (List<Long>) expertParams.get("dims")
                : List.of(128L);
        long expertLast = dimsList.get(dimsList.size() - 1);

        this.embedding = new EmbeddingLayer(features, features.get(0).embedDim(), device);
        register_module("embedding", embedding);

        for (int level = 0; level < nLevel; level++) {
            int levelInput = level == 0 ? inputDims : (int) expertLast;
            CGC cgc = new CGC(level + 1, nLevel, nTask, nExpertSpecific, nExpertShared,
                    levelInput, expertParams, device);
            register_module("cgc_" + level, cgc);
            cgcRefs.add(cgc);
        }

        for (int i = 0; i < nTask; i++) {
            Map<String, Object> params = (towerParamsList == null || towerParamsList.isEmpty())
                    ? Collections.emptyMap()
                    : towerParamsList.get(i);
            @SuppressWarnings("unchecked")
            List<Long> tDims = params.containsKey("dims")
                    ? (List<Long>) params.get("dims")
                    : List.of(expertLast);
            String activation = params.containsKey("activation")
                    ? String.valueOf(params.get("activation"))
                    : "relu";
            float dropout = params.containsKey("dropout")
                    ? ((Number) params.get("dropout")).floatValue()
                    : 0.0f;
            long[] towerDims = toLongArray(tDims);
            MLP m = new MLP(expertLast, towerDims, 1L, activation, dropout, false, false, true, device);
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

        // ple_inputs starts as [embed_x] * (n_task + 1)
        List<Tensor> pleInputs = new ArrayList<>();
        for (int i = 0; i < nTask + 1; i++) {
            pleInputs.add(embedX);
        }

        for (int level = 0; level < nLevel; level++) {
            pleInputs = cgcRefs.get(level).forward(pleInputs);
        }

        List<Tensor> ys = new ArrayList<>();
        for (int ti = 0; ti < nTask; ti++) {
            Tensor towerOut = towers.get(ti).forward(pleInputs.get(ti));
            ys.add(predictLayers.get(ti).forward(towerOut));
        }

        TensorVector vec = new TensorVector(ys.toArray(new Tensor[0]));
        return torch.cat(vec, 1L);
    }

    private static long[] toLongArray(List<Long> list) {
        long[] arr = new long[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
