/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/SharedBottom.scala
 *
 * SharedBottom — Caruana 1997 multi-task baseline.
 * One shared bottom MLP, then per-task towers and prediction layers.
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
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
public class SharedBottom extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nTask;
    private final EmbeddingLayer embedding;
    private final MLP bottomMlp;
    private final ModuleListImpl towers = new ModuleListImpl();
    private final ModuleListImpl predictLayers = new ModuleListImpl();

//    private final List<MLP> towers = new ArrayList<>();
//    private final List<PredictionLayer> predictLayers = new ArrayList<>();

    public SharedBottom(List<? extends Feature> features, List<String> taskTypes) {
        this(features, taskTypes, Collections.emptyMap(), Collections.emptyList(), DeviceSupport.backend());
    }

    public SharedBottom(List<? extends Feature> features, List<String> taskTypes,
                        Map<String, Object> bottomParams,
                        List<Map<String, Object>> towerParamsList,
                        String device) {
        super("SharedBottom");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("SharedBottom: features cannot be empty");
        }
        if (taskTypes == null || taskTypes.isEmpty()) {
            throw new IllegalArgumentException("SharedBottom: taskTypes cannot be empty");
        }
        for (String t : taskTypes) {
            if (!"classification".equals(t) && !"regression".equals(t)) {
                throw new IllegalArgumentException(
                        "SharedBottom: taskTypes must be 'classification' or 'regression'");
            }
        }

        List<Feature> featList = new ArrayList<>(features);
        this.nTask = taskTypes.size();
        int bottomDims = 0;
        for (Feature f : featList) bottomDims += f.embedDim();

        if (bottomParams == null) bottomParams = Collections.emptyMap();
        @SuppressWarnings("unchecked")
        List<Long> bottomDimsList = bottomParams.containsKey("dims")
                ? (List<Long>) bottomParams.get("dims")
                : Collections.singletonList(128L);
        long[] bottomArr = toLongArray(bottomDimsList);
        String bottomActivation = bottomParams.containsKey("activation")
                ? bottomParams.get("activation").toString() : "relu";
        float bottomDropout = bottomParams.containsKey("dropout")
                ? ((Number) bottomParams.get("dropout")).floatValue() : 0.0f;
        long towerLast = bottomArr[bottomArr.length - 1];

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        this.bottomMlp = new MLP(bottomDims, bottomArr, towerLast, bottomActivation, bottomDropout,
                false, false, false, device);
        register_module("bottom_mlp", bottomMlp);

        if (towerParamsList == null) towerParamsList = Collections.emptyList();
        for (int i = 0; i < nTask; i++) {
            Map<String, Object> params = towerParamsList.isEmpty()
                    ? Collections.emptyMap() : towerParamsList.get(i);
            @SuppressWarnings("unchecked")
            List<Long> dimsList = params.containsKey("dims")
                    ? (List<Long>) params.get("dims")
                    : Collections.singletonList(towerLast);
            long[] dims = toLongArray(dimsList);
            String activation = params.containsKey("activation")
                    ? params.get("activation").toString() : "relu";
            float dropout = params.containsKey("dropout")
                    ? ((Number) params.get("dropout")).floatValue() : 0.0f;
            MLP m = new MLP(towerLast, dims, 1L, activation, dropout, false, false, true, device);
            register_module("tower_" + i, m);
            towers.insert(i, m);
        }

        for (int i = 0; i < nTask; i++) {
            PredictionLayer m = new PredictionLayer(taskTypes.get(i));
            register_module("predictLayer_" + i, m);
            predictLayers.insert(i, m);
        }
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor inputBottom = embedding.forward(x, Collections.emptyMap(), true);
        Tensor shared = bottomMlp.forward(inputBottom);

        TensorVector ys = new TensorVector();
        for (int i = 0; i < nTask; i++) {
            Tensor towerOut = towers.get(i).forward(shared);
            ys.push_back(predictLayers.get(i).forward(towerOut));
        }
        return torch.cat(ys, 1L);
    }

    private static long[] toLongArray(List<Long> list) {
        long[] a = new long[list.size()];
        for (int i = 0; i < list.size(); i++) a[i] = list.get(i);
        return a;
    }
}
