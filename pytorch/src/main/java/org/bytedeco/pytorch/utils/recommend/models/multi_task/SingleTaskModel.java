/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/SingleTaskModel.scala
 *
 * Single-Task Model (Independent Multi-Task) — no parameter sharing.
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
public class SingleTaskModel extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int nTask;
    private final List<EmbeddingLayer> taskEmbeddings = new ArrayList<>();
    private final List<MLP> taskBottoms = new ArrayList<>();
    private final List<MLP> taskTowers = new ArrayList<>();

    public SingleTaskModel(List<? extends Feature> features, List<String> taskNames) {
        this(features, taskNames, 8, new long[]{128L}, new long[]{64L}, 0.2f, DeviceSupport.backend());
    }

    public SingleTaskModel(List<? extends Feature> features, List<String> taskNames,
                           int embedDim, long[] bottomDims, long[] towerDims,
                           float dropout, String device) {
        super("SingleTaskModel");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (taskNames == null || taskNames.isEmpty()) {
            throw new IllegalArgumentException("taskNames cannot be empty");
        }

        List<Feature> featList = new ArrayList<>(features);
        this.nTask = taskNames.size();
        int inputDims = 0;
        for (Feature f : featList) inputDims += f.embedDim();

        long bottomLast = bottomDims[bottomDims.length - 1];

        for (int i = 0; i < nTask; i++) {
            EmbeddingLayer emb = new EmbeddingLayer(featList, embedDim, device);
            register_module("embedding_" + i, emb);
            taskEmbeddings.add(emb);

            MLP bottom = new MLP(inputDims, bottomDims, bottomLast, "relu", dropout,
                    false, false, false, device);
            register_module("bottom_" + i, bottom);
            taskBottoms.add(bottom);

            MLP tower = new MLP(bottomLast, towerDims, 1L, "relu", dropout,
                    false, false, true, device);
            register_module("tower_" + i, tower);
            taskTowers.add(tower);
        }
    }

    public Tensor forward(Map<String, Tensor> x) {
        TensorVector outputs = new TensorVector();
        for (int i = 0; i < nTask; i++) {
            Tensor embeddings = taskEmbeddings.get(i).forward(x, Collections.emptyMap(), true);
            Tensor bottomOut = taskBottoms.get(i).forward(embeddings);
            Tensor towerOut = taskTowers.get(i).forward(bottomOut);
            outputs.push_back(torch.sigmoid(towerOut));
        }
        return torch.cat(outputs, 1);
    }
}
