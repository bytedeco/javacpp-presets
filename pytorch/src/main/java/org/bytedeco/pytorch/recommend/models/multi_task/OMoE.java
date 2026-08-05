/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/OMoE.scala
 *
 * One-gate Mixture-of-Experts (OMoE) — single shared gate across all tasks.
 * Architecture: Input → Embedding → Shared Gate → Expert Routing → Per-Task Towers → Outputs
 * Reference: Ma et al., KDD 2018 (single-gate variant of MMOE).
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class OMoE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int nTask;
    private final int numExperts;
    private final int numSparseFeatures;
    private final EmbeddingLayer embedding;
//    private final List<MLP> expertsList = new ArrayList<>();
//    private final LinearImpl gate;
//    private final List<MLP> towersList = new ArrayList<>();
    private final ModuleListImpl expertsList = new ModuleListImpl();
    private final LinearImpl gate;
    private final ModuleListImpl towersList = new ModuleListImpl();

    public OMoE(List<Feature> features, List<String> taskNames) {
        this(features, taskNames, 8, 4, new long[]{128L}, new long[]{64L}, 0.2f, DeviceSupport.backend());
    }

    public OMoE(
            List<Feature> features,
            List<String> taskNames,
            int embedDim,
            int numExperts,
            long[] expertDims,
            long[] towerDims,
            float dropout,
            String device) {
        super("OMoE");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (taskNames == null || taskNames.isEmpty()) {
            throw new IllegalArgumentException("taskNames cannot be empty");
        }
        if (numExperts < 1) {
            throw new IllegalArgumentException("numExperts must be >= 1, got " + numExperts);
        }
        this.embedDim = embedDim;
        this.nTask = taskNames.size();
        this.numExperts = numExperts;

        int sparseCount = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) {
                sparseCount++;
            }
        }
        this.numSparseFeatures = sparseCount;

        Device targetDevice = new Device(device);

        this.embedding = new EmbeddingLayer(features, embedDim, device);
        register_module("embedding", embedding);

        long expertLast = expertDims[expertDims.length - 1];
        for (int i = 0; i < numExperts; i++) {
            MLP expert = new MLP(embedDim, expertDims, expertLast, "relu", dropout, false, false, false, device);
            expert.to(targetDevice, false);
            register_module("expert_" + i, expert);
            expertsList.insert(i,expert);
        }

        LinearOptions opts = new LinearOptions(embedDim, numExperts);
        this.gate = new LinearImpl(opts);
        this.gate.to(targetDevice, false);
        register_module("gate", gate);

        for (int i = 0; i < nTask; i++) {
            MLP tower = new MLP((int) expertLast, towerDims, 1L, "relu", dropout, false, false, true, device);
            register_module("tower_" + i, tower);
            towersList.insert(i,tower);
        }
    }

    public Tensor forward(Map<String, Tensor> x) {
        Tensor embeddings = embedding.forward(x, Collections.emptyMap(), true);
        long batchSize = embeddings.size(0);

        Tensor reshaped = embeddings.view(batchSize, (long) numSparseFeatures, (long) embedDim);
        Tensor pooled = reshaped.sum(1);

        Tensor gateWeights = torch.softmax(gate.forward(pooled), 1);

        List<Tensor> expertOutputs = new ArrayList<>();
        for (int i = 0; i < expertsList.size(); i++) {
            expertOutputs.add(expertsList.get(i).forward(pooled));
        }
//        for (MLP expert : expertsList) {
//            expertOutputs.add(expert.forward(pooled));
//        }

        Tensor combined = null;
        for (int i = 0; i < expertOutputs.size(); i++) {
            Tensor weighted = gateWeights.select(1, i).unsqueeze(1).mul(expertOutputs.get(i));
            combined = combined == null ? weighted : combined.add(weighted);
        }

        List<Tensor> outputs = new ArrayList<>();
//        for (MLP tower : towersList) {
//            outputs.add(torch.sigmoid(tower.forward(combined)));
//        }
        for (int i = 0; i < towersList.size(); i++) {
            outputs.add(torch.sigmoid(towersList.get(i).forward(combined)));

        }

        TensorVector vec = new TensorVector(outputs.toArray(new Tensor[0]));
        return torch.cat(vec, 1);
    }
}
