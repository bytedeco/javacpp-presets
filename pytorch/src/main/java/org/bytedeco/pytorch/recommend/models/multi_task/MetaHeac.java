/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/MetaHeac.scala
 *
 * Meta Hybrid Experts and Critics (MetaHeac).
 * Reference: Zhu et al., KDD 2021 — "Learning to Expand Audience via Meta Hybrid Experts and Critics"
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MetaHeac extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<String> taskNames;
    private final int embedDim;
    private final int expertNum;
    private final int criticNum;
    private final int numSparseFeatures;
    private final Device targetDevice;
    private final EmbeddingLayer embedding;
    private final Map<String, MetaEmbedding> taskEmbeddingsMap = new LinkedHashMap<>();
    private final List<MLP> expertsList = new ArrayList<>();
    private final MetaLinear expertGate;
    private final Map<String, List<MLP>> taskCriticsMap = new LinkedHashMap<>();
    private final Map<String, MetaLinear> criticGatesMap = new LinkedHashMap<>();

    public MetaHeac(List<Feature> features, List<String> taskNames) {
        this(features, taskNames, 8, new long[]{128L, 64L}, new long[]{32L, 16L}, 4, 5, 0.2f, DeviceSupport.backend());
    }

    public MetaHeac(
            List<Feature> features,
            List<String> taskNames,
            int embedDim,
            long[] bottomDims,
            long[] towerDims,
            int expertNum,
            int criticNum,
            float dropout,
            String device) {
        super("MetaHeac");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (taskNames == null || taskNames.isEmpty()) {
            throw new IllegalArgumentException("taskNames cannot be empty");
        }
        if (expertNum < 1) {
            throw new IllegalArgumentException("expertNum must be >= 1, got " + expertNum);
        }
        if (criticNum < 1) {
            throw new IllegalArgumentException("criticNum must be >= 1, got " + criticNum);
        }

        this.taskNames = new ArrayList<>(taskNames);
        this.embedDim = embedDim;
        this.expertNum = expertNum;
        this.criticNum = criticNum;
        this.targetDevice = new Device(device);

        int sparseCount = 0;
        for (Feature f : features) {
            if (f instanceof SparseFeature) {
                sparseCount++;
            }
        }
        this.numSparseFeatures = sparseCount;

        long sparseDim = Features.calcSparseDim(features);
        long bottomLast = bottomDims[bottomDims.length - 1];

        this.embedding = new EmbeddingLayer(features, embedDim, device);
        register_module("embedding", embedding);

        int taskNum = taskNames.size();
        for (int i = 0; i < taskNum; i++) {
            String name = "taskEmbedding_" + i;
            MetaEmbedding emb = new MetaEmbedding(taskNum, embedDim, device);
            register_module(name, emb);
            taskEmbeddingsMap.put(name, emb);
        }

        for (int i = 0; i < expertNum; i++) {
            String name = "expert_" + i;
            MLP expert = new MLP(sparseDim, bottomDims, bottomLast, "relu", dropout, false, false, false, device);
            register_module(name, expert);
            expertsList.add(expert);
        }

        this.expertGate = new MetaLinear(embedDim * 2L, expertNum, device);
        this.expertGate.to(targetDevice, false);
        register_module("expertGate", expertGate);

        for (String taskName : taskNames) {
            List<MLP> critics = new ArrayList<>();
            for (int i = 0; i < criticNum; i++) {
                String name = "critic_" + taskName + "_" + i;
                MLP critic = new MLP((int) bottomLast, towerDims, 1L, "relu", dropout, false, false, true, device);
                register_module(name, critic);
                critics.add(critic);
            }
            taskCriticsMap.put(taskName, critics);
        }

        for (int i = 0; i < taskNum; i++) {
            String name = "criticGate_" + i;
            MetaLinear gate = new MetaLinear(embedDim * 2L, criticNum, device);
            gate.to(targetDevice, false);
            register_module(name, gate);
            criticGatesMap.put(name, gate);
        }
    }

    /** Forward with task index tensor. */
    public Map<String, Tensor> forward(Map<String, Tensor> sparseFeats, Tensor taskIdx) {
        Tensor emb = embedding.forward(sparseFeats, Collections.emptyMap(), true);
        long batchSize = emb.size(0);

        Tensor pooled = emb.view(batchSize, (long) numSparseFeatures, (long) embedDim).mean(1);
        Tensor taskIdxCuda = taskIdx.to(targetDevice, ScalarType.Long);

        List<Tensor> expertOutputs = new ArrayList<>();
        for (MLP expert : expertsList) {
            expertOutputs.add(expert.forward(emb));
        }

        Map<String, Tensor> result = new LinkedHashMap<>();
        for (int idx = 0; idx < taskNames.size(); idx++) {
            String name = taskNames.get(idx);
            Tensor taskEmbBatched = taskEmbeddingsMap.get("taskEmbedding_" + idx).forward(taskIdxCuda);

            TensorVector expertGateInputVec = new TensorVector(taskEmbBatched, pooled);
            Tensor expertGateInput = torch.cat(expertGateInputVec, 1);
            Tensor expertWeights = torch.softmax(expertGate.forward(expertGateInput), 1);

            Tensor expertFused = null;
            for (int i = 0; i < expertOutputs.size(); i++) {
                Tensor weighted = expertWeights.select(1, i).unsqueeze(1).mul(expertOutputs.get(i));
                expertFused = expertFused == null ? weighted : expertFused.add(weighted);
            }

            TensorVector criticGateInputVec = new TensorVector(taskEmbBatched, pooled);
            Tensor criticGateInput = torch.cat(criticGateInputVec, 1);
            Tensor criticWeights = torch.softmax(criticGatesMap.get("criticGate_" + idx).forward(criticGateInput), 1);

            List<MLP> taskCritics = taskCriticsMap.get(name);
            List<Tensor> criticOutputs = new ArrayList<>();
            for (MLP critic : taskCritics) {
                criticOutputs.add(critic.forward(expertFused));
            }

            Tensor combinedOut = null;
            for (int i = 0; i < criticOutputs.size(); i++) {
                Tensor weighted = criticWeights.select(1, i).unsqueeze(1).mul(criticOutputs.get(i));
                combinedOut = combinedOut == null ? weighted : combinedOut.add(weighted);
            }

            result.put(name, torch.sigmoid(combinedOut));
        }
        return result;
    }

    /** Forward with explicit task name (no task index tensor needed). */
    public Map<String, Tensor> forwardByName(Map<String, Tensor> sparseFeats) {
        Tensor emb = embedding.forward(sparseFeats, Collections.emptyMap(), true);
        long batchSize = emb.size(0);

        Tensor pooled = emb.view(batchSize, (long) numSparseFeatures, (long) embedDim).mean(1);

        List<Tensor> expertOutputs = new ArrayList<>();
        for (MLP expert : expertsList) {
            expertOutputs.add(expert.forward(emb));
        }

        Map<String, Tensor> result = new LinkedHashMap<>();
        for (int idx = 0; idx < taskNames.size(); idx++) {
            String name = taskNames.get(idx);

            Tensor taskIdxTensor = torch.zeros(
                    new long[]{batchSize},
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
            taskIdxTensor = taskIdxTensor.to(targetDevice, ScalarType.Long);

            Tensor taskEmbBatched = taskEmbeddingsMap.get("taskEmbedding_" + idx).forward(taskIdxTensor);

            TensorVector expertGateInputVec = new TensorVector(taskEmbBatched, pooled);
            Tensor expertGateInput = torch.cat(expertGateInputVec, 1);
            Tensor expertWeights = torch.softmax(expertGate.forward(expertGateInput), 1);

            Tensor expertFused = null;
            for (int i = 0; i < expertOutputs.size(); i++) {
                Tensor weighted = expertWeights.select(1, i).unsqueeze(1).mul(expertOutputs.get(i));
                expertFused = expertFused == null ? weighted : expertFused.add(weighted);
            }

            TensorVector criticGateInputVec = new TensorVector(taskEmbBatched, pooled);
            Tensor criticGateInput = torch.cat(criticGateInputVec, 1);
            Tensor criticWeights = torch.softmax(criticGatesMap.get("criticGate_" + idx).forward(criticGateInput), 1);

            List<MLP> taskCritics = taskCriticsMap.get(name);
            List<Tensor> criticOutputs = new ArrayList<>();
            for (MLP critic : taskCritics) {
                criticOutputs.add(critic.forward(expertFused));
            }

            Tensor combinedOut = null;
            for (int i = 0; i < criticOutputs.size(); i++) {
                Tensor weighted = criticWeights.select(1, i).unsqueeze(1).mul(criticOutputs.get(i));
                combinedOut = combinedOut == null ? weighted : combinedOut.add(weighted);
            }

            result.put(name, torch.sigmoid(combinedOut));
        }
        return result;
    }
}
