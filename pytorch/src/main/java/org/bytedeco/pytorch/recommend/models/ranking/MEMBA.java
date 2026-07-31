/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/MEMBA.scala
 *
 * MEMBA: Memory-Bidirection for Sequential Recommendation.
 * Two memory banks (forward/backward) with target-aware multi-head attention read.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MEMBA extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numMemorySlots;
    private final int numHeads;
    private final int headDim;
    private final Device targetDevice;
    private final EmbeddingLayer embeddingLayer;
    private final MLP forwardEncoder;
    private final MLP backwardEncoder;
    private final Tensor fwdMemBank;
    private final Tensor bwdMemBank;
    private final LinearImpl memoryGate;
    private final LinearImpl queryProj;
    private final LinearImpl keyProj;
    private final LinearImpl valueProj;
    private final MLP fusionMLP;
    private final MLP mlp;

    public MEMBA(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures) {
        this(features, sequenceFeatures, 8, 16, 2, new long[]{256L, 128L}, 0.2f,
                DeviceSupport.backend());
    }

    public MEMBA(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures,
                 int embedDim, int numMemorySlots, int numHeads, long[] mlpDims,
                 float dropout, String device) {
        super("MEMBA");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numMemorySlots = numMemorySlots;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.targetDevice = new Device(device);

        List<Feature> allFeatures = new ArrayList<>();
        allFeatures.addAll(features);
        allFeatures.addAll(sequenceFeatures);
        this.embeddingLayer = new EmbeddingLayer(allFeatures, embedDim, device);
        register_module("embeddingLayer", embeddingLayer);

        this.forwardEncoder = new MLP(embedDim, new long[]{embedDim}, embedDim, "relu", dropout, false, device);
        register_module("forwardEncoder", forwardEncoder);

        this.backwardEncoder = new MLP(embedDim, new long[]{embedDim}, embedDim, "relu", dropout, false, device);
        register_module("backwardEncoder", backwardEncoder);

        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor fwd = torch.zeros(new long[]{1L, numMemorySlots, embedDim}, opts)
                .to(targetDevice, ScalarType.Float);
        register_parameter("fwdMemBank", fwd);
        this.fwdMemBank = fwd;

        Tensor bwd = torch.zeros(new long[]{1L, numMemorySlots, embedDim}, opts)
                .to(targetDevice, ScalarType.Float);
        register_parameter("bwdMemBank", bwd);
        this.bwdMemBank = bwd;

        this.memoryGate = new LinearImpl(embedDim, embedDim);
        memoryGate.to(targetDevice, false);
        register_module("memoryGate", memoryGate);

        this.queryProj = new LinearImpl(embedDim, embedDim);
        this.keyProj = new LinearImpl(embedDim, embedDim);
        this.valueProj = new LinearImpl(embedDim, embedDim);
        queryProj.to(targetDevice, false);
        keyProj.to(targetDevice, false);
        valueProj.to(targetDevice, false);
        register_module("queryProj", queryProj);
        register_module("keyProj", keyProj);
        register_module("valueProj", valueProj);

        this.fusionMLP = new MLP(embedDim * 3L, new long[]{embedDim}, embedDim, "relu", dropout, false, device);
        register_module("fusionMLP", fusionMLP);

        // Final tower sees sparse context embedding (sum of sparse field dims) + fused memory (embedDim).
        long sparseDim = Features.calcSparseDim(features);
        if (sparseDim <= 0L) sparseDim = embedDim; // fallback if only sequence features
        this.mlp = new MLP(sparseDim + embedDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    public Tensor forward(Map<String, Tensor> sparseFeats,
                          Map<String, Tensor> sequenceFeats,
                          Tensor targetIdx) {
        long batchSize = targetIdx.size(0);

        Tensor featEmb;
        try {
            featEmb = embeddingLayer.forward(sparseFeats, Collections.emptyMap(), true);
        } catch (Exception e) {
            featEmb = torch.zeros(new long[]{batchSize, embedDim},
                    new TensorOptions()
                            .dtype(new ScalarTypeOptional(ScalarType.Float))
                            .device(new DeviceOptional(targetDevice)));
        }
        featEmb = featEmb.to(targetDevice, ScalarType.Float);

        Tensor seqEmb = embeddingLayer.forward(Collections.emptyMap(), sequenceFeats, false);
        Tensor seqEmbFixed = seqEmb.dim() == 2 ? seqEmb.unsqueeze(1) : seqEmb;
        seqEmbFixed = seqEmbFixed.to(targetDevice, ScalarType.Float);

        Tensor fwdPooled = seqEmbFixed.mean(1);
        Tensor fwdH = forwardEncoder.forward(fwdPooled);
        Tensor bwdH = backwardEncoder.forward(fwdPooled);

        Tensor fwdUpdate = memoryGate.forward(fwdH);
        Tensor bwdUpdate = memoryGate.forward(bwdH);

        Tensor fwdMem = fwdMemBank.expand(batchSize, numMemorySlots, embedDim);
        Tensor bwdMem = bwdMemBank.expand(batchSize, numMemorySlots, embedDim);

        // Target embedding from mean of history sequence (active Scala path)
        Tensor targetEmb = seqEmbFixed.mean(1).to(targetDevice, ScalarType.Float);

        Tensor readFwd = memoryAttentionRead(targetEmb, fwdMem, fwdUpdate);
        Tensor readBwd = memoryAttentionRead(targetEmb, bwdMem, bwdUpdate);

        TensorVector fVec = new TensorVector();
        fVec.push_back(targetEmb);
        fVec.push_back(readFwd);
        fVec.push_back(readBwd);
        Tensor fused = torch.cat(fVec, 1);
        Tensor fusedOut = fusionMLP.forward(fused);

        TensorVector cVec = new TensorVector();
        cVec.push_back(featEmb);
        cVec.push_back(fusedOut);
        Tensor combined = torch.cat(cVec, 1);
        Tensor logits = mlp.forward(combined);

        return logits.to(targetDevice, ScalarType.Float);
    }

    private Tensor memoryAttentionRead(Tensor query, Tensor memory, Tensor update) {
        Tensor gate = update.sigmoid().unsqueeze(1);
        Tensor gatedMem = memory.mul(gate);

        long batchSize = query.size(0);
        Tensor q = queryProj.forward(query).view(batchSize, numHeads, headDim);
        Tensor k = keyProj.forward(gatedMem)
                .view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2);
        Tensor v = valueProj.forward(gatedMem)
                .view(batchSize, numMemorySlots, numHeads, headDim).transpose(1, 2);

        float scale = (float) Math.sqrt(headDim);
        Tensor scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)).div(new Scalar(scale));
        Tensor attn = scores.squeeze(2).softmax(-1);

        Tensor out = torch.matmul(attn.unsqueeze(2), v).squeeze(2);
        return out.transpose(1, 2).contiguous().view(batchSize, embedDim);
    }
}
