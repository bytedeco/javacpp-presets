/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/BST.scala
 *
 * Behavior Sequence Transformer (BST).
 * Reference: Alibaba, SIGIR 2019
 *
 * Note: Scala intentionally bypasses the full transformer encoder path and uses
 * the last position of (history+target+pos) as interest representation.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BST extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> features;
    private final List<SequenceFeature> sequenceFeatures;
    private final List<SequenceFeature> targetFeatures;
    private final int embedDim;
    private final int itemDim;
    private final int maxSeqLen;
    private final EmbeddingLayer embedding;
    private final EmbeddingImpl posEmbedding;
    private final MLP mlp;
    // Encoder layers constructed for API parity / future use (Scala also builds them).
    @SuppressWarnings("unused")
    private final List<BSTEncoderLayer> encoderLayers = new ArrayList<>();

    public BST(List<? extends Feature> features,
               List<SequenceFeature> sequenceFeatures,
               List<SequenceFeature> targetFeatures) {
        this(features, sequenceFeatures, targetFeatures, 8, 8, 1, 51,
                new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public BST(List<? extends Feature> features,
               List<SequenceFeature> sequenceFeatures,
               List<SequenceFeature> targetFeatures,
               int embedDim, int numHeads, int numLayers, int maxSeqLen,
               long[] mlpDims, float dropout, String device) {
        super("BST");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (sequenceFeatures == null || sequenceFeatures.isEmpty()) {
            throw new IllegalArgumentException("sequenceFeatures cannot be empty");
        }
        if (targetFeatures == null || targetFeatures.isEmpty()) {
            throw new IllegalArgumentException("targetFeatures cannot be empty");
        }
        this.features = new ArrayList<>(features);
        this.sequenceFeatures = new ArrayList<>(sequenceFeatures);
        this.targetFeatures = new ArrayList<>(targetFeatures);
        this.embedDim = embedDim;
        this.maxSeqLen = maxSeqLen;

        int histDim = 0;
        for (SequenceFeature sf : this.sequenceFeatures) histDim += sf.embedDim();
        int tgtDim = 0;
        for (SequenceFeature sf : this.targetFeatures) tgtDim += sf.embedDim();
        if (histDim != tgtDim) {
            throw new IllegalArgumentException("sequence and target feature dims must match");
        }
        if (histDim % numHeads != 0) {
            throw new IllegalArgumentException("itemDim must be divisible by numHeads");
        }
        this.itemDim = histDim;

        Device targetDevice = new Device(device);

        List<Feature> allFeatures = new ArrayList<>();
        allFeatures.addAll(this.features);
        allFeatures.addAll(this.sequenceFeatures);
        allFeatures.addAll(this.targetFeatures);
        this.embedding = new EmbeddingLayer(allFeatures, embedDim, device);
        register_module("embedding", embedding);

        this.posEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxSeqLen + 1L, itemDim));
        posEmbedding.to(targetDevice, false);
        register_module("pos_embedding", posEmbedding);

        for (int i = 0; i < numLayers; i++) {
            BSTEncoderLayer layer = new BSTEncoderLayer(itemDim, numHeads, dropout, device);
            register_module("encoder_" + i, layer);
            encoderLayers.add(layer);
        }

        long allDims = itemDim + tgtDim;
        for (Feature f : this.features) allDims += f.embedDim();
        this.mlp = new MLP(allDims, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    private Tensor normalizeHistoryEmb(Tensor raw, int batchSize, int seqLen) {
        if (raw.dim() == 4L && raw.size(1) == 1L) {
            return raw.squeeze(1L);
        } else if (raw.dim() == 3L) {
            return raw;
        } else if (raw.dim() == 2L) {
            long total = raw.size(1);
            long expected = (long) batchSize * seqLen * itemDim;
            if (seqLen > 0 && total == (long) seqLen * itemDim) {
                return raw.view(batchSize, seqLen, itemDim);
            } else if (seqLen > 0 && raw.numel() == expected) {
                return raw.view(batchSize, seqLen, itemDim);
            } else if (total == itemDim) {
                return raw.unsqueeze(1L);
            }
            return raw;
        }
        return raw;
    }

    private Tensor normalizeTargetEmb(Tensor raw, int batchSize) {
        if (raw.dim() == 3L && raw.size(1) == 1L) {
            return raw.squeeze(1L);
        } else if (raw.dim() == 2L) {
            if (raw.size(1) == itemDim) return raw;
            if (raw.numel() == (long) batchSize * itemDim) {
                return raw.view(batchSize, itemDim);
            }
            return raw;
        }
        return raw;
    }

    public Tensor forward(Map<String, Tensor> sparseFeats, Map<String, Tensor> seqFeats) {
        Map<String, Tensor> historyFeats = new LinkedHashMap<>();
        for (SequenceFeature f : sequenceFeatures) {
            Tensor t = seqFeats.get(f.name());
            if (t != null) historyFeats.put(f.name(), t);
        }

        int batchSize = 1;
        int seqLen = 1;
        if (!historyFeats.isEmpty()) {
            Tensor first = historyFeats.values().iterator().next();
            batchSize = (int) first.size(0);
            seqLen = (int) first.size(1);
        } else if (!seqFeats.isEmpty()) {
            for (Map.Entry<String, Tensor> e : seqFeats.entrySet()) {
                if (!"target_feat".equals(e.getKey())) {
                    batchSize = (int) e.getValue().size(0);
                    seqLen = (int) e.getValue().size(1);
                    break;
                }
            }
            if (batchSize == 1) {
                Tensor any = seqFeats.values().iterator().next();
                batchSize = (int) any.size(0);
                seqLen = (int) any.size(1);
            }
        }

        Tensor hist = normalizeHistoryEmb(embedding.forwardSeqRaw(historyFeats), batchSize, seqLen);

        Tensor sparseEmbeddings = embedding.forward(sparseFeats);
        Tensor contextEmbeddings = sparseEmbeddings.view(batchSize, features.size(), embedDim);

        Map<String, Tensor> targetFeats = new LinkedHashMap<>();
        for (SequenceFeature f : targetFeatures) {
            Tensor t = sparseFeats.get(f.name());
            if (t == null) t = seqFeats.get(f.name());
            if (t != null) targetFeats.put(f.name(), t);
        }
        if (targetFeats.isEmpty()) {
            throw new IllegalArgumentException(
                    "Target features must be provided via sparseFeats or seqFeats");
        }

        Tensor tgt = normalizeTargetEmb(
                embedding.forward(Collections.emptyMap(), targetFeats, false), batchSize);

        // Append target to end of sequence: (batch, seqLen + 1, itemDim)
        Tensor tgtExpanded = tgt.unsqueeze(1);
        TensorVector sVec = new TensorVector();
        sVec.push_back(hist);
        sVec.push_back(tgtExpanded);
        Tensor seq = torch.cat(sVec, 1);

        if (seq.size(1) > maxSeqLen + 1L) {
            throw new IllegalArgumentException(
                    "sequence length " + seq.size(1) + " exceeds max_seq_len " + maxSeqLen);
        }

        // Add positional encoding
        long finalSeqLen = seq.size(1);
        Tensor positions = torch.arange(
                new Scalar(finalSeqLen),
                new TensorOptions().device(new DeviceOptional(seq.device())));
        Tensor posEnc = posEmbedding.forward(positions.unsqueeze(0));
        Tensor seqWithPos = seq.add(posEnc);

        // Scala intentionally bypasses full transformer; use last position as interest.
        Tensor interest = seqWithPos.select(1, seqWithPos.size(1) - 1);

        Tensor contextFlat = contextEmbeddings.view(batchSize, -1);
        TensorVector cVec = new TensorVector();
        cVec.push_back(interest);
        cVec.push_back(tgt);
        cVec.push_back(contextFlat);
        Tensor combined = torch.cat(cVec, 1);

        return mlp.forward(combined);
    }
}
