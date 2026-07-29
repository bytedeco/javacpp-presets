/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/ETA.scala
 *
 * End-to-End Target Attention (ETA).
 * LSH-based retrieval + target attention over long history.
 * Reference: Alibaba end-to-end target attention.
 *
 * Note: uses basic.layers.ActivationUnit (item1/item2 API), not DINActivationUnit.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.ActivationUnit;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ETA extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<SequenceFeature> seqFeatures;
    private final int embedDim;
    private final int hashSize;
    private final String device;
    private final EmbeddingLayer featureEmbedding;
    private final EmbeddingLayer seqEmbedding;
    private final LinearImpl hashProjection;
    private final ActivationUnit attentionNet;
    private final MLP mlp;

    public ETA(List<? extends Feature> features, List<SequenceFeature> seqFeatures) {
        this(features, seqFeatures, 8, 64, 36, 20, new long[]{256L, 128L, 64L}, 0.2f,
                DeviceSupport.backend());
    }

    public ETA(List<? extends Feature> features, List<SequenceFeature> seqFeatures,
               int embedDim, int hashSize, int attentionUnits, int topK,
               long[] mlpDims, float dropout, String device) {
        super("ETA");
        // topK kept for API parity (Scala builds it but uses full-sequence attention).
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (seqFeatures == null || seqFeatures.isEmpty()) {
            throw new IllegalArgumentException("seqFeatures cannot be empty");
        }
        this.seqFeatures = new ArrayList<>(seqFeatures);
        this.embedDim = embedDim;
        this.hashSize = hashSize;
        this.device = device;

        this.featureEmbedding = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("featureEmbedding", featureEmbedding);

        this.seqEmbedding = new EmbeddingLayer(new ArrayList<>(this.seqFeatures), embedDim, device);
        register_module("seqEmbedding", seqEmbedding);

        long sparseDim = Features.calcSparseDim(new ArrayList<>(features));

        this.hashProjection = new LinearImpl(embedDim, hashSize);
        hashProjection.to(new Device(device), false);
        register_module("hashProjection", hashProjection);

        this.attentionNet = new ActivationUnit(embedDim, attentionUnits, "dice", device);
        register_module("attentionNet", attentionNet);

        long totalDim = sparseDim + embedDim * 2L;
        this.mlp = new MLP(totalDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);
    }

    private Tensor normalizeSequenceEmb(Tensor raw, int batchSize, int seqLen) {
        if (raw.dim() == 4L && raw.size(1) == 1L) {
            return raw.squeeze(1L);
        } else if (raw.dim() == 3L) {
            return raw;
        } else if (raw.dim() == 2L) {
            long total = raw.size(1);
            long expected = (long) batchSize * seqLen * embedDim;
            if (seqLen > 0 && total == (long) seqLen * embedDim) {
                return raw.view(batchSize, seqLen, embedDim);
            } else if (seqLen > 0 && raw.numel() == expected) {
                return raw.view(batchSize, seqLen, embedDim);
            } else if (total == embedDim) {
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
            if (raw.size(1) == embedDim) return raw;
            if (raw.numel() == (long) batchSize * embedDim) {
                return raw.view(batchSize, embedDim);
            }
            return raw;
        }
        return raw;
    }

    private Tensor safeSequenceEmbedding(String name, Tensor indices, int batchSize) {
        Tensor raw = seqEmbedding.getSequenceEmbedding(name, indices);
        if (raw.dim() == 3L) return raw;
        if (raw.dim() == 2L && raw.size(1) == embedDim) return raw.unsqueeze(1L);
        if (raw.dim() == 2L && raw.numel() == batchSize * indices.size(1) * embedDim) {
            return raw.view(batchSize, indices.size(1), embedDim);
        }
        return raw;
    }

    private Tensor onModelDevice(Tensor t) {
        try {
            if ("cpu".equals(device)) {
                return t.toType(ScalarType.Float);
            }
            return t.to(new Device(device), ScalarType.Float);
        } catch (Throwable e) {
            return t.toType(ScalarType.Float);
        }
    }

    public Tensor forward(Map<String, Tensor> sparseFeats,
                          Map<String, Tensor> seqFeats,
                          Map<String, Tensor> targetFeats) {
        Tensor featEmb = onModelDevice(featureEmbedding.forward(sparseFeats));

        int batchSize = 1;
        if (!seqFeats.isEmpty()) {
            batchSize = (int) seqFeats.values().iterator().next().size(0);
        } else if (!targetFeats.isEmpty()) {
            batchSize = (int) targetFeats.values().iterator().next().size(0);
        }

        if (targetFeats.isEmpty()) {
            throw new IllegalArgumentException("targetFeats cannot be empty");
        }
        Map.Entry<String, Tensor> targetEntry = targetFeats.entrySet().iterator().next();
        Tensor targetRaw = seqEmbedding.getSequenceEmbedding(targetEntry.getKey(), targetEntry.getValue());
        Tensor targetEmb = onModelDevice(normalizeTargetEmb(targetRaw, batchSize));
        Tensor targetFlat = targetEmb.dim() == 2L ? targetEmb : targetEmb.mean(1);

        List<Tensor> seqEmbs = new ArrayList<>();
        for (SequenceFeature f : seqFeatures) {
            Tensor indices = seqFeats.get(f.name());
            int seqLen = (int) indices.size(1);
            seqEmbs.add(normalizeSequenceEmb(
                    safeSequenceEmbedding(f.name(), indices, batchSize), batchSize, seqLen));
        }
        Tensor seqEmb;
        if (seqEmbs.size() == 1) {
            seqEmb = seqEmbs.get(0);
        } else {
            TensorVector vec = new TensorVector();
            for (Tensor t : seqEmbs) vec.push_back(t);
            seqEmb = torch.cat(vec, 1);
        }
        Tensor seqEmbAligned = onModelDevice(seqEmb);
        int seqLen = (int) seqEmb.size(1);

        // LSH: hash target item
        Tensor targetHashed = hashProjection.forward(targetFlat);
        Tensor targetSign = targetHashed.ge(new Scalar(0.0f)).toType(ScalarType.Float);

        // Hash all history items
        Tensor seqFlat = seqEmbAligned.view((long) batchSize * seqLen, embedDim);
        Tensor seqHashed = hashProjection.forward(seqFlat);
        Tensor seqSign = seqHashed.ge(new Scalar(0.0f)).toType(ScalarType.Float);
        Tensor seqHashed2D = seqSign.view(batchSize, seqLen, hashSize);

        // Hamming similarity
        Tensor matchMask = targetSign.unsqueeze(1).eq(seqHashed2D).toType(ScalarType.Float);
        Tensor hammingSim = matchMask.sum(2L); // (batch, seq_len)

        // Target attention over full sequence (simplified ETA without topk gather)
        Tensor targetExpanded = onModelDevice(
                targetFlat.unsqueeze(1).repeat(1L, seqLen, 1L));

        Tensor attnScores = seqEmbAligned.mul(targetExpanded).sum(2L);
        Tensor combinedScores = attnScores.add(hammingSim);
        Tensor attnWeights = combinedScores.unsqueeze(2)
                .div(new Scalar((float) Math.sqrt(embedDim)))
                .softmax(1L);

        Tensor attendedLong = seqEmbAligned.mul(attnWeights).sum(1L);
        Tensor attendedShort = seqEmbAligned.mean(1L);

        TensorVector cVec = new TensorVector();
        cVec.push_back(onModelDevice(featEmb));
        cVec.push_back(onModelDevice(attendedLong));
        cVec.push_back(onModelDevice(attendedShort));
        Tensor combined = torch.cat(cVec, 1L);

        return mlp.forward(combined);
    }
}
