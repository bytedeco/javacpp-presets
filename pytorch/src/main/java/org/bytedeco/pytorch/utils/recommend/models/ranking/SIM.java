/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/SIM.scala
 *
 * Search-based Interest Model (SIM).
 * Reference: Alibaba, CIKM 2020
 * Category hard/soft filtering + attention aggregation over long history.
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
public class SIM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<SequenceFeature> seqFeatures;
    private final List<SequenceFeature> cateFeatures;
    private final List<SequenceFeature> timeFeatures;
    private final int embedDim;
    private final String mode;
    private final float threshold;
    private final EmbeddingLayer featureEmbedding;
    private final EmbeddingLayer seqEmbedding;
    private final EmbeddingLayer cateEmbedding;
    private final EmbeddingLayer timeEmbedding;
    private final ActivationUnit attentionNet;
    private final LinearImpl seqProj;
    private final MLP mlp;

    public SIM(List<? extends Feature> features,
               List<SequenceFeature> seqFeatures,
               List<SequenceFeature> cateFeatures,
               List<SequenceFeature> timeFeatures) {
        this(features, seqFeatures, cateFeatures, timeFeatures, 8, 36, "hard", 0.8f,
                new long[]{256L, 128L, 64L}, 0.2f, DeviceSupport.backend());
    }

    public SIM(List<? extends Feature> features,
               List<SequenceFeature> seqFeatures,
               List<SequenceFeature> cateFeatures,
               List<SequenceFeature> timeFeatures,
               int embedDim, int attentionUnits, String mode, float threshold,
               long[] mlpDims, float dropout, String device) {
        super("SIM");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (seqFeatures == null || seqFeatures.isEmpty()) {
            throw new IllegalArgumentException("seqFeatures cannot be empty");
        }
        if (!"hard".equals(mode) && !"soft".equals(mode)) {
            throw new IllegalArgumentException("mode must be 'hard' or 'soft'");
        }
        this.seqFeatures = new ArrayList<>(seqFeatures);
        this.cateFeatures = cateFeatures != null ? new ArrayList<>(cateFeatures) : new ArrayList<>();
        this.timeFeatures = timeFeatures != null ? new ArrayList<>(timeFeatures) : new ArrayList<>();
        this.embedDim = embedDim;
        this.mode = mode;
        this.threshold = threshold;

        this.featureEmbedding = new EmbeddingLayer(new ArrayList<>(features), embedDim, device);
        register_module("featureEmbedding", featureEmbedding);

        this.seqEmbedding = new EmbeddingLayer(new ArrayList<>(this.seqFeatures), embedDim, device);
        register_module("seqEmbedding", seqEmbedding);

        this.cateEmbedding = new EmbeddingLayer(new ArrayList<>(this.cateFeatures), embedDim, device);
        register_module("cateEmbedding", cateEmbedding);

        this.timeEmbedding = new EmbeddingLayer(new ArrayList<>(this.timeFeatures), embedDim, device);
        register_module("timeEmbedding", timeEmbedding);

        long sparseDim = Features.calcSparseDim(new ArrayList<>(features));

        this.attentionNet = new ActivationUnit(embedDim * 2, attentionUnits, "dice", device);
        register_module("attentionNet", attentionNet);

        this.seqProj = new LinearImpl(embedDim * 2L, embedDim);
        seqProj.to(new Device(device), false);
        register_module("seqProj", seqProj);

        long totalDim = sparseDim + embedDim;
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
        if (raw.dim() == 3L) {
            // [B, S, D] → last token as target item; [B, 1, D] → squeeze
            if (raw.size(1) == 1L) return raw.squeeze(1L);
            return raw.select(1, raw.size(1) - 1);
        } else if (raw.dim() == 2L) {
            if (raw.size(1) == embedDim) return raw;
            if (raw.numel() == (long) batchSize * embedDim) {
                return raw.view(batchSize, embedDim);
            }
            // [B, S*D] flattened — take last embedDim slice if divisible
            if (raw.size(1) % embedDim == 0 && raw.size(1) > embedDim) {
                long s = raw.size(1) / embedDim;
                return raw.view(batchSize, s, embedDim).select(1, s - 1);
            }
            return raw;
        }
        return raw;
    }

    private Tensor safeSequenceEmbedding(EmbeddingLayer layer, String name, Tensor indices, int batchSize) {
        Tensor raw = layer.getSequenceEmbedding(name, indices);
        if (raw.dim() == 3L) return raw;
        if (raw.dim() == 2L && raw.size(1) == embedDim) return raw.unsqueeze(1L);
        if (raw.dim() == 2L && raw.numel() == batchSize * indices.size(1) * embedDim) {
            return raw.view(batchSize, indices.size(1), embedDim);
        }
        return raw;
    }

    public Tensor forward(Map<String, Tensor> sparseFeats,
                          Map<String, Tensor> seqFeats,
                          Map<String, Tensor> cateFeats,
                          Map<String, Tensor> timeFeats,
                          Map<String, Tensor> targetFeats) {
        Tensor featEmb = featureEmbedding.forward(sparseFeats);

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
        Tensor targetEmb = normalizeTargetEmb(targetRaw, batchSize);

        List<Tensor> seqEmbs = new ArrayList<>();
        for (SequenceFeature f : seqFeatures) {
            Tensor indices = seqFeats.get(f.name());
            int seqLen = (int) indices.size(1);
            seqEmbs.add(normalizeSequenceEmb(
                    safeSequenceEmbedding(seqEmbedding, f.name(), indices, batchSize),
                    batchSize, seqLen));
        }
        Tensor seqEmb = catAlongDim1(seqEmbs);

        List<Tensor> cateEmbs = new ArrayList<>();
        for (SequenceFeature f : cateFeatures) {
            cateEmbs.add(cateEmbedding.getSequenceEmbedding(f.name(), cateFeats.get(f.name())));
        }
        Tensor cateEmb = catAlongDim1(cateEmbs);

        List<Tensor> timeEmbs = new ArrayList<>();
        for (SequenceFeature f : timeFeatures) {
            Tensor indices = timeFeats.get(f.name());
            int seqLen = (int) indices.size(1);
            timeEmbs.add(normalizeSequenceEmb(
                    safeSequenceEmbedding(timeEmbedding, f.name(), indices, batchSize),
                    batchSize, seqLen));
        }
        Tensor timeEmb = catAlongDim1(timeEmbs);

        int seqLen = (int) seqEmb.size(1);

        Tensor targetExpanded = targetEmb.unsqueeze(1).repeat(1, seqLen, 1);

        TensorVector ttVec = new TensorVector();
        ttVec.push_back(targetExpanded);
        ttVec.push_back(timeEmb);
        Tensor targetCatTime = torch.cat(ttVec, 2);

        TensorVector scVec = new TensorVector();
        scVec.push_back(seqEmb);
        scVec.push_back(cateEmb);
        Tensor seqCatCate = torch.cat(scVec, 2);

        Tensor filteredSeq;
        if ("hard".equals(mode)) {
            filteredSeq = hardFilter(seqCatCate, targetCatTime, threshold, batchSize, seqLen);
        } else {
            filteredSeq = softFilter(seqCatCate, targetCatTime, threshold, batchSize, seqLen);
        }

        Tensor attendedSeq = applyAttention(filteredSeq, targetExpanded, batchSize, seqLen);

        TensorVector cVec = new TensorVector();
        cVec.push_back(featEmb);
        cVec.push_back(attendedSeq);
        Tensor combined = torch.cat(cVec, 1L);

        return mlp.forward(combined);
    }

    private Tensor catAlongDim1(List<Tensor> embs) {
        if (embs.isEmpty()) {
            throw new IllegalArgumentException("empty embedding list");
        }
        if (embs.size() == 1) return embs.get(0);
        TensorVector vec = new TensorVector();
        for (Tensor t : embs) vec.push_back(t);
        return torch.cat(vec, 1);
    }

    /** Hard filter: keep items with high category similarity using dot product. */
    private Tensor hardFilter(Tensor seqCatCate, Tensor targetCatTime, float threshold,
                              int batchSize, int seqLen) {
        Tensor sim = seqCatCate.mul(targetCatTime).sum(2).unsqueeze(2);
        Tensor mask = sim.gt(new Scalar(threshold)).toType(ScalarType.Float);
        return seqCatCate.mul(mask);
    }

    /** Soft filter: dot-product similarity weighted filtering. */
    private Tensor softFilter(Tensor seqCatCate, Tensor targetCatTime, float threshold,
                              int batchSize, int seqLen) {
        Tensor dotProd = seqCatCate.mul(targetCatTime).sum(2L).unsqueeze(2L);
        return dotProd.mul(seqCatCate);
    }

    /** Apply attention over filtered sequence to get aggregated interest representation. */
    private Tensor applyAttention(Tensor filteredSeq, Tensor target, int batchSize, int seqLen) {
        Tensor projected = seqProj.forward(filteredSeq);
        Tensor attnWeights = projected.mul(target).sum(2).unsqueeze(2);
        Scalar scale = new Scalar((float) Math.sqrt(embedDim));
        Tensor attnNorm = attnWeights.div(scale).softmax(1);
        return projected.mul(attnNorm).sum(1);
    }
}
