/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DIN.scala
 *
 * Deep Interest Network (Alibaba, KDD'2018).
 * Reference: https://arxiv.org/abs/1706.06978
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
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
public class DIN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<Feature> features;
    private final List<SequenceFeature> sequenceFeatures;
    private final int historyDim;
    private final int targetDim;
    private final int allDims;
    private final EmbeddingLayer embedding;
    private final List<DINActivationUnit> attentionLayers = new ArrayList<>();
    private final MLP mlp;

    public DIN(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures) {
        this(features, sequenceFeatures, new long[]{256L, 128L}, 0.2f, 36, DeviceSupport.backend());
    }

    public DIN(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures,
               long[] mlpDims, float dropout, int attentionUnits, String device) {
        super("DIN");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("DIN: features cannot be empty");
        }
        if (sequenceFeatures == null || sequenceFeatures.isEmpty()) {
            throw new IllegalArgumentException("DIN: sequenceFeatures cannot be empty");
        }
        this.features = new ArrayList<>(features);
        this.sequenceFeatures = new ArrayList<>(sequenceFeatures);

        int contextDim = 0;
        for (Feature f : this.features) contextDim += f.embedDim();
        int histDim = 0;
        for (SequenceFeature sf : this.sequenceFeatures) histDim += sf.embedDim();
        this.historyDim = histDim;
        this.targetDim = histDim;
        this.allDims = contextDim + historyDim + targetDim;

        // features ++ sequenceFeatures ++ sequenceFeatures (targets share with history)
        List<Feature> allFeats = new ArrayList<>();
        allFeats.addAll(this.features);
        allFeats.addAll(this.sequenceFeatures);
        allFeats.addAll(this.sequenceFeatures);
        this.embedding = new EmbeddingLayer(allFeats, 8, device);
        register_module("embedding", embedding);

        for (SequenceFeature sf : this.sequenceFeatures) {
            DINActivationUnit unit = new DINActivationUnit(sf.embedDim(), attentionUnits, device);
            register_module("attentionUnit_" + sf.name(), unit);
            attentionLayers.add(unit);
        }

        this.mlp = new MLP(allDims, mlpDims, 1L, "dice", dropout, false, device);
        register_module("mlp", mlp);
    }

    /** Primary forward — per-field target feature map. */
    public Tensor forward(Map<String, Tensor> sparseFeats,
                          Map<String, Tensor> sequenceFeats,
                          Map<String, Tensor> targetFeats) {
        Tensor[] att = computeAttentions(sparseFeats, sequenceFeats, targetFeats);
        Tensor pooled = att[0];
        Tensor flatTarget = att[1];
        Tensor embedCtx = embedding.forward(sparseFeats, Collections.emptyMap(), true);
        TensorVector vec = new TensorVector();
        vec.push_back(pooled.flatten(1L, 2L));
        vec.push_back(flatTarget);
        vec.push_back(embedCtx);
        Tensor mlpIn = torch.cat(vec, 1L);
        return mlp.forward(mlpIn).squeeze(1L);
    }

    /** Backward-compat: single target index broadcast over every history field. */
    public Tensor forward(Map<String, Tensor> sparseFeats,
                          Map<String, Tensor> sequenceFeats,
                          Tensor targetIdx) {
        Map<String, Tensor> targetFeats = new LinkedHashMap<>();
        for (SequenceFeature sf : sequenceFeatures) {
            targetFeats.put(sf.name(), targetIdx);
        }
        return forward(sparseFeats, sequenceFeats, targetFeats);
    }

    /** Ensure target is [B, D] (last step if sequence-shaped). */
    private static Tensor asTargetVector(Tensor emb) {
        if (emb == null) throw new IllegalArgumentException("target emb is null");
        if (emb.dim() == 3L) {
            // [B, S, D] → last position (candidate item)
            return emb.select(1, emb.size(1) - 1);
        }
        if (emb.dim() == 2L) return emb;
        if (emb.dim() == 1L) return emb.unsqueeze(0L);
        throw new IllegalArgumentException("unexpected target emb rank=" + emb.dim()
                + " shape=" + emb.sizes());
    }

    /** Ensure history is [B, S, D]. */
    private static Tensor asHistorySequence(Tensor emb) {
        if (emb == null) throw new IllegalArgumentException("history emb is null");
        if (emb.dim() == 3L) return emb;
        if (emb.dim() == 2L) return emb.unsqueeze(1L); // [B,D] → [B,1,D]
        throw new IllegalArgumentException("unexpected history emb rank=" + emb.dim()
                + " shape=" + emb.sizes());
    }

    /** Returns [attentionPooled (B, numSeq, D_sum-ish), flatTarget (B, targetDim)]. */
    private Tensor[] computeAttentions(Map<String, Tensor> sparseFeats,
                                       Map<String, Tensor> sequenceFeats,
                                       Map<String, Tensor> targetFeats) {
        // Use raw sequence embeddings (no mean-pool) so ActivationUnit can attend over time.
        Map<String, Tensor> historyByName = new LinkedHashMap<>();
        for (SequenceFeature sf : sequenceFeatures) {
            Tensor indices = sequenceFeats.get(sf.name());
            if (indices == null) {
                throw new IllegalArgumentException("missing sequence feature: " + sf.name());
            }
            Tensor emb = embedding.getSequenceEmbedding(sf.name(), indices);
            historyByName.put(sf.name(), asHistorySequence(emb));
        }

        Map<String, Tensor> targetByName = new LinkedHashMap<>();
        for (SequenceFeature sf : sequenceFeatures) {
            Tensor indices = targetFeats.get(sf.name());
            if (indices == null) {
                throw new IllegalArgumentException("missing target feature: " + sf.name());
            }
            Tensor emb = embedding.getSequenceEmbedding(sf.name(), indices);
            targetByName.put(sf.name(), asTargetVector(emb));
        }

        List<Tensor> pooled = new ArrayList<>();
        for (int i = 0; i < sequenceFeatures.size(); i++) {
            SequenceFeature sf = sequenceFeatures.get(i);
            Tensor h = historyByName.get(sf.name());
            Tensor t = targetByName.get(sf.name());
            // ActivationUnit → [B, D]; stack as [B, 1, D] for cat over fields
            Tensor att = attentionLayers.get(i).forward(h, t);
            pooled.add(att.unsqueeze(1L));
        }
        TensorVector pVec = new TensorVector();
        for (Tensor t : pooled) pVec.push_back(t);
        Tensor attentionPooled = torch.cat(pVec, 1L); // [B, numSeq, D]

        TensorVector tVec = new TensorVector();
        for (SequenceFeature sf : sequenceFeatures) {
            tVec.push_back(targetByName.get(sf.name()));
        }
        Tensor flatTarget = torch.cat(tVec, 1L); // [B, targetDim]
        return new Tensor[]{attentionPooled, flatTarget};
    }
}
