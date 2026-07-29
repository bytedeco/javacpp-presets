/*
 * Ported from torch-rechub-scala: torchrec/models/matching/MIND.scala
 *
 * Multi-Interest Network with Dynamic Routing.
 * Reference: Alibaba, CIKM 2019
 */
package org.bytedeco.pytorch.utils.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MIND extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final SequenceFeature sequenceFeature;
    private final EmbeddingLayer featureEmbedding;
    private final EmbeddingLayer sequenceEmbedding;
    private final MindCapsuleNetwork capsuleNet;
    private final MLP tower;

    public MIND(List<? extends Feature> features, SequenceFeature sequenceFeature) {
        this(features, sequenceFeature, 8, 4, 4, new long[]{256L, 128L}, 0.2f,
                DeviceSupport.backend());
    }

    public MIND(List<? extends Feature> features, SequenceFeature sequenceFeature,
                int embedDim, int numInterests, int capsuleDim, long[] mlpDims,
                float dropout, String device) {
        super("MIND");
        this.sequenceFeature = sequenceFeature;

        List<Feature> featList = new ArrayList<>(features);
        this.featureEmbedding = new EmbeddingLayer(featList, embedDim, device);
        register_module("featureEmbedding", featureEmbedding);

        List<Feature> seqList = new ArrayList<>();
        seqList.add(sequenceFeature);
        this.sequenceEmbedding = new EmbeddingLayer(seqList, embedDim, device);
        register_module("sequenceEmbedding", sequenceEmbedding);

        // Scala: new CapsuleNetwork(embedDim, numInterests, capsuleDim, 3, device)
        this.capsuleNet = new MindCapsuleNetwork(embedDim, numInterests, capsuleDim, 3, device);
        register_module("capsuleNet", capsuleNet);

        long featSparseDim = Features.calcSparseDim(featList);
        long totalInputDim = featSparseDim + (long) numInterests * capsuleDim;
        this.tower = new MLP(totalInputDim, mlpDims, embedDim, "relu", dropout, false, device);
        register_module("tower", tower);
    }

    private Tensor normalizeSparseIndex(String name, Tensor t) {
        try {
            switch ((int) t.dim()) {
                case 0:
                    return t.unsqueeze(0L);
                case 1:
                    return t;
                case 2:
                    if (t.size(1) == 1L) return t.squeeze(1L);
                    System.err.println("[MIND WARNING] Sparse feature '" + name
                            + "' has shape " + t.sizes() + "; using first column as the ID index.");
                    return t.select(1L, 0L);
                default:
                    System.err.println("[MIND WARNING] Sparse feature '" + name
                            + "' has unexpected shape " + t.sizes() + "; flattening to 1D.");
                    return t.contiguous().view(-1);
            }
        } catch (Throwable e) {
            return t;
        }
    }

    private Tensor normalizeSequenceInput(String name, Tensor t) {
        try {
            switch ((int) t.dim()) {
                case 1:
                    System.err.println("[MIND WARNING] Sequence feature '" + name
                            + "' received 1D input " + t.sizes() + "; unsqueezing to (batch, seqLen).");
                    return t.unsqueeze(1L);
                case 2:
                    return t;
                case 3:
                    if (t.size(1) == 1L) return t.squeeze(1L);
                    if (t.size(2) == 1L) return t.squeeze(2L);
                    // fall through
                default:
                    System.err.println("[MIND WARNING] Sequence feature '" + name
                            + "' has unexpected shape " + t.sizes() + "; flattening trailing dims.");
                    long batch = t.size(0);
                    return t.contiguous().view(batch, -1L);
            }
        } catch (Throwable e) {
            return t;
        }
    }

    public Tensor forward(Map<String, Tensor> features, Tensor sequenceIndices) {
        Map<String, Tensor> normFeatures = new LinkedHashMap<>();
        for (Map.Entry<String, Tensor> e : features.entrySet()) {
            normFeatures.put(e.getKey(), normalizeSparseIndex(e.getKey(), e.getValue()));
        }
        Tensor normSequence = normalizeSequenceInput(sequenceFeature.name(), sequenceIndices);

        Tensor featEmb = featureEmbedding.forward(normFeatures);
        Tensor seqEmb = sequenceEmbedding.getSequenceEmbedding(sequenceFeature.name(), normSequence);

        Tensor interests = capsuleNet.forward(seqEmb);
        // Keep a strong Java + storage ref before reshape (avoids dangling Dropout/Linear inputs).
        interests = interests.contiguous().toType(ScalarType.Float);

        // Flatten interests [B, numInterests, capsuleDim] → [B, numInterests*capsuleDim]
        int b = (int) interests.size(0);
        int total = (int) interests.numel();
        int second = (b == 0) ? 0 : total / b;
        Tensor interestsFlat;
        try {
            interestsFlat = interests.reshape(b, second);
        } catch (Throwable e1) {
            try {
                interestsFlat = interests.contiguous().reshape(b, second);
            } catch (Throwable e2) {
                throw new RuntimeException("MIND view reshape failed: " + e2.getMessage(), e2);
            }
        }
        interestsFlat = interestsFlat.contiguous();

        Tensor featOn = featEmb.contiguous().toType(ScalarType.Float);
        TensorVector cVec = new TensorVector();
        cVec.push_back(featOn);
        cVec.push_back(interestsFlat);
        Tensor combined = torch.cat(cVec, 1L).contiguous();

        return tower.forward(combined);
    }
}
