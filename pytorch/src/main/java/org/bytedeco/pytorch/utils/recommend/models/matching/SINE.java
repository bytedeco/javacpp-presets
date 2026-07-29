/*
 * Ported from torch-rechub-scala: torchrec/models/matching/SINE.scala
 *
 * SINE - Self-supervised Interest Network.
 * Reference: Zhang et al., 2021
 */
package org.bytedeco.pytorch.utils.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SINE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final SequenceFeature sequenceFeature;
    private final EmbeddingLayer embedding;
    private final EmbeddingLayer seqEmbedding;
    private final LinearImpl featProj;
    private final MLP interestExtractor;
    private final MLP tower;

    public SINE(List<? extends Feature> features, SequenceFeature sequenceFeature) {
        this(features, sequenceFeature, 8, 4, new long[]{128L, 64L}, 0.2f, DeviceSupport.backend());
    }

    public SINE(List<? extends Feature> features, SequenceFeature sequenceFeature,
                int embedDim, int numInterests, long[] mlpDims, float dropout, String device) {
        super("SINE");
        this.sequenceFeature = sequenceFeature;

        List<Feature> featList = new ArrayList<>(features);
        this.embedding = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embedding);

        List<Feature> seqList = new ArrayList<>();
        seqList.add(sequenceFeature);
        this.seqEmbedding = new EmbeddingLayer(seqList, embedDim, device);
        register_module("seqEmbedding", seqEmbedding);

        // Sparse concat dim may be > embedDim; project to shared interest space.
        long sparseDim = Features.calcSparseDim(featList);
        if (sparseDim <= 0L) sparseDim = embedDim;
        this.featProj = new LinearImpl(sparseDim, embedDim);
        if (device != null && !"cpu".equals(device)) {
            featProj.to(new Device(device), false);
        }
        register_module("featProj", featProj);

        this.interestExtractor = new MLP(embedDim, new long[]{(long) numInterests * embedDim},
                embedDim, "relu", dropout, false, device);
        register_module("interestExtractor", interestExtractor);

        this.tower = new MLP(embedDim, mlpDims, embedDim, "relu", dropout, false, device);
        register_module("tower", tower);
    }

    public Tensor forward(Map<String, Tensor> features, Tensor sequence) {
        Tensor featEmb = featProj.forward(embedding.forward(features));
        Tensor seqEmb = seqEmbedding.getSequenceEmbedding(sequenceFeature.name(), sequence);
        // seqEmb [B,S,D] → mean pool over time
        Tensor seqPooled = seqEmb.dim() >= 3L ? seqEmb.mean(1) : seqEmb;
        Tensor interests = interestExtractor.forward(seqPooled);
        Tensor combined = featEmb.add(interests);
        return tower.forward(combined);
    }
}
