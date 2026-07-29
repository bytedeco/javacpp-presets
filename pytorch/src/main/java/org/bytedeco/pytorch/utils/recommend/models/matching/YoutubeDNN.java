/*
 * Ported from torch-rechub-scala: torchrec/models/matching/YoutubeDNN.scala
 *
 * YouTubeDNN Matching Model. Reference: YouTube
 */
package org.bytedeco.pytorch.utils.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.features.Features;
import org.bytedeco.pytorch.utils.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class YoutubeDNN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long totalInputDim;
    private final EmbeddingLayer featureEmbedding;
    private final EmbeddingLayer sequenceEmbedding;
    private final MLP tower;

    public YoutubeDNN(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures) {
        this(features, sequenceFeatures, 8, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public YoutubeDNN(List<? extends Feature> features, List<SequenceFeature> sequenceFeatures,
                      int embedDim, long[] towerDims, float dropout, String device) {
        super("YoutubeDNN");
        List<Feature> featList = new ArrayList<>(features);
        List<SequenceFeature> seqList = sequenceFeatures != null
                ? new ArrayList<>(sequenceFeatures) : new ArrayList<>();

        this.featureEmbedding = new EmbeddingLayer(featList, embedDim, device);
        register_module("featureEmbedding", featureEmbedding);

        this.sequenceEmbedding = new EmbeddingLayer(new ArrayList<>(seqList), embedDim, device);
        register_module("sequenceEmbedding", sequenceEmbedding);

        long featSparseDim = Features.calcSparseDim(featList);
        long seqSparseDim = 0L;
        for (SequenceFeature sf : seqList) {
            seqSparseDim += sf.embedDim();
        }
        this.totalInputDim = featSparseDim + seqSparseDim;

        this.tower = new MLP(totalInputDim, towerDims, embedDim, "relu", dropout, false, device);
        register_module("tower", tower);
    }

    public Tensor forward(Map<String, Tensor> features, Map<String, Tensor> sequenceFeatures) {
        Tensor featEmb = featureEmbedding.forward(features);

        List<Tensor> seqEmbs = new ArrayList<>();
        if (sequenceFeatures != null) {
            for (Map.Entry<String, Tensor> e : sequenceFeatures.entrySet()) {
                seqEmbs.add(sequenceEmbedding.getSequenceEmbedding(e.getKey(), e.getValue()));
            }
        }

        Tensor seqPooled;
        if (!seqEmbs.isEmpty()) {
            TensorVector vec = new TensorVector();
            for (Tensor t : seqEmbs) vec.push_back(t);
            seqPooled = torch.cat(vec, 1).mean(1);
        } else {
            seqPooled = TensorHelpers.zeros(totalInputDim);
        }

        TensorVector cVec = new TensorVector();
        cVec.push_back(featEmb);
        cVec.push_back(seqPooled);
        Tensor combined = torch.cat(cVec, 1L);
        return tower.forward(combined);
    }

    public Tensor userTowerForward(Map<String, Tensor> features, Map<String, Tensor> sequenceFeatures) {
        return forward(features, sequenceFeatures);
    }

    public Tensor userTowerForward(Map<String, Tensor> features) {
        return forward(features, Collections.emptyMap());
    }
}
