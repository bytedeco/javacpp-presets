/*
 * Ported from torch-rechub-scala: torchrec/models/matching/ComirecDR.scala
 *
 * Comirec-DR: Dynamic Routing Multi-Interest Framework. Reference: RecSys 2020
 */
package org.bytedeco.pytorch.utils.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
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
public class ComirecDR extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final SequenceFeature sequenceFeature;
    private final int numInterests;
    private final EmbeddingLayer featureEmbedding;
    private final EmbeddingLayer sequenceEmbedding;
    private final DynamicRouter dynamicRouter;
    private final MLP tower;

    public ComirecDR(List<? extends Feature> features, SequenceFeature sequenceFeature) {
        this(features, sequenceFeature, 8, 4, new long[]{256L, 128L}, 0.2f, DeviceSupport.backend());
    }

    public ComirecDR(List<? extends Feature> features, SequenceFeature sequenceFeature,
                     int embedDim, int numInterests, long[] mlpDims,
                     float dropout, String device) {
        super("ComirecDR");
        this.sequenceFeature = sequenceFeature;
        this.numInterests = numInterests;

        List<Feature> featList = new ArrayList<>(features);
        this.featureEmbedding = new EmbeddingLayer(featList, embedDim, device);
        register_module("featureEmbedding", featureEmbedding);

        List<Feature> seqList = new ArrayList<>();
        seqList.add(sequenceFeature);
        this.sequenceEmbedding = new EmbeddingLayer(seqList, embedDim, device);
        register_module("sequenceEmbedding", sequenceEmbedding);

        this.dynamicRouter = new DynamicRouter(embedDim, numInterests, 3);
        register_module("dynamicRouter", dynamicRouter);

        long featSparseDim = Features.calcSparseDim(featList);
        long totalInputDim = numInterests * (featSparseDim + embedDim);
        this.tower = new MLP(totalInputDim, mlpDims, embedDim, "relu", dropout, false, device);
        register_module("tower", tower);
    }

    public Tensor forward(Map<String, Tensor> features, Tensor sequenceIndices) {
        Tensor featEmb = featureEmbedding.forward(features);
        Tensor seqEmb = sequenceEmbedding.getSequenceEmbedding(sequenceFeature.name(), sequenceIndices);

        Tensor interests = dynamicRouter.forward(seqEmb);

        Tensor featExpanded = featEmb.unsqueeze(1).repeat(1, numInterests, 1);
        var dev = featExpanded.device();
        Tensor interestsOnDev = interests.device().equals(dev)
                ? interests : interests.to(dev, interests.dtype());

        TensorVector vec = new TensorVector();
        vec.push_back(featExpanded);
        vec.push_back(interestsOnDev);
        Tensor featWithInterests = torch.cat(vec, 2L);
        long batchSize = featEmb.size(0);
        Tensor flattened = featWithInterests.view(batchSize, -1);

        return tower.forward(flattened);
    }
}
