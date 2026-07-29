/*
 * Ported from torch-rechub-scala: torchrec/models/matching/ComirecSA.scala
 * (InterestExtractor + ComirecSA)
 *
 * Comirec-SA: Self-Attentive Multi-Interest Framework. Reference: RecSys 2020
 */
package org.bytedeco.pytorch.utils.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
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

/** Interest Extractor using Self-Attention. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class InterestExtractor extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numInterests;
    private final LinearImpl queryProj;
    private final LinearImpl keyProj;
    private final LinearImpl valueProj;
    private final LinearImpl interestQuery;
    private final DropoutImpl dropoutLayer;

    public InterestExtractor(int embedDim, int numInterests, int numHeads, float dropout) {
        this(embedDim, numInterests, numHeads, dropout, DeviceSupport.backend());
    }

    public InterestExtractor(int embedDim, int numInterests, int numHeads, float dropout, String device) {
        super("InterestExtractor");
        this.embedDim = embedDim;
        this.numInterests = numInterests;
        // numHeads kept for API parity

        this.queryProj = new LinearImpl(embedDim, embedDim);
        this.keyProj = new LinearImpl(embedDim, embedDim);
        this.valueProj = new LinearImpl(embedDim, embedDim);
        this.interestQuery = new LinearImpl(embedDim, (long) numInterests * embedDim);

        Device dev = new Device(device);
        queryProj.to(dev, false);
        keyProj.to(dev, false);
        valueProj.to(dev, false);
        interestQuery.to(dev, false);

        register_module("queryProj", queryProj);
        register_module("keyProj", keyProj);
        register_module("valueProj", valueProj);
        register_module("interestQuery", interestQuery);

        this.dropoutLayer = new DropoutImpl(dropout);
    }

    @Override
    public Tensor forward(Tensor seqEmb) {
        // seqEmb: (batch, seq_len, embed_dim)
        long batchSize = seqEmb.size(0);

        Tensor q = queryProj.forward(seqEmb);
        Tensor k = keyProj.forward(seqEmb);
        Tensor v = valueProj.forward(seqEmb);

        float scale = (float) Math.sqrt(embedDim);
        Scalar invScale = new Scalar(1.0f / scale);
        Tensor scores = torch.matmul(q, k.transpose(1, 2)).mul(invScale);
        Tensor attn = dropoutLayer.forward(scores.softmax(-1));
        Tensor attended = torch.matmul(attn, v);

        Tensor interestQ = interestQuery.forward(attended.mean(1));
        Tensor interestQReshaped = interestQ.view(batchSize, numInterests, embedDim);

        Tensor interestEmb = torch.matmul(interestQReshaped, attended.transpose(1, 2)).softmax(-1);
        return torch.matmul(interestEmb, attended); // (batch, num_interests, embed_dim)
    }
}
