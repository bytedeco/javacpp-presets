/*
 * Ported from torch-rechub-scala: torchrec/models/matching/NARM.scala
 *
 * NARM - Neural Attentive Sequential Recommender.
 * Reference: Li et al., 2017
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NARM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String seqFeatureName;
    private final EmbeddingLayer embedding;
    private final MLP encoder;
    private final MLP attention;
    private final LinearImpl output;

    public NARM(List<? extends Feature> features) {
        this(features, 8, 8, 8, DeviceSupport.backend());
    }

    public NARM(List<? extends Feature> features, int embedDim, int hiddenDim,
                int attentionDim, String device) {
        super("NARM");
        List<Feature> featList = new ArrayList<>(features);
        this.embedding = new EmbeddingLayer(featList, embedDim, device);
        register_module("embedding", embedding);

        String name = "seq_feat";
        for (Feature f : featList) {
            if (f instanceof SequenceFeature) {
                name = f.name();
                break;
            }
        }
        this.seqFeatureName = name;

        this.encoder = new MLP(embedDim, new long[]{hiddenDim}, hiddenDim, "relu", 0f, false, device);
        register_module("encoder", encoder);

        this.attention = new MLP(embedDim, new long[]{attentionDim}, hiddenDim, "relu", 0f, false, device);
        register_module("attention", attention);

        this.output = new LinearImpl(hiddenDim * 2L, embedDim);
        output.to(new Device(device), false);
        register_module("output", output);
    }

    @Override
    public Tensor forward(Tensor sequence) {
        Map<String, Tensor> seqMap = Collections.singletonMap(seqFeatureName, sequence);
        Tensor raw = embedding.forwardSeqRaw(seqMap);
        Tensor emb = raw.dim() == 4L ? raw.squeeze(1L) : raw;
        Tensor pooled = emb.mean(1);
        Tensor encodedPooled = encoder.forward(pooled);
        Tensor encoded = encodedPooled.unsqueeze(1).repeat(1, emb.size(1), 1);
        Tensor last = encoded.select(1, encoded.size(1) - 1);
        Tensor attn = attention.forward(pooled);
        TensorVector vec = new TensorVector();
        vec.push_back(last);
        vec.push_back(attn);
        Tensor combined = torch.cat(vec, 1L);
        return output.forward(combined);
    }
}
