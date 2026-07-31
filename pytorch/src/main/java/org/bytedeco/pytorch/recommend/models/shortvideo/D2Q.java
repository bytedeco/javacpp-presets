/*
 * D2Q — Duration Deconfounded watch-time ranking model (short-video).
 *
 * Reference:
 *   Zhan et al., "Deconfounding Duration Bias in Watch-time Prediction for
 *   Video Recommendation", KDD 2022 (Kuaishou).
 *
 * Wraps shared DurationDeconfoundHead on top of a standard feature tower so
 * callers can train:
 *   L = L_watchtime(duration-conditioned) + L_interest(deconfounded)
 * without hand-wiring the head.
 *
 * See also WLR which optionally embeds the same head.
 */
package org.bytedeco.pytorch.recommend.models.shortvideo;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.industry.DurationDeconfoundHead;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class D2Q extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embedding;
    private final MLP backbone;
    private final DurationDeconfoundHead head;
    private final int repDim;

    public D2Q(List<? extends Feature> features, int numDurationBuckets) {
        this(features, numDurationBuckets, new long[]{256L, 128L}, DeviceSupport.backend());
    }

    public D2Q(List<? extends Feature> features, int numDurationBuckets,
               long[] hiddenDims, String device) {
        super("D2Q");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("D2Q: features cannot be empty");
        }
        if (numDurationBuckets < 2) {
            throw new IllegalArgumentException("numDurationBuckets must be >= 2");
        }
        List<Feature> featList = new ArrayList<>(features);
        int featDim = 0;
        for (Feature f : featList) featDim += f.embedDim();

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        this.repDim = (int) hiddenDims[hiddenDims.length - 1];
        this.backbone = new MLP(featDim, hiddenDims, repDim, "relu", 0.1f,
                false, false, true, device);
        register_module("backbone", backbone);

        this.head = new DurationDeconfoundHead(repDim, numDurationBuckets, 16,
                new long[]{64L, 32L}, device);
        register_module("d2q_head", head);
    }

    /**
     * @param features       feature map
     * @param durationBucket [B] long bucket id
     * @return [B, 2] = [watch_time_pred, interest_score]
     */
    public Tensor forward(Map<String, Tensor> features, Tensor durationBucket) {
        Tensor emb = embedding.forward(features, Collections.emptyMap(), true);
        Tensor rep = backbone.forward(emb);
        return head.forward(rep, durationBucket);
    }

    public Tensor interestScore(Map<String, Tensor> features, Tensor durationBucket) {
        return forward(features, durationBucket).select(1L, 1L);
    }

    public Tensor watchTimePred(Map<String, Tensor> features, Tensor durationBucket) {
        return forward(features, durationBucket).select(1L, 0L);
    }
}
