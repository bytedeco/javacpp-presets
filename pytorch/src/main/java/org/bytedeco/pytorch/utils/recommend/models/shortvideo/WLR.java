/*
 * WLR — Weighted Logistic Regression for watch-time ranking (YouTube / short-video).
 *
 * Production / paper references:
 *   - Covington et al., "Deep Neural Networks for YouTube Recommendations", RecSys 2016
 *     (watch time as implicit feedback; weighted logistic regression surrogate)
 *   - Industrial short-video ranking at ByteDance / Kuaishou / YouTube routinely treats
 *     watch-time (or completion ratio) as a continuous label and trains with
 *     watch-time-weighted binary cross-entropy:
 *       loss = w_i * BCE(p_i, y_i)  where w_i ∝ watch_time (or soft labels)
 *
 * This module is a ranking tower that:
 *   1. Embeds sparse + dense features (EmbeddingLayer + optional dense MLP)
 *   2. Predicts click/consume probability p
 *   3. Exposes weighted BCE helper for watch-time labels
 *
 * Pair with DurationDeconfoundHead (D2Q) when duration bias must be removed.
 */
package org.bytedeco.pytorch.utils.recommend.models.shortvideo;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.DurationDeconfoundHead;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class WLR extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embedding;
    private final MLP tower;
    private final DurationDeconfoundHead d2qHead; // optional; null if disabled
    private final int embedOutDim;
    private final boolean useD2Q;

    public WLR(List<? extends Feature> features) {
        this(features, new long[]{256L, 128L, 64L}, 0, false, DeviceSupport.backend());
    }

    /**
     * @param features            sparse/dense feature specs
     * @param hiddenDims          MLP hidden sizes
     * @param numDurationBuckets  >0 enables D2Q head
     * @param useD2Q              enable duration-deconfounded interest head
     */
    public WLR(List<? extends Feature> features, long[] hiddenDims, int numDurationBuckets,
               boolean useD2Q, String device) {
        super("WLR");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("WLR: features cannot be empty");
        }
        List<Feature> featList = new ArrayList<>(features);
        int sumDim = 0;
        for (Feature f : featList) sumDim += f.embedDim();
        this.embedOutDim = sumDim;
        this.useD2Q = useD2Q && numDurationBuckets > 1;

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        this.tower = new MLP(sumDim, hiddenDims, 1L, "relu", 0.1f, false, false, true, device);
        register_module("tower", tower);

        if (this.useD2Q) {
            // D2Q operates on penultimate representation: reuse last hidden as inputDim
            long lastHidden = hiddenDims != null && hiddenDims.length > 0
                    ? hiddenDims[hiddenDims.length - 1] : sumDim;
            // Simpler: run D2Q on raw embedding concat
            this.d2qHead = new DurationDeconfoundHead(sumDim, numDurationBuckets, 16,
                    new long[]{128L, 64L}, device);
            register_module("d2q_head", d2qHead);
        } else {
            this.d2qHead = null;
        }
    }

    /** CTR / consume probability [B]. */
    public Tensor forward(Map<String, Tensor> features) {
        Tensor emb = embedding.forward(features, Collections.emptyMap(), true);
        return tower.forward(emb).squeeze(1L).sigmoid();
    }

    /**
     * When D2Q enabled: returns [B, 3] = [p_consume, watch_time, interest].
     * Otherwise returns [B, 1] = [p_consume].
     */
    public Tensor forwardWithDuration(Map<String, Tensor> features, Tensor durationBucket) {
        Tensor emb = embedding.forward(features, Collections.emptyMap(), true);
        Tensor p = tower.forward(emb).squeeze(1L).sigmoid();
        if (!useD2Q || d2qHead == null) {
            return p.unsqueeze(1L);
        }
        Tensor d2q = d2qHead.forward(emb, durationBucket); // [B, 2]
        TensorVector out = new TensorVector();
        out.push_back(p.unsqueeze(1L));
        out.push_back(d2q);
        return torch.cat(out, 1L); // [B, 3]
    }

    /**
     * YouTube-style weighted logistic loss.
     *
     * @param logitsOrProb model output probability [B] (will be clamped)
     * @param label        binary label or soft label in [0,1] [B]
     * @param weight       watch-time weight [B] (e.g. seconds watched); null = uniform
     * @return scalar mean weighted BCE
     */
    public static Tensor weightedBceLoss(Tensor logitsOrProb, Tensor label, Tensor weight) {
        Tensor p = logitsOrProb.clamp(new ScalarOptional(new Scalar(1e-6f)), new ScalarOptional(new Scalar(1.0f - 1e-6f)));
        Tensor y = label.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor bce = y.neg().mul(p.log()).add(
                torch.sub(torch.ones_like(y), y).neg().mul(torch.sub(torch.ones_like(p), p).log()));
        if (weight != null && !weight.isNull() && weight.numel() > 0) {
            Tensor w = weight.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
            // normalize weights to mean 1 for stable LR
            Tensor wNorm = w.div(w.mean().clamp_min(new Scalar(1e-6f)));
            return bce.mul(wNorm).mean();
        }
        return bce.mean();
    }

    public boolean useD2Q() {
        return useD2Q;
    }

    public int embedOutDim() {
        return embedOutDim;
    }
}
