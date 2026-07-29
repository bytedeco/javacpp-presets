/*
 * LiveMultiTask — multi-task ranking for live-streaming (gift / CTR / stay / follow).
 *
 * Production context (Douyin / Kuaishou / Twitch-style live rec):
 *   Simultaneous prediction of:
 *     - CTR (enter / click room)
 *     - stay / watch duration (often quantized)
 *     - gift / pay propensity (CVR-like, sparse positive)
 *     - follow / share
 *   Entire-space multi-task (ESMM-style) is commonly applied so that gift CVR
 *   is trained over the full impression space: p(gift) = p(ctr) * p(gift|click).
 *
 * Architecture:
 *   shared EmbeddingLayer + shared MLP bottom
 *   per-task towers
 *   ESMM product heads for sparse pay actions
 *
 * References mixed from industrial practice:
 *   - ESMM (Alibaba SIGIR'18) for entire-space CVR
 *   - MMoE / PLE for multi-task live ranking (already in multi_task package)
 *   - This model is a live-domain specialization with gift/stay heads
 */
package org.bytedeco.pytorch.utils.recommend.models.live;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LiveMultiTask extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Output column order. */
    public static final int COL_CTR = 0;
    public static final int COL_STAY = 1;
    public static final int COL_GIFT_CVR = 2;   // p(gift | click)
    public static final int COL_GIFT_CTCVR = 3; // p(ctr)*p(gift|click)
    public static final int COL_FOLLOW = 4;
    public static final int NUM_OUTPUTS = 5;

    private final EmbeddingLayer embedding;
    private final MLP sharedBottom;
    private final MLP towerCtr;
    private final MLP towerStay;
    private final MLP towerGiftCvr;
    private final MLP towerFollow;

    public LiveMultiTask(List<? extends Feature> features) {
        this(features, new long[]{256L, 128L}, new long[]{64L, 32L}, DeviceSupport.backend());
    }

    public LiveMultiTask(List<? extends Feature> features, long[] sharedHidden,
                         long[] towerHidden, String device) {
        super("LiveMultiTask");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("LiveMultiTask: features cannot be empty");
        }
        List<Feature> featList = new ArrayList<>(features);
        int featDim = 0;
        for (Feature f : featList) featDim += f.embedDim();

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        long sharedOut = sharedHidden[sharedHidden.length - 1];
        this.sharedBottom = new MLP(featDim, sharedHidden, sharedOut, "relu", 0.1f,
                false, false, true, device);
        register_module("shared_bottom", sharedBottom);

        this.towerCtr = new MLP(sharedOut, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
        this.towerStay = new MLP(sharedOut, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
        this.towerGiftCvr = new MLP(sharedOut, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
        this.towerFollow = new MLP(sharedOut, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
        register_module("tower_ctr", towerCtr);
        register_module("tower_stay", towerStay);
        register_module("tower_gift_cvr", towerGiftCvr);
        register_module("tower_follow", towerFollow);
    }

    /**
     * @return [B, 5] = [p_ctr, stay_score, p_gift_cvr, p_gift_ctcvr, p_follow]
     *         stay_score is softplus (unbounded positive duration proxy)
     */
    public Tensor forward(Map<String, Tensor> features) {
        Tensor emb = embedding.forward(features, Collections.emptyMap(), true);
        Tensor h = sharedBottom.forward(emb);

        Tensor pCtr = towerCtr.forward(h).squeeze(1L).sigmoid();
        Tensor stay = torch.softplus(towerStay.forward(h).squeeze(1L));
        Tensor pGiftCvr = towerGiftCvr.forward(h).squeeze(1L).sigmoid();
        Tensor pGiftCtcvr = pCtr.mul(pGiftCvr); // ESMM entire-space
        Tensor pFollow = towerFollow.forward(h).squeeze(1L).sigmoid();

        TensorVector out = new TensorVector();
        out.push_back(pCtr.unsqueeze(1L));
        out.push_back(stay.unsqueeze(1L));
        out.push_back(pGiftCvr.unsqueeze(1L));
        out.push_back(pGiftCtcvr.unsqueeze(1L));
        out.push_back(pFollow.unsqueeze(1L));
        return torch.cat(out, 1L);
    }
}
