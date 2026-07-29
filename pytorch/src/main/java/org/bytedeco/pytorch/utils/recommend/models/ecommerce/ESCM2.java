/*
 * ESCM2 — Entire Space Counterfactual Multi-task Model for CVR.
 *
 * Reference:
 *   Ma et al. / Wang et al., "ESCM2: Entire Space Counterfactual Multi-Task Model
 *   for Post-Click Conversion Rate Estimation", SIGIR 2022 (Alibaba).
 *   Builds on ESMM (SIGIR'18) with counterfactual risk mitigation for CVR.
 *
 * Industrial context (Taobao / Tmall / JD checkout funnel):
 *   impression -> click -> conversion. CVR trained only on clicked samples is
 *   selection-biased; ESMM uses p(CTCVR)=p(CTR)*p(CVR) over entire space.
 *   ESCM2 further introduces counterfactual regularizers / IPS-style correction.
 *
 * This implementation:
 *   - Shared embedding + two towers (CTR, CVR) like ESMM
 *   - Auxiliary IMPRESSION-space CVR tower (counterfactual)
 *   - Outputs [p_cvr, p_ctr, p_ctcvr, p_cvr_cf]
 *   - Helper loss combining CTR BCE + CTCVR BCE + CF regularizer
 *
 * Reuses existing ESMM ideas already in multi_task.ESMM; this is the industrial
 * e-commerce specialization with counterfactual head.
 */
package org.bytedeco.pytorch.utils.recommend.models.ecommerce;

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
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.DelayedFeedbackHead;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.DomainAdapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ESCM2 extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public static final int COL_CVR = 0;
    public static final int COL_CTR = 1;
    public static final int COL_CTCVR = 2;
    public static final int COL_CVR_CF = 3; // counterfactual / impression-space CVR
    public static final int NUM_OUTPUTS = 4;

    private final EmbeddingLayer embedding;
    private final MLP towerCtr;
    private final MLP towerCvr;
    private final MLP towerCvrCf; // counterfactual CVR over entire space
    private final DomainAdapter domainAdapter; // optional multi-domain
    private final DelayedFeedbackHead delayedHead; // optional delayed conversion
    private final boolean useDomain;
    private final boolean useDelayedFeedback;
    private final int featDim;

    public ESCM2(List<? extends Feature> features) {
        this(features, new long[]{128L, 64L}, 0, false, false, DeviceSupport.backend());
    }

    public ESCM2(List<? extends Feature> features, long[] towerDims, int numDomains,
                 boolean useDomain, boolean useDelayedFeedback, String device) {
        super("ESCM2");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("ESCM2: features cannot be empty");
        }
        this.useDomain = useDomain && numDomains > 0;
        this.useDelayedFeedback = useDelayedFeedback;

        List<Feature> featList = new ArrayList<>(features);
        int dim = 0;
        for (Feature f : featList) dim += f.embedDim();
        this.featDim = dim;

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        if (this.useDomain) {
            this.domainAdapter = new DomainAdapter(dim, numDomains, 16, true, device);
            register_module("domain_adapter", domainAdapter);
        } else {
            this.domainAdapter = null;
        }

        this.towerCtr = new MLP(dim, towerDims, 1L, "relu", 0.1f, false, false, true, device);
        this.towerCvr = new MLP(dim, towerDims, 1L, "relu", 0.1f, false, false, true, device);
        this.towerCvrCf = new MLP(dim, towerDims, 1L, "relu", 0.1f, false, false, true, device);
        register_module("tower_ctr", towerCtr);
        register_module("tower_cvr", towerCvr);
        register_module("tower_cvr_cf", towerCvrCf);

        if (this.useDelayedFeedback) {
            this.delayedHead = new DelayedFeedbackHead(dim, towerDims, device);
            register_module("delayed_feedback_head", delayedHead);
        } else {
            this.delayedHead = null;
        }
    }

    private Tensor backbone(Map<String, Tensor> features, Tensor domainIds) {
        Tensor h = embedding.forward(features, Collections.emptyMap(), true);
        if (useDomain && domainAdapter != null && domainIds != null) {
            h = domainAdapter.forward(h, domainIds);
        }
        return h;
    }

    /**
     * @return [B, 4] = [p_cvr, p_ctr, p_ctcvr, p_cvr_cf]
     */
    public Tensor forward(Map<String, Tensor> features) {
        return forward(features, null);
    }

    public Tensor forward(Map<String, Tensor> features, Tensor domainIds) {
        Tensor h = backbone(features, domainIds);
        Tensor pCtr = towerCtr.forward(h).squeeze(1L).sigmoid();
        Tensor pCvr = towerCvr.forward(h).squeeze(1L).sigmoid();
        Tensor pCvrCf = towerCvrCf.forward(h).squeeze(1L).sigmoid();
        Tensor pCtcvr = pCtr.mul(pCvr);

        TensorVector out = new TensorVector();
        out.push_back(pCvr.unsqueeze(1L));
        out.push_back(pCtr.unsqueeze(1L));
        out.push_back(pCtcvr.unsqueeze(1L));
        out.push_back(pCvrCf.unsqueeze(1L));
        return torch.cat(out, 1L);
    }

    /**
     * Combined industrial loss:
     *   L = BCE(ctr) + BCE(ctcvr) + λ * BCE(cvr_cf on clicked only, stop-grad weight)
     *     + optional delayed-feedback NLL
     *
     * @param preds     forward() output [B, 4]
     * @param click     [B] 0/1
     * @param conversion [B] 0/1 (only meaningful when click=1; CTCVR uses raw)
     * @param h         optional backbone features for delayed head; null to skip
     * @param elapsedHours optional; required if delayed head used
     * @param lambdaCf  weight for counterfactual regularizer
     */
    public Tensor computeLoss(Tensor preds, Tensor click, Tensor conversion,
                              Tensor h, Tensor elapsedHours, float lambdaCf) {
        Tensor pCvr = preds.select(1L, COL_CVR);
        Tensor pCtr = preds.select(1L, COL_CTR);
        Tensor pCtcvr = preds.select(1L, COL_CTCVR);
        Tensor pCvrCf = preds.select(1L, COL_CVR_CF);

        Tensor yClick = click.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor yConv = conversion.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);

        Tensor lossCtr = bce(pCtr, yClick);
        Tensor lossCtcvr = bce(pCtcvr, yClick.mul(yConv)); // conversion in entire space

        // CF regularizer: encourage p_cvr_cf ≈ p_cvr on clicked samples
        Tensor clickedMask = yClick;
        Tensor cfDiff = pCvrCf.sub(pCvr).pow(new Scalar(2.0f)).mul(clickedMask);
        Tensor lossCf = cfDiff.sum().div(clickedMask.sum().clamp_min(new Scalar(1.0f)));

        Tensor loss = lossCtr.add(lossCtcvr).add(lossCf.mul(new Scalar(lambdaCf)));

        if (useDelayedFeedback && delayedHead != null && h != null && elapsedHours != null) {
            loss = loss.add(delayedHead.delayedFeedbackNll(h, yConv, elapsedHours));
        }
        return loss;
    }

    private static Tensor bce(Tensor p, Tensor y) {
        Tensor pp = p.clamp(new ScalarOptional(new Scalar(1e-6f)), new ScalarOptional(new Scalar(1.0f - 1e-6f)));
        return y.neg().mul(pp.log())
                .add(torch.sub(torch.ones_like(y), y).neg()
                        .mul(torch.sub(torch.ones_like(pp), pp).log()))
                .mean();
    }

    public Tensor backboneFeatures(Map<String, Tensor> features, Tensor domainIds) {
        return backbone(features, domainIds);
    }

    public boolean useDomain() {
        return useDomain;
    }

    public boolean useDelayedFeedback() {
        return useDelayedFeedback;
    }
}
