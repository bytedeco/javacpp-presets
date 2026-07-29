/*
 * DBMTL — Deep Bayesian Multi-Task Learning style multi-task ranking (e-commerce).
 *
 * Industrial multi-task ranking often jointly predicts CTR / CVR / CTCVR with
 * shared bottom + task-specific towers and uncertainty-aware loss weighting
 * (Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses",
 * CVPR 2018) — used widely in ads / checkout funnels (Alibaba/JD variants).
 *
 * This module:
 *   shared EmbeddingLayer + MLP bottom
 *   per-task towers (ctr, cvr, optional aux)
 *   learnable log-variance parameters for homoscedastic uncertainty weighting:
 *     L = Σ_i  exp(-s_i) * L_i + s_i
 *   ESMM product head: p_ctcvr = p_ctr * p_cvr
 *
 * Pairs with DelayedFeedbackHead for post-click conversion delay.
 * Can sit on top of DIN/SIM sequence features by feeding pre-pooled embeddings
 * via the dense feature path (caller concatenates).
 */
package org.bytedeco.pytorch.utils.recommend.models.ecommerce;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
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
public class DBMTL extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public static final int COL_CTR = 0;
    public static final int COL_CVR = 1;
    public static final int COL_CTCVR = 2;
    public static final int COL_AUX = 3; // optional cart / wishlist
    public static final int NUM_OUTPUTS = 4;

    private final EmbeddingLayer embedding;
    private final MLP sharedBottom;
    private final MLP towerCtr;
    private final MLP towerCvr;
    private final MLP towerAux;
    private final DomainAdapter domainAdapter; // optional
    private final DelayedFeedbackHead delayedHead; // optional
    private final LinearImpl logVarCtr;  // scalar s for uncertainty weight
    private final LinearImpl logVarCvr;
    private final LinearImpl logVarAux;
    private final boolean useDomain;
    private final boolean useDelayed;
    private final boolean useAux;

    public DBMTL(List<? extends Feature> features) {
        this(features, new long[]{128L, 64L}, new long[]{64L, 32L},
                0, false, false, true, DeviceSupport.backend());
    }

    public DBMTL(List<? extends Feature> features, long[] sharedHidden, long[] towerHidden,
                 int numDomains, boolean useDomain, boolean useDelayed, boolean useAux,
                 String device) {
        super("DBMTL");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("DBMTL: features cannot be empty");
        }
        this.useDomain = useDomain && numDomains > 0;
        this.useDelayed = useDelayed;
        this.useAux = useAux;

        List<Feature> featList = new ArrayList<>(features);
        int featDim = 0;
        for (Feature f : featList) featDim += f.embedDim();

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        long sharedOut = sharedHidden[sharedHidden.length - 1];
        this.sharedBottom = new MLP(featDim, sharedHidden, sharedOut, "relu", 0.1f,
                false, false, true, device);
        register_module("shared_bottom", sharedBottom);

        if (this.useDomain) {
            this.domainAdapter = new DomainAdapter((int) sharedOut, numDomains, 8, true, device);
            register_module("domain_adapter", domainAdapter);
        } else {
            this.domainAdapter = null;
        }

        this.towerCtr = new MLP(sharedOut, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
        this.towerCvr = new MLP(sharedOut, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
        register_module("tower_ctr", towerCtr);
        register_module("tower_cvr", towerCvr);

        if (useAux) {
            this.towerAux = new MLP(sharedOut, towerHidden, 1L, "relu", 0.1f, false, false, true, device);
            register_module("tower_aux", towerAux);
        } else {
            this.towerAux = null;
        }

        // Learn scalar log-variance via bias of Linear(1->1) on a constant 1 input — simpler:
        // store as Linear with no input dependency: we just use bias through forward(ones).
        this.logVarCtr = new LinearImpl(1L, 1L);
        this.logVarCvr = new LinearImpl(1L, 1L);
        this.logVarAux = new LinearImpl(1L, 1L);
        register_module("log_var_ctr", logVarCtr);
        register_module("log_var_cvr", logVarCvr);
        register_module("log_var_aux", logVarAux);

        if (this.useDelayed) {
            this.delayedHead = new DelayedFeedbackHead((int) sharedOut, towerHidden, device);
            register_module("delayed_head", delayedHead);
        } else {
            this.delayedHead = null;
        }
    }

    private Tensor backbone(Map<String, Tensor> features, Tensor domainIds) {
        Tensor h = embedding.forward(features, Collections.emptyMap(), true);
        h = sharedBottom.forward(h);
        if (useDomain && domainAdapter != null && domainIds != null) {
            h = domainAdapter.forward(h, domainIds);
        }
        return h;
    }

    /**
     * @return [B, 4] = [p_ctr, p_cvr, p_ctcvr, p_aux]  (p_aux=0 if disabled)
     */
    public Tensor forward(Map<String, Tensor> features) {
        return forward(features, null);
    }

    public Tensor forward(Map<String, Tensor> features, Tensor domainIds) {
        Tensor h = backbone(features, domainIds);
        Tensor pCtr = towerCtr.forward(h).squeeze(1L).sigmoid();
        Tensor pCvr = towerCvr.forward(h).squeeze(1L).sigmoid();
        Tensor pCtcvr = pCtr.mul(pCvr);
        Tensor pAux;
        if (useAux && towerAux != null) {
            pAux = towerAux.forward(h).squeeze(1L).sigmoid();
        } else {
            pAux = torch.zeros_like(pCtr);
        }
        TensorVector out = new TensorVector();
        out.push_back(pCtr.unsqueeze(1L));
        out.push_back(pCvr.unsqueeze(1L));
        out.push_back(pCtcvr.unsqueeze(1L));
        out.push_back(pAux.unsqueeze(1L));
        return torch.cat(out, 1L);
    }

    /**
     * Uncertainty-weighted multi-task loss + ESMM CTCVR term.
     *
     * @param preds        forward() output [B,4]
     * @param click        [B]
     * @param conversion   [B]
     * @param auxLabel     [B] or null
     * @param h            backbone features for delayed NLL (optional)
     * @param elapsedHours optional delayed feedback times
     */
    public Tensor computeLoss(Tensor preds, Tensor click, Tensor conversion, Tensor auxLabel,
                              Tensor h, Tensor elapsedHours) {
        Tensor pCtr = preds.select(1L, COL_CTR);
        Tensor pCtcvr = preds.select(1L, COL_CTCVR);
        Tensor pAux = preds.select(1L, COL_AUX);

        Tensor yClick = click.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor yConv = conversion.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);

        Tensor lCtr = bce(pCtr, yClick);
        Tensor lCtcvr = bce(pCtcvr, yClick.mul(yConv));

        // log-sigma via Linear on ones → scalar per batch mean
        Tensor ones = torch.ones(new long[]{1L, 1L},
                new org.bytedeco.pytorch.TensorOptions()
                        .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(
                                org.bytedeco.pytorch.global.torch.ScalarType.Float)));
        Tensor sCtr = logVarCtr.forward(ones).squeeze();
        Tensor sCvr = logVarCvr.forward(ones).squeeze();

        // L = exp(-s)*loss + s
        Tensor loss = lCtr.mul(torch.exp(sCtr.neg())).add(sCtr)
                .add(lCtcvr.mul(torch.exp(sCvr.neg())).add(sCvr));

        if (useAux && auxLabel != null && towerAux != null) {
            Tensor yAux = auxLabel.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
            Tensor lAux = bce(pAux, yAux);
            Tensor sAux = logVarAux.forward(ones).squeeze();
            loss = loss.add(lAux.mul(torch.exp(sAux.neg())).add(sAux));
        }

        if (useDelayed && delayedHead != null && h != null && elapsedHours != null) {
            loss = loss.add(delayedHead.delayedFeedbackNll(h, yConv, elapsedHours));
        }
        return loss;
    }

    public Tensor backboneFeatures(Map<String, Tensor> features, Tensor domainIds) {
        return backbone(features, domainIds);
    }

    private static Tensor bce(Tensor p, Tensor y) {
        Tensor pp = p.clamp(
                new org.bytedeco.pytorch.ScalarOptional(new Scalar(1e-6f)),
                new org.bytedeco.pytorch.ScalarOptional(new Scalar(1.0f - 1e-6f)));
        return y.neg().mul(pp.log())
                .add(torch.sub(torch.ones_like(y), y).neg()
                        .mul(torch.sub(torch.ones_like(pp), pp).log()))
                .mean();
    }
}
