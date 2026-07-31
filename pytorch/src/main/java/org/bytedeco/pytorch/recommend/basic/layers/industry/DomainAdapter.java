/*
 * Domain / scenario adapter used in multi-domain e-commerce and short-video ranking.
 *
 * Production references:
 *   - STAR (Alibaba): Partitioned Normalization + specific FN for multi-domain CTR
 *     Sheng et al., "One Model to Serve All: Star Topology Adaptive Recommender
 *     for Multi-Domain CTR Prediction", CIKM 2021
 *   - PEPNet EPNet (Kuaishou CIKM 2022): embedding personalization gate per scenario
 *   - M2M / HiNet multi-domain towers in industrial ads
 *
 * This module implements a lightweight STAR-style residual domain FN:
 *   h' = h ⊙ (1 + softplus(W_d h + b_d))   where W_d is domain-specific
 * plus an optional domain embedding added to the representation.
 */
package org.bytedeco.pytorch.recommend.basic.layers.industry;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DomainAdapter extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int numDomains;
    private final int featureDim;
    private final EmbeddingImpl domainEmbedding;
    private final List<LinearImpl> domainFns = new ArrayList<>();
    private final LinearImpl sharedFn;
    private final boolean useDomainEmbedding;

    public DomainAdapter(int featureDim, int numDomains) {
        this(featureDim, numDomains, 16, true, DeviceSupport.backend());
    }

    public DomainAdapter(int featureDim, int numDomains, int domainEmbedDim,
                         boolean useDomainEmbedding, String device) {
        super("DomainAdapter");
        if (numDomains < 1) {
            throw new IllegalArgumentException("numDomains must be >= 1");
        }
        this.numDomains = numDomains;
        this.featureDim = featureDim;
        this.useDomainEmbedding = useDomainEmbedding;

        this.sharedFn = new LinearImpl(featureDim, featureDim);
        register_module("shared_fn", sharedFn);

        for (int d = 0; d < numDomains; d++) {
            LinearImpl fn = new LinearImpl(featureDim, featureDim);
            register_module("domain_fn_" + d, fn);
            domainFns.add(fn);
        }

        if (useDomainEmbedding) {
            this.domainEmbedding = new EmbeddingImpl(new EmbeddingOptions(numDomains, domainEmbedDim));
            register_module("domain_embedding", domainEmbedding);
        } else {
            this.domainEmbedding = null;
        }

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            sharedFn.to(dev, false);
            for (LinearImpl fn : domainFns) fn.to(dev, false);
            if (domainEmbedding != null) domainEmbedding.to(dev, false);
        }
    }

    /**
     * STAR-style: output = h ⊙ (1 + softplus(shared(h) + domain_fn(h)))
     *
     * @param h          [B, D]
     * @param domainIds  [B] long in [0, numDomains)
     * @return [B, D] adapted representation (does NOT concat domain emb; caller may)
     */
    public Tensor forward(Tensor h, Tensor domainIds) {
        Tensor shared = sharedFn.forward(h);
        // Gather per-sample domain FN output. For library simplicity we compute all
        // domain FNs and select via one-hot mix (batch-friendly, no python loop).
        Tensor mixed = torch.zeros_like(shared);
        Tensor dom = domainIds.toType(org.bytedeco.pytorch.global.torch.ScalarType.Long);
        for (int d = 0; d < numDomains; d++) {
            Tensor mask = dom.eq(new Scalar((long) d)).toType(
                    org.bytedeco.pytorch.global.torch.ScalarType.Float).unsqueeze(1L);
            Tensor local = domainFns.get(d).forward(h);
            mixed = mixed.add(local.mul(mask));
        }
        Tensor scale = torch.softplus(shared.add(mixed));
        return h.mul(torch.add(scale, new Scalar(1.0f)));
    }

    /** Optional domain embedding lookup [B, E]. */
    public Tensor domainEmbedding(Tensor domainIds) {
        if (domainEmbedding == null) {
            throw new IllegalStateException("DomainAdapter was built without domain embedding");
        }
        return domainEmbedding.forward(domainIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long));
    }

    public boolean hasDomainEmbedding() {
        return domainEmbedding != null;
    }

    public int numDomains() {
        return numDomains;
    }

    public int featureDim() {
        return featureDim;
    }
}
