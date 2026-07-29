/*
 * MultiDomainCTR — STAR-style multi-domain CTR for e-commerce / ads.
 *
 * Reference:
 *   Sheng et al., "One Model to Serve All: Star Topology Adaptive Recommender
 *   for Multi-Domain CTR Prediction", CIKM 2021 (Alibaba).
 *
 *   Shared centered network + domain-specific FN (implemented via DomainAdapter)
 *   serves many business domains (homepage, search, detail-page, push, ...)
 *   with one model — standard pattern at Taobao / Lazada / Shopee.
 */
package org.bytedeco.pytorch.utils.recommend.models.ecommerce;

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
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.DomainAdapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MultiDomainCTR extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingLayer embedding;
    private final DomainAdapter domainAdapter;
    private final MLP sharedMlp;
    private final MLP outputLayer;
    private final int featDim;
    private final int numDomains;

    public MultiDomainCTR(List<? extends Feature> features, int numDomains) {
        this(features, numDomains, new long[]{256L, 128L, 64L}, DeviceSupport.backend());
    }

    public MultiDomainCTR(List<? extends Feature> features, int numDomains,
                          long[] hiddenDims, String device) {
        super("MultiDomainCTR");
        if (features == null || features.isEmpty()) {
            throw new IllegalArgumentException("features cannot be empty");
        }
        if (numDomains < 1) {
            throw new IllegalArgumentException("numDomains must be >= 1");
        }
        this.numDomains = numDomains;
        List<Feature> featList = new ArrayList<>(features);
        int dim = 0;
        for (Feature f : featList) dim += f.embedDim();
        this.featDim = dim;

        this.embedding = new EmbeddingLayer(featList, featList.get(0).embedDim(), device);
        register_module("embedding", embedding);

        this.domainAdapter = new DomainAdapter(dim, numDomains, 16, true, device);
        register_module("domain_adapter", domainAdapter);

        // After adapter, optionally concat domain embedding
        int domainEmbDim = 16;
        long mlpIn = dim + domainEmbDim;
        this.sharedMlp = new MLP(mlpIn, hiddenDims, hiddenDims[hiddenDims.length - 1],
                "relu", 0.1f, true, false, true, device);
        register_module("shared_mlp", sharedMlp);

        this.outputLayer = new MLP(hiddenDims[hiddenDims.length - 1], new long[]{}, 1L,
                "relu", 0.0f, false, false, true, device);
        register_module("output", outputLayer);
    }

    /**
     * @param features   feature map
     * @param domainIds  [B] domain id
     * @return CTR probability [B]
     */
    public Tensor forward(Map<String, Tensor> features, Tensor domainIds) {
        Tensor h = embedding.forward(features, Collections.emptyMap(), true);
        Tensor adapted = domainAdapter.forward(h, domainIds);
        Tensor dEmb = domainAdapter.domainEmbedding(domainIds);
        TensorVector cat = new TensorVector();
        cat.push_back(adapted);
        cat.push_back(dEmb);
        Tensor x = torch.cat(cat, 1L);
        Tensor rep = sharedMlp.forward(x);
        return outputLayer.forward(rep).squeeze(1L).sigmoid();
    }

    public int numDomains() {
        return numDomains;
    }
}
