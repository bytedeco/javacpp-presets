/*
 * SearchConversion — e-commerce search CTR / CVR dual tower specialization.
 *
 * Production context (Amazon / Taobao / JD search):
 *   Query text + user history + item features jointly predict click and
 *   post-click conversion. Query is encoded as a bag / sequence of token ids;
 *   item side reuses EmbeddingLayer; ESMM-style product for CTCVR.
 *
 * References:
 *   - Amazon search ranking literature (DSSM-style query-item matching)
 *   - Alibaba search multi-task (ESMM / ESCM2 on search traffic)
 *   - Existing matching.DSSM / multi_task.ESMM in this library
 */
package org.bytedeco.pytorch.recommend.models.ecommerce;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.layers.EmbeddingLayer;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.industry.AdditiveAttention;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SearchConversion extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public static final int COL_CTR = 0;
    public static final int COL_CVR = 1;
    public static final int COL_CTCVR = 2;

    private final EmbeddingImpl queryEmbedding;
    private final AdditiveAttention queryPool;
    private final EmbeddingLayer itemEmbedding;
    private final MLP fusion;
    private final MLP towerCtr;
    private final MLP towerCvr;
    private final int queryDim;

    public SearchConversion(int queryVocabSize, List<? extends Feature> itemFeatures) {
        this(queryVocabSize, itemFeatures, 64, new long[]{128L, 64L}, DeviceSupport.backend());
    }

    public SearchConversion(int queryVocabSize, List<? extends Feature> itemFeatures,
                            int queryEmbedDim, long[] towerDims, String device) {
        super("SearchConversion");
        if (itemFeatures == null || itemFeatures.isEmpty()) {
            throw new IllegalArgumentException("itemFeatures cannot be empty");
        }
        this.queryDim = queryEmbedDim;

        EmbeddingOptions qOpts = new EmbeddingOptions(Math.max(queryVocabSize, 2), queryEmbedDim);
        qOpts.padding_idx().put(new LongOptional(0L));
        this.queryEmbedding = new EmbeddingImpl(qOpts);
        register_module("query_embedding", queryEmbedding);

        this.queryPool = new AdditiveAttention(queryEmbedDim, queryEmbedDim, device);
        register_module("query_pool", queryPool);

        List<Feature> itemList = new ArrayList<>(itemFeatures);
        int itemDim = 0;
        for (Feature f : itemList) itemDim += f.embedDim();
        this.itemEmbedding = new EmbeddingLayer(itemList, itemList.get(0).embedDim(), device);
        register_module("item_embedding", itemEmbedding);

        long fusedIn = queryEmbedDim + itemDim;
        this.fusion = new MLP(fusedIn, new long[]{towerDims[0]}, towerDims[0], "relu", 0.1f,
                false, false, true, device);
        register_module("fusion", fusion);

        this.towerCtr = new MLP(towerDims[0], towerDims, 1L, "relu", 0.1f, false, false, true, device);
        this.towerCvr = new MLP(towerDims[0], towerDims, 1L, "relu", 0.1f, false, false, true, device);
        register_module("tower_ctr", towerCtr);
        register_module("tower_cvr", towerCvr);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            queryEmbedding.to(dev, false);
        }
    }

    /**
     * @param queryTokenIds [B, Lq] query word / token ids
     * @param itemFeatures  item-side sparse/dense map
     * @return [B, 3] = [p_ctr, p_cvr, p_ctcvr]
     */
    public Tensor forward(Tensor queryTokenIds, Map<String, Tensor> itemFeatures) {
        Tensor qEmb = queryEmbedding.forward(queryTokenIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long));
        Tensor qMask = queryTokenIds.ne(new Scalar(0L)).toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor qVec = queryPool.forward(qEmb, qMask);

        Tensor iVec = itemEmbedding.forward(itemFeatures, Collections.emptyMap(), true);
        TensorVector cat = new TensorVector();
        cat.push_back(qVec);
        cat.push_back(iVec);
        Tensor h = fusion.forward(torch.cat(cat, 1L));

        Tensor pCtr = towerCtr.forward(h).squeeze(1L).sigmoid();
        Tensor pCvr = towerCvr.forward(h).squeeze(1L).sigmoid();
        Tensor pCtcvr = pCtr.mul(pCvr);

        TensorVector out = new TensorVector();
        out.push_back(pCtr.unsqueeze(1L));
        out.push_back(pCvr.unsqueeze(1L));
        out.push_back(pCtcvr.unsqueeze(1L));
        return torch.cat(out, 1L);
    }
}
