/*
 * NAML — Neural News Recommendation with Attentive Multi-View Learning.
 *
 * Reference:
 *   Wu et al., "Neural News Recommendation with Attentive Multi-View Learning",
 *   IJCAI 2019 (Microsoft). https://www.ijcai.org/proceedings/2019/0536.pdf
 *
 * Multi-view news encoder:
 *   title view  : word emb -> CNN / MHSA -> additive attn
 *   abstract view: same
 *   category / subcategory: embedding lookup
 *   view-level additive attention fuses views into one news vector.
 *
 * This implementation uses MHSA (consistent with NRMS) for title/abstract text
 * views and embedding tables for categorical views — matching the paper's
 * multi-view spirit while reusing library attention primitives.
 */
package org.bytedeco.pytorch.utils.recommend.models.news;

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
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.AdditiveAttention;
import org.bytedeco.pytorch.utils.recommend.models.news.layers.NewsEncoder;
import org.bytedeco.pytorch.utils.recommend.models.news.layers.UserEncoder;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NAML extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final NewsEncoder titleEncoder;
    private final NewsEncoder abstractEncoder;
    private final EmbeddingImpl categoryEmbedding;
    private final EmbeddingImpl subcategoryEmbedding;
    private final AdditiveAttention viewAttention;
    private final UserEncoder userEncoder;
    private final int embedDim;
    private final boolean useAbstract;
    private final boolean useCategory;

    public NAML(int wordVocabSize, int numCategories, int numSubcategories) {
        this(wordVocabSize, numCategories, numSubcategories, 256, 16, 200,
                true, true, 0.2f, DeviceSupport.backend());
    }

    public NAML(int wordVocabSize, int numCategories, int numSubcategories,
                int embedDim, int numHeads, int attentionDim,
                boolean useAbstract, boolean useCategory,
                float dropout, String device) {
        super("NAML");
        this.embedDim = embedDim;
        this.useAbstract = useAbstract;
        this.useCategory = useCategory;

        this.titleEncoder = new NewsEncoder(wordVocabSize, embedDim, numHeads, attentionDim, dropout, device);
        register_module("title_encoder", titleEncoder);

        if (useAbstract) {
            this.abstractEncoder = new NewsEncoder(wordVocabSize, embedDim, numHeads, attentionDim, dropout, device);
            register_module("abstract_encoder", abstractEncoder);
        } else {
            this.abstractEncoder = null;
        }

        if (useCategory) {
            EmbeddingOptions cOpts = new EmbeddingOptions(Math.max(numCategories, 1), embedDim);
            cOpts.padding_idx().put(new LongOptional(0L));
            this.categoryEmbedding = new EmbeddingImpl(cOpts);
            register_module("category_embedding", categoryEmbedding);

            EmbeddingOptions sOpts = new EmbeddingOptions(Math.max(numSubcategories, 1), embedDim);
            sOpts.padding_idx().put(new LongOptional(0L));
            this.subcategoryEmbedding = new EmbeddingImpl(sOpts);
            register_module("subcategory_embedding", subcategoryEmbedding);
        } else {
            this.categoryEmbedding = null;
            this.subcategoryEmbedding = null;
        }

        // view attention pools V views each of dim embedDim
        this.viewAttention = new AdditiveAttention(embedDim, attentionDim, device);
        register_module("view_attention", viewAttention);

        this.userEncoder = new UserEncoder(embedDim, numHeads, attentionDim, dropout, device);
        register_module("user_encoder", userEncoder);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            if (categoryEmbedding != null) categoryEmbedding.to(dev, false);
            if (subcategoryEmbedding != null) subcategoryEmbedding.to(dev, false);
        }
    }

    /**
     * Encode multi-view news.
     * @param titleIds    [B, L]
     * @param abstractIds [B, L] or null if disabled
     * @param categoryIds [B] or null
     * @param subcategoryIds [B] or null
     * @return [B, D]
     */
    public Tensor encodeNews(Tensor titleIds, Tensor abstractIds,
                             Tensor categoryIds, Tensor subcategoryIds) {
        java.util.List<Tensor> views = new java.util.ArrayList<>();
        views.add(titleEncoder.forward(titleIds));
        if (useAbstract && abstractIds != null) {
            views.add(abstractEncoder.forward(abstractIds));
        }
        if (useCategory && categoryIds != null && categoryEmbedding != null) {
            views.add(categoryEmbedding.forward(categoryIds.toType(
                    org.bytedeco.pytorch.global.torch.ScalarType.Long)));
        }
        if (useCategory && subcategoryIds != null && subcategoryEmbedding != null) {
            views.add(subcategoryEmbedding.forward(subcategoryIds.toType(
                    org.bytedeco.pytorch.global.torch.ScalarType.Long)));
        }
        // stack views: [B, V, D]
        TensorVector vec = new TensorVector();
        for (Tensor v : views) {
            vec.push_back(v.unsqueeze(1L));
        }
        Tensor stacked = torch.cat(vec, 1L);
        return viewAttention.forward(stacked);
    }

    /**
     * @param histTitle [B, N, L]
     * @param candTitle [B, C, L]
     * Optional multi-view side features may be null (title-only fallback).
     */
    public Tensor forward(Tensor histTitle, Tensor candTitle) {
        return forward(histTitle, (Tensor) null, (Tensor) null, (Tensor) null, candTitle, (Tensor) null, (Tensor) null, (Tensor) null);
    }

    public Tensor forward(Tensor histTitle, Tensor histAbstract, Tensor histCat, Tensor histSubcat,
                          Tensor candTitle, Tensor candAbstract, Tensor candCat, Tensor candSubcat) {
        long batch = histTitle.size(0);
        long nHist = histTitle.size(1);
        long nCand = candTitle.size(1);
        long titleLen = histTitle.size(2);

        Tensor histFlat = histTitle.contiguous().view(batch * nHist, titleLen);
        Tensor histAbsFlat = flattenOptional(histAbstract, batch, nHist, titleLen);
        Tensor histCatFlat = flattenOptional1d(histCat, batch, nHist);
        Tensor histSubFlat = flattenOptional1d(histSubcat, batch, nHist);
        Tensor histNews = encodeNews(histFlat, histAbsFlat, histCatFlat, histSubFlat)
                .view(batch, nHist, embedDim);
        Tensor histMask = histTitle.sum(2L).ne(new Scalar(0L))
                .toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor userVec = userEncoder.forward(histNews, histMask);

        Tensor candFlat = candTitle.contiguous().view(batch * nCand, titleLen);
        Tensor candAbsFlat = flattenOptional(candAbstract, batch, nCand, titleLen);
        Tensor candCatFlat = flattenOptional1d(candCat, batch, nCand);
        Tensor candSubFlat = flattenOptional1d(candSubcat, batch, nCand);
        Tensor candNews = encodeNews(candFlat, candAbsFlat, candCatFlat, candSubFlat)
                .view(batch, nCand, embedDim);

        return candNews.mul(userVec.unsqueeze(1L)).sum(2L);
    }

    private static Tensor flattenOptional(Tensor t, long batch, long n, long len) {
        if (t == null || t.isNull() || t.numel() == 0) return null;
        return t.contiguous().view(batch * n, len);
    }

    private static Tensor flattenOptional1d(Tensor t, long batch, long n) {
        if (t == null || t.isNull() || t.numel() == 0) return null;
        return t.contiguous().view(batch * n);
    }

    public int embedDim() {
        return embedDim;
    }
}
