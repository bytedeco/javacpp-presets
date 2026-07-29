/*
 * NPA — Neural News Recommendation with Personalized Attention.
 *
 * Reference:
 *   Wu et al., "NPA: Neural News Recommendation with Personalized Attention",
 *   WWW 2019 (Microsoft). https://dl.acm.org/doi/10.1145/3308558.3313562
 *
 * Key idea: attention queries are conditioned on the user id embedding so that
 * different users attend to different words / news in their history.
 *
 *   q_word = tanh(W_w * userEmb + b_w)
 *   word attention scores over title words use q_word (personalized)
 *   similarly q_news personalizes aggregation of clicked news.
 */
package org.bytedeco.pytorch.utils.recommend.models.news;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.TanhImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NPA extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl wordEmbedding;
    private final EmbeddingImpl userEmbedding;
    private final LinearImpl wordPrefProj;   // userEmb -> preference query for words
    private final LinearImpl newsPrefProj;   // userEmb -> preference query for news
    private final LinearImpl wordAttnProj;   // word emb -> attn dim
    private final LinearImpl newsAttnProj;   // news emb -> attn dim
    private final TanhImpl tanh;
    private final DropoutImpl dropout;
    private final int embedDim;
    private final int preferenceDim;

    public NPA(int wordVocabSize, int numUsers) {
        this(wordVocabSize, numUsers, 256, 200, 0.2f, DeviceSupport.backend());
    }

    public NPA(int wordVocabSize, int numUsers, int embedDim, int preferenceDim,
               float dropoutProb, String device) {
        super("NPA");
        this.embedDim = embedDim;
        this.preferenceDim = preferenceDim;
        this.tanh = new TanhImpl();
        this.dropout = new DropoutImpl(dropoutProb);

        EmbeddingOptions wOpts = new EmbeddingOptions(wordVocabSize, embedDim);
        wOpts.padding_idx().put(new LongOptional(0L));
        this.wordEmbedding = new EmbeddingImpl(wOpts);
        register_module("word_embedding", wordEmbedding);

        EmbeddingOptions uOpts = new EmbeddingOptions(Math.max(numUsers, 1), embedDim);
        uOpts.padding_idx().put(new LongOptional(0L));
        this.userEmbedding = new EmbeddingImpl(uOpts);
        register_module("user_embedding", userEmbedding);

        this.wordPrefProj = new LinearImpl(embedDim, preferenceDim);
        this.newsPrefProj = new LinearImpl(embedDim, preferenceDim);
        this.wordAttnProj = new LinearImpl(embedDim, preferenceDim);
        this.newsAttnProj = new LinearImpl(embedDim, preferenceDim);
        register_module("word_pref_proj", wordPrefProj);
        register_module("news_pref_proj", newsPrefProj);
        register_module("word_attn_proj", wordAttnProj);
        register_module("news_attn_proj", newsAttnProj);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            wordEmbedding.to(dev, false);
            userEmbedding.to(dev, false);
            wordPrefProj.to(dev, false);
            newsPrefProj.to(dev, false);
            wordAttnProj.to(dev, false);
            newsAttnProj.to(dev, false);
        }
    }

    /** Personalized additive attention: query from user, keys from sequence. */
    private Tensor personalizedPool(Tensor seq, Tensor prefQuery, LinearImpl proj, Tensor mask) {
        // seq [B, L, D], prefQuery [B, P]
        Tensor keys = tanh.forward(proj.forward(seq));          // [B, L, P]
        Tensor scores = keys.mul(prefQuery.unsqueeze(1L)).sum(2L); // [B, L]
        if (mask != null && !mask.isNull() && mask.numel() > 0) {
            Tensor m = mask.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
            Tensor neg = torch.full_like(scores, new Scalar(-1e9f));
            scores = scores.mul(m).add(neg.mul(torch.sub(torch.ones_like(m), m)));
        }
        Tensor weights = scores.softmax(1L).unsqueeze(2L); // [B, L, 1]
        return seq.mul(weights).sum(1L);                   // [B, D]
    }

    /**
     * Encode news titles with personalized word attention.
     * @param tokenIds [B, L]
     * @param userEmb  [B, D]
     */
    public Tensor encodeNews(Tensor tokenIds, Tensor userEmb) {
        Tensor emb = dropout.forward(wordEmbedding.forward(
                tokenIds.toType(org.bytedeco.pytorch.global.torch.ScalarType.Long)));
        Tensor mask = tokenIds.ne(new Scalar(0L)).toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor wordPref = tanh.forward(wordPrefProj.forward(userEmb)); // [B, P]
        return personalizedPool(emb, wordPref, wordAttnProj, mask);
    }

    /**
     * @param userIds           [B]
     * @param historyTokenIds   [B, N, L]
     * @param candidateTokenIds [B, C, L]
     * @return scores [B, C]
     */
    public Tensor forward(Tensor userIds, Tensor historyTokenIds, Tensor candidateTokenIds) {
        long batch = historyTokenIds.size(0);
        long nHist = historyTokenIds.size(1);
        long nCand = candidateTokenIds.size(1);
        long titleLen = historyTokenIds.size(2);

        Tensor userEmb = userEmbedding.forward(userIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long));

        // Encode each history news with the same user preference query
        Tensor histFlat = historyTokenIds.contiguous().view(batch * nHist, titleLen);
        Tensor userEmbHist = userEmb.unsqueeze(1L).expand(batch, nHist, embedDim)
                .contiguous().view(batch * nHist, embedDim);
        Tensor histNews = encodeNews(histFlat, userEmbHist).view(batch, nHist, embedDim);
        Tensor histMask = historyTokenIds.sum(2L).ne(new Scalar(0L))
                .toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor newsPref = tanh.forward(newsPrefProj.forward(userEmb));
        Tensor userVec = personalizedPool(histNews, newsPref, newsAttnProj, histMask);

        Tensor candFlat = candidateTokenIds.contiguous().view(batch * nCand, titleLen);
        Tensor userEmbCand = userEmb.unsqueeze(1L).expand(batch, nCand, embedDim)
                .contiguous().view(batch * nCand, embedDim);
        Tensor candNews = encodeNews(candFlat, userEmbCand).view(batch, nCand, embedDim);

        return candNews.mul(userVec.unsqueeze(1L)).sum(2L);
    }

    public int embedDim() {
        return embedDim;
    }
}
