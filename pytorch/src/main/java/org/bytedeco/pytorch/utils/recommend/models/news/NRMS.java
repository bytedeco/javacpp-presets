/*
 * NRMS — Neural News Recommendation with Multi-Head Self-Attention.
 *
 * Reference:
 *   Wu et al., "Neural News Recommendation with Multi-Head Self-Attention",
 *   EMNLP 2019 (Microsoft Research Asia). Companion model of the MIND dataset.
 *   https://aclanthology.org/D19-1678/
 *
 * Architecture:
 *   NewsEncoder (word emb -> MHSA -> additive attn)
 *   UserEncoder (clicked news -> MHSA -> additive attn)
 *   Score = dot(user, candidate_news)  (optionally sigmoid for CTR)
 *
 * Input conventions (library-friendly, matching MIND preprocessing):
 *   historyTokenIds:    [B, N, L]  word ids of N clicked news (each length L)
 *   candidateTokenIds:  [B, C, L]  word ids of C candidate news
 *   historyMask:        [B, N]     optional
 *   candidateMask:      [B, C]     optional (usually all-ones)
 * Output:
 *   scores [B, C]
 */
package org.bytedeco.pytorch.utils.recommend.models.news;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.models.news.layers.NewsEncoder;
import org.bytedeco.pytorch.utils.recommend.models.news.layers.UserEncoder;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NRMS extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final NewsEncoder newsEncoder;
    private final UserEncoder userEncoder;
    private final int embedDim;

    public NRMS(int vocabSize) {
        this(vocabSize, 256, 16, 200, 0.2f, DeviceSupport.backend());
    }

    public NRMS(int vocabSize, int embedDim, int numHeads, int attentionDim,
                float dropout, String device) {
        super("NRMS");
        this.embedDim = embedDim;
        this.newsEncoder = new NewsEncoder(vocabSize, embedDim, numHeads, attentionDim, dropout, device);
        register_module("news_encoder", newsEncoder);
        this.userEncoder = new UserEncoder(embedDim, numHeads, attentionDim, dropout, device);
        register_module("user_encoder", userEncoder);
    }

    /**
     * Encode a flat batch of news token sequences.
     * @param tokenIds [B*N, L] or [B, L]
     */
    public Tensor encodeNews(Tensor tokenIds) {
        return newsEncoder.forward(tokenIds);
    }

    /**
     * Full ranking forward.
     * @param historyTokenIds   [B, N, L]
     * @param candidateTokenIds [B, C, L]
     * @return click scores [B, C]
     */
    public Tensor forward(Tensor historyTokenIds, Tensor candidateTokenIds) {
        long batch = historyTokenIds.size(0);
        long nHist = historyTokenIds.size(1);
        long nCand = candidateTokenIds.size(1);
        long titleLen = historyTokenIds.size(2);

        // Encode history news: reshape to [B*N, L] -> [B*N, D] -> [B, N, D]
        Tensor histFlat = historyTokenIds.contiguous().view(batch * nHist, titleLen);
        Tensor histNews = newsEncoder.forward(histFlat).view(batch, nHist, embedDim);
        Tensor histMask = historyTokenIds.sum(2L).ne(new Scalar(0L))
                .toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor userVec = userEncoder.forward(histNews, histMask); // [B, D]

        // Encode candidates: [B*C, L] -> [B, C, D]
        Tensor candFlat = candidateTokenIds.contiguous().view(batch * nCand, titleLen);
        Tensor candNews = newsEncoder.forward(candFlat).view(batch, nCand, embedDim);

        // Dot-product scoring: [B, C]
        Tensor scores = candNews.mul(userVec.unsqueeze(1L)).sum(2L);
        return scores;
    }

    /**
     * Single-candidate convenience (C=1): returns [B] scores.
     */
    public Tensor forwardSingle(Tensor historyTokenIds, Tensor candidateTokenIds) {
        // candidateTokenIds: [B, L]
        Tensor cand = candidateTokenIds.unsqueeze(1L); // [B, 1, L]
        return forward(historyTokenIds, cand).squeeze(1L);
    }

    public int embedDim() {
        return embedDim;
    }

    public NewsEncoder newsEncoder() {
        return newsEncoder;
    }

    public UserEncoder userEncoder() {
        return userEncoder;
    }
}
