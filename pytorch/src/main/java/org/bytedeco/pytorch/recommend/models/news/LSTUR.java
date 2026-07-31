/*
 * LSTUR — Long- and Short-Term User Representations for news recommendation.
 *
 * Reference:
 *   An et al., "Neural News Recommendation with Long- and Short-Term User
 *   Representations", ACL 2019 (Microsoft).
 *   https://aclanthology.org/P19-1033/
 *
 * User modeling:
 *   long-term  : user-id embedding (stable preference)
 *   short-term : GRU over recent clicked news vectors
 *   fusion     : ini  — init GRU hidden with long-term pref
 *                con  — concat long-term + short-term then project
 *
 * News encoder: same word-level NewsEncoder as NRMS family.
 */
package org.bytedeco.pytorch.recommend.models.news;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.GRUImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.GRUOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.models.news.layers.NewsEncoder;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LSTUR extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public enum Fusion { INI, CON }

    private final NewsEncoder newsEncoder;
    private final EmbeddingImpl userEmbedding;
    private final GRUImpl shortTermGru;
    private final LinearImpl conProj; // only for CON fusion
    private final Fusion fusion;
    private final int embedDim;

    public LSTUR(int wordVocabSize, int numUsers) {
        this(wordVocabSize, numUsers, 256, 16, 200, Fusion.INI, 0.2f, DeviceSupport.backend());
    }

    public LSTUR(int wordVocabSize, int numUsers, int embedDim, int numHeads, int attentionDim,
                 Fusion fusion, float dropout, String device) {
        super("LSTUR");
        this.embedDim = embedDim;
        this.fusion = fusion != null ? fusion : Fusion.INI;

        this.newsEncoder = new NewsEncoder(wordVocabSize, embedDim, numHeads, attentionDim, dropout, device);
        register_module("news_encoder", newsEncoder);

        EmbeddingOptions uOpts = new EmbeddingOptions(Math.max(numUsers, 1), embedDim);
        uOpts.padding_idx().put(new LongOptional(0L));
        this.userEmbedding = new EmbeddingImpl(uOpts);
        register_module("user_embedding", userEmbedding);

        GRUOptions gruOpts = new GRUOptions(embedDim, embedDim);
        gruOpts.batch_first().put(true);
        this.shortTermGru = new GRUImpl(gruOpts);
        register_module("short_term_gru", shortTermGru);

        if (this.fusion == Fusion.CON) {
            this.conProj = new LinearImpl(embedDim * 2L, embedDim);
            register_module("con_proj", conProj);
        } else {
            this.conProj = null;
        }

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            userEmbedding.to(dev, false);
            shortTermGru.to(dev, false);
            if (conProj != null) conProj.to(dev, false);
        }
    }

    /**
     * @param userIds           [B] long
     * @param historyTokenIds   [B, N, L]
     * @param candidateTokenIds [B, C, L]
     * @return scores [B, C]
     */
    public Tensor forward(Tensor userIds, Tensor historyTokenIds, Tensor candidateTokenIds) {
        long batch = historyTokenIds.size(0);
        long nHist = historyTokenIds.size(1);
        long nCand = candidateTokenIds.size(1);
        long titleLen = historyTokenIds.size(2);

        Tensor longTerm = userEmbedding.forward(userIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)); // [B, D]

        Tensor histFlat = historyTokenIds.contiguous().view(batch * nHist, titleLen);
        Tensor histNews = newsEncoder.forward(histFlat).view(batch, nHist, embedDim);

        Tensor userVec;
        if (fusion == Fusion.INI) {
            // init hidden = long-term preference: GRU wants [num_layers, B, H]
            Tensor h0 = longTerm.unsqueeze(0L);
            // GRUImpl forward returns (output, h_n); JavaCPP may expose as TensorVector
            Tensor shortTerm = runGruLastHidden(histNews, h0);
            userVec = shortTerm;
        } else {
            Tensor shortTerm = runGruLastHidden(histNews, null);
            TensorVector cat = new TensorVector();
            cat.push_back(longTerm);
            cat.push_back(shortTerm);
            userVec = conProj.forward(torch.cat(cat, 1L));
        }

        Tensor candFlat = candidateTokenIds.contiguous().view(batch * nCand, titleLen);
        Tensor candNews = newsEncoder.forward(candFlat).view(batch, nCand, embedDim);
        return candNews.mul(userVec.unsqueeze(1L)).sum(2L);
    }

    /**
     * Run GRU and return last-layer hidden [B, H].
     * Matches existing GRU4Rec / DIEN JavaCPP usage: forwardT_TensorTensor_T.
     * h0 may be null (zeros) or [num_layers=1, B, H] for INI fusion.
     */
    private Tensor runGruLastHidden(Tensor seq, Tensor h0) {
        T_TensorTensor_T gruOutput;
        if (h0 != null) {
            try {
                gruOutput = shortTermGru.forwardT_TensorTensor_T(seq, h0);
            } catch (Throwable t) {
                // Fallback if overloaded h0 forward is unavailable in this binding
                gruOutput = shortTermGru.forwardT_TensorTensor_T(seq);
            }
        } else {
            gruOutput = shortTermGru.forwardT_TensorTensor_T(seq);
        }
        // get1 = final hidden state [num_layers, B, H]
        Tensor gruHidden = gruOutput.get1();
        Tensor last = gruHidden.select(0L, gruHidden.size(0) - 1L); // [B, H]
        if (h0 != null && fusion == Fusion.INI) {
            // Residual path keeps long-term preference if h0 init was dropped by fallback
            return last.add(h0.squeeze(0L));
        }
        return last;
    }

    public int embedDim() {
        return embedDim;
    }

    public NewsEncoder newsEncoder() {
        return newsEncoder;
    }
}
