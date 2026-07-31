/*
 * DKN — Deep Knowledge-Aware Network for news recommendation.
 *
 * Reference:
 *   Wang et al., "DKN: Deep Knowledge-Aware Network for News Recommendation",
 *   WWW 2018 (Microsoft). https://dl.acm.org/doi/10.1145/3178876.3186175
 *
 * Core idea: enrich word embeddings with entity embeddings from a knowledge
 * graph (TransE / TransR style pre-trained). Multiple CNN filters with
 * different window sizes extract multi-scale patterns; an attention network
 * aggregates historical clicks w.r.t. the candidate.
 *
 * Library design:
 *   - Accepts pre-aligned word ids and entity ids (same sequence length).
 *   - Entity embedding table is separate (can be initialized from TransE).
 *   - KCNN fuses word+entity via concatenation then multi-filter CNN.
 *   - Candidate-aware attention over user click history.
 *
 * Note: full TransE training is out of scope; callers supply entity ids /
 * pre-trained entity embedding weights via the entity embedding module.
 */
package org.bytedeco.pytorch.recommend.models.news;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DKN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl wordEmbedding;
    private final EmbeddingImpl entityEmbedding;
    private final List<Conv1dImpl> convFilters = new ArrayList<>();
    private final LinearImpl attnQuery;
    private final LinearImpl attnKey;
    private final LinearImpl attnScore;
    private final LinearImpl outputProj;
    private final ReLUImpl relu;
    private final DropoutImpl dropout;
    private final int embedDim;
    private final int numFilters;
    private final int[] windowSizes;
    private final int newsDim; // numFilters * numWindows

    public DKN(int wordVocabSize, int entityVocabSize) {
        this(wordVocabSize, entityVocabSize, 100, 50, new int[]{1, 2, 3}, 100,
                0.2f, DeviceSupport.backend());
    }

    public DKN(int wordVocabSize, int entityVocabSize, int wordEmbedDim, int entityEmbedDim,
               int[] windowSizes, int numFilters, float dropoutProb, String device) {
        super("DKN");
        this.embedDim = wordEmbedDim;
        this.numFilters = numFilters;
        this.windowSizes = windowSizes != null ? windowSizes.clone() : new int[]{1, 2, 3};
        this.newsDim = numFilters * this.windowSizes.length;
        this.relu = new ReLUImpl();
        this.dropout = new DropoutImpl(dropoutProb);

        EmbeddingOptions wOpts = new EmbeddingOptions(wordVocabSize, wordEmbedDim);
        wOpts.padding_idx().put(new LongOptional(0L));
        this.wordEmbedding = new EmbeddingImpl(wOpts);
        register_module("word_embedding", wordEmbedding);

        EmbeddingOptions eOpts = new EmbeddingOptions(Math.max(entityVocabSize, 1), entityEmbedDim);
        eOpts.padding_idx().put(new LongOptional(0L));
        this.entityEmbedding = new EmbeddingImpl(eOpts);
        register_module("entity_embedding", entityEmbedding);

        // KCNN: input channels = wordEmbedDim + entityEmbedDim (concat on channel after transpose)
        int inChannels = wordEmbedDim + entityEmbedDim;
        for (int i = 0; i < this.windowSizes.length; i++) {
            int w = this.windowSizes[i];
            // kernel / padding via ExpandingArray LongPointer (see CIN / CausalConv1d)
            LongPointer kernel = new LongPointer(new long[]{w});
            Conv1dOptions copts = new Conv1dOptions(inChannels, numFilters, kernel);
            copts.padding().put(new LongPointer(new long[]{Math.max(w / 2, 0)}));
            Conv1dImpl conv = new Conv1dImpl(copts);
            register_module("conv_" + i, conv);
            convFilters.add(conv);
        }

        // candidate-aware attention
        this.attnQuery = new LinearImpl(newsDim, newsDim);
        this.attnKey = new LinearImpl(newsDim, newsDim);
        this.attnScore = new LinearImpl(newsDim, 1L);
        this.outputProj = new LinearImpl(newsDim * 2L, 1L);
        register_module("attn_query", attnQuery);
        register_module("attn_key", attnKey);
        register_module("attn_score", attnScore);
        register_module("output_proj", outputProj);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            wordEmbedding.to(dev, false);
            entityEmbedding.to(dev, false);
            for (Conv1dImpl c : convFilters) c.to(dev, false);
            attnQuery.to(dev, false);
            attnKey.to(dev, false);
            attnScore.to(dev, false);
            outputProj.to(dev, false);
        }
    }

    /**
     * KCNN news encoder.
     * @param wordIds   [B, L]
     * @param entityIds [B, L] aligned entity ids (0 if no entity)
     * @return [B, newsDim]
     */
    public Tensor encodeNews(Tensor wordIds, Tensor entityIds) {
        Tensor w = wordEmbedding.forward(wordIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)); // [B, L, Dw]
        Tensor e = entityEmbedding.forward(entityIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long)); // [B, L, De]
        TensorVector cat = new TensorVector();
        cat.push_back(w);
        cat.push_back(e);
        Tensor we = torch.cat(cat, 2L); // [B, L, Dw+De]
        // Conv1d expects [B, C, L]
        Tensor x = we.transpose(1, 2);
        List<Tensor> pooled = new ArrayList<>();
        for (Conv1dImpl conv : convFilters) {
            Tensor h = relu.forward(conv.forward(x)); // [B, F, L']
            // max-over-time pooling (torch.max returns values+indices)
            Tensor p = torch.max(h, 2L).get0(); // [B, F]
            pooled.add(p);
        }
        TensorVector pvec = new TensorVector();
        for (Tensor p : pooled) pvec.push_back(p);
        return dropout.forward(torch.cat(pvec, 1L)); // [B, F * numWindows]
    }

    /**
     * @param histWord    [B, N, L]
     * @param histEntity  [B, N, L]
     * @param candWord    [B, C, L]
     * @param candEntity  [B, C, L]
     * @return scores [B, C]
     */
    public Tensor forward(Tensor histWord, Tensor histEntity,
                          Tensor candWord, Tensor candEntity) {
        long batch = histWord.size(0);
        long nHist = histWord.size(1);
        long nCand = candWord.size(1);
        long len = histWord.size(2);

        Tensor hW = histWord.contiguous().view(batch * nHist, len);
        Tensor hE = histEntity.contiguous().view(batch * nHist, len);
        Tensor histNews = encodeNews(hW, hE).view(batch, nHist, newsDim); // [B, N, D]

        Tensor cW = candWord.contiguous().view(batch * nCand, len);
        Tensor cE = candEntity.contiguous().view(batch * nCand, len);
        Tensor candNews = encodeNews(cW, cE).view(batch, nCand, newsDim); // [B, C, D]

        // Candidate-aware attention for each candidate over history
        // For efficiency: compute attention per candidate via broadcast
        // q = attnQuery(cand) [B, C, D], k = attnKey(hist) [B, N, D]
        Tensor k = attnKey.forward(histNews); // [B, N, D]
        Tensor q = attnQuery.forward(candNews); // [B, C, D]

        // scores_attn[b,c,n] via expanded sum
        // use: tanh(q_c + k_n) then linear
        Tensor qExp = q.unsqueeze(2L); // [B, C, 1, D]
        Tensor kExp = k.unsqueeze(1L); // [B, 1, N, D]
        Tensor attInput = tanh(qExp.add(kExp)); // [B, C, N, D]
        Tensor attScores = attnScore.forward(attInput).squeeze(3L); // [B, C, N]
        // mask empty history news (all-zero word ids)
        Tensor histMask = histWord.sum(2L).ne(new Scalar(0L))
                .toType(org.bytedeco.pytorch.global.torch.ScalarType.Float); // [B, N]
        Tensor m = histMask.unsqueeze(1L); // [B, 1, N]
        Tensor neg = torch.full_like(attScores, new Scalar(-1e9f));
        attScores = attScores.mul(m).add(neg.mul(torch.sub(torch.ones_like(m), m)));
        Tensor attWeights = attScores.softmax(2L); // [B, C, N]
        Tensor userVec = torch.matmul(attWeights, histNews); // [B, C, D]

        TensorVector fuse = new TensorVector();
        fuse.push_back(userVec);
        fuse.push_back(candNews);
        Tensor logits = outputProj.forward(torch.cat(fuse, 2L)).squeeze(2L); // [B, C]
        return logits;
    }

    private Tensor tanh(Tensor t) {
        return t.tanh();
    }

    /** Expose entity embedding for TransE weight loading. */
    public EmbeddingImpl entityEmbedding() {
        return entityEmbedding;
    }

    public int newsDim() {
        return newsDim;
    }
}
