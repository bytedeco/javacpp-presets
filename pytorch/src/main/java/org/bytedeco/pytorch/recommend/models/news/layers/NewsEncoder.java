/*
 * Word-level news encoder used by Microsoft NRMS / NAML / LSTUR / NPA.
 *
 * Pipeline (MIND paper family):
 *   token ids -> word embedding (+ optional positional) -> Multi-Head Self-Attention
 *   -> Additive Attention pooling -> news vector
 *
 * References:
 *   Wu et al., EMNLP 2019 (NRMS)
 *   Wu et al., IJCAI 2019 (NAML)
 *   An et al., ACL 2019 (LSTUR)
 *   Wu et al., WWW 2019 (NPA)
 */
package org.bytedeco.pytorch.recommend.models.news.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.industry.AdditiveAttention;
import org.bytedeco.pytorch.recommend.basic.layers.industry.MultiHeadSelfAttention;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NewsEncoder extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl wordEmbedding;
    private final MultiHeadSelfAttention selfAttention;
    private final AdditiveAttention additiveAttention;
    private final DropoutImpl dropout;
    private final int embedDim;

    public NewsEncoder(int vocabSize, int embedDim, int numHeads, int attentionDim) {
        this(vocabSize, embedDim, numHeads, attentionDim, 0.2f, DeviceSupport.backend());
    }

    public NewsEncoder(int vocabSize, int embedDim, int numHeads, int attentionDim,
                       float dropoutProb, String device) {
        super("NewsEncoder");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;

        EmbeddingOptions opts = new EmbeddingOptions(vocabSize, embedDim);
        opts.padding_idx().put(new LongOptional(0L));
        this.wordEmbedding = new EmbeddingImpl(opts);
        register_module("word_embedding", wordEmbedding);

        this.selfAttention = new MultiHeadSelfAttention(embedDim, numHeads, dropoutProb, device);
        register_module("self_attention", selfAttention);

        this.additiveAttention = new AdditiveAttention(embedDim, attentionDim, device);
        register_module("additive_attention", additiveAttention);

        this.dropout = new DropoutImpl(dropoutProb);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            wordEmbedding.to(dev, false);
        }
    }

    /**
     * @param tokenIds [B, L] long word / token ids (0 = pad)
     * @param mask     optional [B, L]
     * @return news vector [B, embedDim]
     */
    public Tensor forward(Tensor tokenIds, Tensor mask) {
        Tensor emb = dropout.forward(wordEmbedding.forward(
                tokenIds.toType(org.bytedeco.pytorch.global.torch.ScalarType.Long)));
        Tensor ctx = selfAttention.forward(emb, mask);
        return additiveAttention.forward(ctx, mask);
    }

    public Tensor forward(Tensor tokenIds) {
        // build pad mask from token ids == 0
        Tensor mask = tokenIds.ne(new Scalar(0L)).toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        return forward(tokenIds, mask);
    }

    public int embedDim() {
        return embedDim;
    }
}
