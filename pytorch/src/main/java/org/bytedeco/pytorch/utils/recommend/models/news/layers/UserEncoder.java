/*
 * User encoder over clicked-news sequence (Microsoft MIND model family).
 *
 * Pipeline:
 *   news vectors of click history [B, N, D] -> Multi-Head Self-Attention
 *   -> Additive Attention -> user vector [B, D]
 *
 * Used by NRMS / NAML. LSTUR replaces this with GRU + user-id preference.
 */
package org.bytedeco.pytorch.utils.recommend.models.news.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.AdditiveAttention;
import org.bytedeco.pytorch.utils.recommend.basic.layers.industry.MultiHeadSelfAttention;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class UserEncoder extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final MultiHeadSelfAttention selfAttention;
    private final AdditiveAttention additiveAttention;
    private final DropoutImpl dropout;
    private final int embedDim;

    public UserEncoder(int embedDim, int numHeads, int attentionDim) {
        this(embedDim, numHeads, attentionDim, 0.2f, DeviceSupport.backend());
    }

    public UserEncoder(int embedDim, int numHeads, int attentionDim,
                       float dropoutProb, String device) {
        super("UserEncoder");
        this.embedDim = embedDim;
        this.selfAttention = new MultiHeadSelfAttention(embedDim, numHeads, dropoutProb, device);
        register_module("self_attention", selfAttention);
        this.additiveAttention = new AdditiveAttention(embedDim, attentionDim, device);
        register_module("additive_attention", additiveAttention);
        this.dropout = new DropoutImpl(dropoutProb);
    }

    /**
     * @param newsSeq [B, N, D] encoded clicked news
     * @param mask    optional [B, N]
     * @return user vector [B, D]
     */
    public Tensor forward(Tensor newsSeq, Tensor mask) {
        Tensor h = dropout.forward(selfAttention.forward(newsSeq, mask));
        return additiveAttention.forward(h, mask);
    }

    public Tensor forward(Tensor newsSeq) {
        return forward(newsSeq, (Tensor) null);
    }

    public int embedDim() {
        return embedDim;
    }
}
