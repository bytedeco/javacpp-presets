/*
 * Additive (Bahdanau-style) attention used by Microsoft MIND news models:
 * NRMS, NAML, LSTUR, NPA (EMNLP/IJCAI/WWW 2019).
 *
 * Reference:
 *   Wu et al., "Neural News Recommendation with Multi-Head Self-Attention", EMNLP 2019
 *   Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
 *
 * Input:  x [batch, seq_len, dim]
 * Output: [batch, dim]  (weighted sum over sequence)
 * Optional mask: [batch, seq_len] with 1 = valid, 0 = pad.
 */
package org.bytedeco.pytorch.recommend.basic.layers.industry;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.TanhImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AdditiveAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl projection;
    private final LinearImpl query;
    private final TanhImpl tanh;

    public AdditiveAttention(int inputDim, int attentionDim) {
        this(inputDim, attentionDim, DeviceSupport.backend());
    }

    public AdditiveAttention(int inputDim, int attentionDim, String device) {
        super("AdditiveAttention");
        this.projection = new LinearImpl(inputDim, attentionDim);
        this.query = new LinearImpl(new LinearOptions(attentionDim, 1L).bias(false));
        this.tanh = new TanhImpl();
        register_module("projection", projection);
        register_module("query", query);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            projection.to(dev, false);
            query.to(dev, false);
        }
    }

    /**
     * @param x    [B, L, D]
     * @param mask optional [B, L] (1=keep, 0=pad); null = no mask
     * @return [B, D]
     */
    public Tensor forward(Tensor x, Tensor mask) {
        // scores: [B, L, 1]
        Tensor scores = query.forward(tanh.forward(projection.forward(x)));
        if (mask != null && !mask.isNull() && mask.numel() > 0) {
            // large negative for padded positions
            Tensor maskF = mask.toType(org.bytedeco.pytorch.global.torch.ScalarType.Float)
                    .unsqueeze(2L);
            Tensor neg = torch.full_like(scores, new Scalar(-1e9f));
            scores = scores.mul(maskF).add(neg.mul(torch.sub(torch.ones_like(maskF), maskF)));
        }
        Tensor weights = scores.softmax(1L); // [B, L, 1]
        return x.mul(weights).sum(1L);       // [B, D]
    }

    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }
}
