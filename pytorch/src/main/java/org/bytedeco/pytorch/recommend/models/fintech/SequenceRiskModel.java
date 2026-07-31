/*
 * SequenceRiskModel — sequential behavior risk / fraud model for fintech.
 *
 * Production context:
 *   Card / account fraud and credit-line abuse detection use event sequences
 *   (transactions, logins, device changes) with a classifier on top of a
 *   sequential encoder. Common industrial stacks:
 *     - GRU / LSTM over transaction embeddings (classic)
 *     - Transformer / SASRec-style self-attention over events
 *     - Graph features (device-account bipartite) fed as side information
 *       (see existing FraudGNN in ranking package)
 *
 * This model:
 *   event ids -> embedding (+ optional amount / time-delta dense projection)
 *   -> Multi-Head Self-Attention stack (causal optional)
 *   -> last / attentive pool -> MLP risk score
 *
 * References (representative, not a single paper clone):
 *   - SEQ-Fraud style sequential models in IEEE S&P / KDD fraud workshops
 *   - Alipay / Ant Group sequential risk papers (e.g. spatiotemporal
 *     attention for fraud, CIKM/KDD industrial tracks)
 */
package org.bytedeco.pytorch.recommend.models.fintech;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.basic.layers.industry.AdditiveAttention;
import org.bytedeco.pytorch.recommend.basic.layers.industry.MultiHeadSelfAttention;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SequenceRiskModel extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl eventEmbedding;
    private final LinearImpl amountProj;     // optional continuous side feature per step
    private final LinearImpl timeDeltaProj;  // optional
    private final List<MultiHeadSelfAttention> layers = new ArrayList<>();
    private final List<LayerNormImpl> norms = new ArrayList<>();
    private final AdditiveAttention pool;
    private final MLP riskHead;
    private final int embedDim;
    private final boolean useAmount;
    private final boolean useTimeDelta;

    public SequenceRiskModel(int eventVocabSize) {
        this(eventVocabSize, 64, 4, 2, true, true, new long[]{128L, 64L},
                0.1f, DeviceSupport.backend());
    }

    public SequenceRiskModel(int eventVocabSize, int embedDim, int numHeads, int numLayers,
                             boolean useAmount, boolean useTimeDelta, long[] headHidden,
                             float dropout, String device) {
        super("SequenceRiskModel");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.useAmount = useAmount;
        this.useTimeDelta = useTimeDelta;

        EmbeddingOptions opts = new EmbeddingOptions(Math.max(eventVocabSize, 2), embedDim);
        opts.padding_idx().put(new LongOptional(0L));
        this.eventEmbedding = new EmbeddingImpl(opts);
        register_module("event_embedding", eventEmbedding);

        if (useAmount) {
            this.amountProj = new LinearImpl(1L, embedDim);
            register_module("amount_proj", amountProj);
        } else {
            this.amountProj = null;
        }
        if (useTimeDelta) {
            this.timeDeltaProj = new LinearImpl(1L, embedDim);
            register_module("time_delta_proj", timeDeltaProj);
        } else {
            this.timeDeltaProj = null;
        }

        for (int i = 0; i < numLayers; i++) {
            MultiHeadSelfAttention attn = new MultiHeadSelfAttention(embedDim, numHeads, dropout, device);
            LongVector shape = new LongVector(1);
            shape.put(0, embedDim);
            LayerNormImpl ln = new LayerNormImpl(shape);
            register_module("attn_" + i, attn);
            register_module("norm_" + i, ln);
            layers.add(attn);
            norms.add(ln);
        }

        this.pool = new AdditiveAttention(embedDim, embedDim, device);
        register_module("pool", pool);

        this.riskHead = new MLP(embedDim, headHidden, 1L, "relu", dropout, false, false, true, device);
        register_module("risk_head", riskHead);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            eventEmbedding.to(dev, false);
            if (amountProj != null) amountProj.to(dev, false);
            if (timeDeltaProj != null) timeDeltaProj.to(dev, false);
            for (LayerNormImpl n : norms) n.to(dev, false);
        }
    }

    /**
     * @param eventIds   [B, L] long (0=pad)
     * @param amounts    [B, L] float log1p(amount) or null
     * @param timeDeltas [B, L] float seconds since previous event (normalized) or null
     * @return risk probability [B]
     */
    public Tensor forward(Tensor eventIds, Tensor amounts, Tensor timeDeltas) {
        Tensor mask = eventIds.ne(new Scalar(0L)).toType(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor h = eventEmbedding.forward(eventIds.toType(
                org.bytedeco.pytorch.global.torch.ScalarType.Long));
        if (useAmount && amounts != null && amountProj != null) {
            // amounts [B,L] -> [B,L,1] -> Linear(1,D) -> [B,L,D]
            h = h.add(amountProj.forward(amounts.unsqueeze(2L)));
        }
        if (useTimeDelta && timeDeltas != null && timeDeltaProj != null) {
            h = h.add(timeDeltaProj.forward(timeDeltas.unsqueeze(2L)));
        }

        for (int i = 0; i < layers.size(); i++) {
            Tensor n = norms.get(i).forward(h);
            h = h.add(layers.get(i).forward(n, mask));
        }
        Tensor userRiskEmb = pool.forward(h, mask);
        return riskHead.forward(userRiskEmb).squeeze(1L).sigmoid();
    }

    public Tensor forward(Tensor eventIds) {
        return forward(eventIds, (Tensor) null, (Tensor) null);
    }
}
