/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/ATKT.scala (ATKTFix)
 *
 * ATKTFix: ATKT with fixed causal attention (simple concat, no interleaving).
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ATKTFix extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl skillEmb;
    private final EmbeddingImpl answerEmb;
    private final LinearImpl inputProj;
    private final LinearImpl attnMlp;
    private final LinearImpl fc;
    private final DropoutImpl dropoutLayer;

    public ATKTFix(long numConcepts) {
        this(numConcepts, 64, 64, 64, 80, 0.2f, DeviceSupport.backend());
    }

    public ATKTFix(
            long numConcepts,
            int skillDim,
            int answerDim,
            int hiddenDim,
            int attentionDim,
            float dropout,
            String device) {
        super("ATKTFix");
        this.skillEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, skillDim));
        this.answerEmb = new EmbeddingImpl(new EmbeddingOptions(3, answerDim));
        register_module("skill_emb", skillEmb);
        register_module("answer_emb", answerEmb);

        this.inputProj = new LinearImpl(skillDim + answerDim, hiddenDim);
        register_module("input_proj", inputProj);

        this.attnMlp = new LinearImpl(hiddenDim, attentionDim);
        this.fc = new LinearImpl(hiddenDim * 2L, numConcepts);
        register_module("attn_mlp", attnMlp);
        register_module("fc", fc);

        this.dropoutLayer = new DropoutImpl(dropout);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            skillEmb.to(dev, false);
            answerEmb.to(dev, false);
            fc.to(dev, false);
        }
    }

    public Tensor forward(Tensor skillIds, Tensor answerIds) {
        Tensor sEmb = skillEmb.forward(skillIds);
        Tensor aEmb = answerEmb.forward(answerIds);

        Tensor combined = torch.cat(new TensorVector(sEmb, aEmb), 2);
        Tensor hidden = torch.relu(inputProj.forward(combined));

        // attnMlp: hiddenDim → attentionDim; reduce to per-step scalar gate so it
        // broadcasts against hidden [B, T, H] (was [B,T,attentionDim] and crashed).
        Tensor attnInput = torch.tanh(attnMlp.forward(hidden)); // [B, T, A]
        Tensor attnScore = attnInput.mean(-1).unsqueeze(-1);    // [B, T, 1]
        // causal-ish cumulative attention mass
        Tensor cumsumAttn = attnScore.cumsum(1);
        Tensor gate = cumsumAttn.sub(attnScore);                // [B, T, 1]

        Tensor attended = hidden.mul(gate);
        Tensor combinedOut = torch.cat(new TensorVector(attended, hidden), 2);
        Tensor out = fc.forward(dropoutLayer.forward(combinedOut));
        return out.sigmoid();
    }

    public Tensor predict(Tensor skillIds, Tensor answerIds) {
        return forward(skillIds, answerIds);
    }
}
