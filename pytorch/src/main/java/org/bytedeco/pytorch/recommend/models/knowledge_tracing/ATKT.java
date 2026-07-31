/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/ATKT.scala
 *
 * ATKT: Attention-based Knowledge Tracing with skill-answer interleaving
 * and cumulative attention. Also includes ATKTFix variant.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ATKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final EmbeddingImpl skillEmb;
    private final EmbeddingImpl answerEmb;
    private final LinearImpl inputProj;
    private final LinearImpl attnMlp;
    private final LinearImpl attnSim;
    private final LinearImpl fc;
    private final DropoutImpl dropoutLayer;

    public ATKT(long numConcepts) {
        this(numConcepts, 64, 64, 64, 80, 0.2f, true, DeviceSupport.backend());
    }

    public ATKT(
            long numConcepts,
            int skillDim,
            int answerDim,
            int hiddenDim,
            int attentionDim,
            float dropout,
            boolean fix,
            String device) {
        super("ATKT");
        this.skillEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, skillDim));
        register_module("skill_emb", skillEmb);

        this.answerEmb = new EmbeddingImpl(new EmbeddingOptions(3, answerDim));
        register_module("answer_emb", answerEmb);

        this.inputProj = new LinearImpl(skillDim + answerDim, hiddenDim);
        register_module("input_proj", inputProj);

        this.attnMlp = new LinearImpl(hiddenDim, attentionDim);
        register_module("attn_mlp", attnMlp);

        this.attnSim = new LinearImpl(attentionDim, 1);
        register_module("attn_sim", attnSim);

        this.fc = new LinearImpl(hiddenDim * 2L, numConcepts);
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

        // Interleaving mask based on response correctness
        Tensor isWrong = answerIds.ne(new Scalar(1.0)).toType(ScalarType.Float);
        Tensor isCorrect = answerIds.eq(new Scalar(1.0)).toType(ScalarType.Float);

        Tensor part1 = sEmb.mul(isWrong.unsqueeze(2)).add(aEmb.mul(isCorrect.unsqueeze(2)));
        Tensor part2 = aEmb.mul(isWrong.unsqueeze(2)).add(sEmb.mul(isCorrect.unsqueeze(2)));
        Tensor combined = torch.cat(new TensorVector(part1, part2), 2);

        Tensor hidden = torch.relu(inputProj.forward(combined));

        Tensor attnInput = torch.tanh(attnMlp.forward(hidden));
        Tensor cumsumAttn = attnInput.cumsum(1);
        Tensor attnScore = cumsumAttn.sub(attnInput);
        Tensor attnWeights = torch.sigmoid(attnSim.forward(attnScore));

        Tensor attended = hidden.mul(attnWeights);
        Tensor combinedOut = torch.cat(new TensorVector(attended, hidden), 2);
        Tensor out = fc.forward(dropoutLayer.forward(combinedOut));
        return out.sigmoid();
    }

    public Tensor predict(Tensor skillIds, Tensor answerIds) {
        return forward(skillIds, answerIds);
    }
}
