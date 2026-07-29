/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/DKVMN.scala (DeepIRT)
 *
 * DeepIRT: Deep Item Response Theory — DKVMN memory + IRT prediction.
 * P(correct) = sigmoid(3 * ability - difficulty)
 * Reference: "Deep-IRT: Making Deep Knowledge Tracing Explainable"
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DeepIRT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int memDim;
    private final int memSize;
    private final EmbeddingImpl kEmb;
    private final EmbeddingImpl vEmb;
    private final Tensor mkInit;
    private final Tensor mvInit;
    private final LinearImpl eLayer;
    private final LinearImpl aLayer;
    private final LinearImpl fLayer;
    private final LinearImpl diffLayer;
    private final LinearImpl abilityLayer;
    private final DropoutImpl dropoutLayer;

    public DeepIRT(long numConcepts, long numQuestions) {
        this(numConcepts, numQuestions, 64, 20, 0.2f, DeviceSupport.backend());
    }

    public DeepIRT(
            long numConcepts,
            long numQuestions,
            int memDim,
            int memSize,
            float dropout,
            String device) {
        super("DeepIRT");
        this.memDim = memDim;
        this.memSize = memSize;

        this.kEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, memDim));
        this.vEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 2, memDim));
        register_module("k_emb", kEmb);
        register_module("v_emb", vEmb);

        Tensor mk = torch.randn(
                new long[]{memSize, memDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar((float) Math.sqrt(1.0 / memDim)));
        this.mkInit = mk;
        register_parameter("mk", mkInit);

        Tensor mv = torch.zeros(
                new long[]{memSize, memDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        this.mvInit = mv;
        register_parameter("mv", mvInit);

        this.eLayer = new LinearImpl(memDim, 1);
        this.aLayer = new LinearImpl(memDim, 1);
        this.fLayer = new LinearImpl(memDim * 2L, memDim);
        register_module("e_layer", eLayer);
        register_module("a_layer", aLayer);
        register_module("f_layer", fLayer);

        this.diffLayer = new LinearImpl(memDim, 1);
        this.abilityLayer = new LinearImpl(memDim, 1);
        register_module("diff_layer", diffLayer);
        register_module("ability_layer", abilityLayer);

        this.dropoutLayer = new DropoutImpl(dropout);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            this.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor k = kEmb.forward(conceptIds);
        Tensor interactionIds = conceptIds.mul(new Scalar(2.0)).add(responses).toType(ScalarType.Long);
        Tensor v = vEmb.forward(interactionIds);

        Tensor mkBatch = mkInit.clone()
                .view(1L, (long) memSize, (long) memDim)
                .expand(batchSize, -1L, -1L)
                .detach();
        Tensor mv = mvInit.clone()
                .view(1L, (long) memSize, (long) memDim)
                .expand(batchSize, -1L, -1L)
                .clone();

        List<Tensor> preds = new ArrayList<>();
        for (int t = 0; t < seqLen; t++) {
            Tensor kt = k.select(1, t);
            Tensor vt = v.select(1, t);

            Tensor mvCurrent = t == 0 ? mv : mv.detach();

            Tensor scores = torch.bmm(kt.unsqueeze(1), mkBatch.transpose(1, 2)).squeeze(1);
            Tensor attn = scores.softmax(1);

            Tensor eraseGate = torch.sigmoid(eLayer.forward(kt)).squeeze(1);
            Tensor attnEx = attn.unsqueeze(2);
            Tensor eraseGateEx = eraseGate.unsqueeze(1).unsqueeze(2);
            Tensor attnScaled = attnEx.mul(eraseGateEx);
            Tensor eraseFactor = attnScaled.neg().add(new Scalar(1.0)).contiguous();
            Tensor newMv = mvCurrent.mul(eraseFactor);

            Tensor addGate = torch.tanh(aLayer.forward(kt)).squeeze(1);
            Tensor addGateEx = addGate.unsqueeze(1).unsqueeze(2);
            Tensor addScaled = attnEx.mul(addGateEx);
            mv = newMv.add(addScaled);

            Tensor readMem = attn.unsqueeze(2).mul(mv).sum(1);
            Tensor combined = torch.cat(new TensorVector(readMem, kt), 1);
            Tensor fused = torch.tanh(fLayer.forward(combined));

            // IRT: P(correct) = sigmoid(3 * ability - difficulty)
            Tensor ability = torch.tanh(abilityLayer.forward(dropoutLayer.forward(fused)));
            Tensor difficulty = torch.tanh(diffLayer.forward(dropoutLayer.forward(kt)));
            Tensor pred = torch.sigmoid(ability.mul(new Scalar(3.0)).sub(difficulty));
            preds.add(pred.squeeze(1));
        }

        if (preds.isEmpty()) {
            return torch.zeros(batchSize, seqLen);
        }
        return torch.stack(new TensorVector(preds.toArray(new Tensor[0])), 1);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
