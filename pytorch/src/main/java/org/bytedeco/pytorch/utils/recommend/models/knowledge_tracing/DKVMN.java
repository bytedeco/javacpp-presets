/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/DKVMN.scala
 *
 * DKVMN: Dynamic Key-Value Memory Networks (Zhang et al., 2017).
 * Key memory (static concepts) + value memory (dynamic student state) with erase-and-add write.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
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
public class DKVMN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int memDim;
    private final int memSize;
    private final EmbeddingImpl kEmb;
    private final EmbeddingImpl vEmb;
    private final Tensor mkInit;
    private final Tensor mvInit;
    private final LinearImpl eLayer;
    private final LinearImpl aLayer;
    private final LinearImpl fLayer;
    private final LinearImpl pLayer;
    private final DropoutImpl dropoutLayer;

    public DKVMN(long numConcepts, long numQuestions) {
        this(numConcepts, numQuestions, 64, 20, 0.2f, DeviceSupport.backend());
    }

    public DKVMN(
            long numConcepts,
            long numQuestions,
            int memDim,
            int memSize,
            float dropout,
            String device) {
        super("DKVMN");
        this.numConcepts = numConcepts;
        this.memDim = memDim;
        this.memSize = memSize;

        this.kEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, memDim));
        register_module("k_emb", kEmb);

        this.vEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 1, memDim));
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
        register_module("e_layer", eLayer);
        this.aLayer = new LinearImpl(memDim, 1);
        register_module("a_layer", aLayer);
        this.fLayer = new LinearImpl(memDim * 2L, memDim);
        register_module("f_layer", fLayer);
        this.pLayer = new LinearImpl(memDim, 1);
        register_module("p_layer", pLayer);
        this.dropoutLayer = new DropoutImpl(dropout);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            this.to(dev, false);
        }
    }

    /**
     * @param conceptIds (batch, seqLen)
     * @param responses  (batch, seqLen) 0/1
     * @return predictions (batch, seqLen)
     */
    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor conceptIdsLong = conceptIds.toType(ScalarType.Long);
        Tensor responsesLong = responses.toType(ScalarType.Long);

        Tensor kIdsClamped = conceptIdsLong.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(numConcepts)))
                .toType(ScalarType.Long);
        Tensor k = kEmb.forward(kIdsClamped);

        Tensor interactionIdsRaw = conceptIdsLong.add(responsesLong.mul(new Scalar((double) numConcepts)));
        Tensor interactionIds = interactionIdsRaw.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(numConcepts * 2)))
                .toType(ScalarType.Long);
        Tensor v = vEmb.forward(interactionIds);

        Tensor prevMv = mvInit.clone()
                .view(1L, (long) memSize, (long) memDim)
                .expand(batchSize, -1L, -1L)
                .clone();

        List<Tensor> preds = new ArrayList<>();
        for (int t = 0; t < seqLen; t++) {
            Tensor kt = k.select(1, t);
            Tensor vt = v.select(1, t);

            Tensor mkT = mkInit.clone()
                    .view(1L, (long) memSize, (long) memDim)
                    .expand(batchSize, -1L, -1L)
                    .detach();
            Tensor scores = torch.bmm(kt.unsqueeze(1), mkT.transpose(1, 2)).squeeze(1);
            Tensor attn = scores.softmax(1);

            Tensor eraseGate = torch.sigmoid(eLayer.forward(kt)).squeeze(1);
            Tensor eraseGateEx = eraseGate.unsqueeze(1).unsqueeze(2);
            Tensor attnEx = attn.unsqueeze(2);
            Tensor newMv = prevMv.mul(attnEx.mul(eraseGateEx).neg().add(new Scalar(1.0)));

            Tensor addGate = torch.tanh(aLayer.forward(kt)).squeeze(1);
            Tensor addGateEx = addGate.unsqueeze(1).unsqueeze(2);
            Tensor addedMv = newMv.add(attnEx.mul(addGateEx));
            prevMv = addedMv;

            Tensor readMem = attn.unsqueeze(2).mul(addedMv).sum(1);
            Tensor combined = torch.cat(new TensorVector(readMem, kt), 1);
            Tensor fused = torch.tanh(fLayer.forward(combined));

            Tensor pred = pLayer.forward(dropoutLayer.forward(fused));
            preds.add(pred.sigmoid().squeeze(1));
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
