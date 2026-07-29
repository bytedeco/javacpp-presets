/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/SKVMN.scala
 *
 * SKVMN: Student-friendly Key-Value Memory Network.
 * Attention-weighted memory read/write + LSTM over fused states → final prediction.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LSTMImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SKVMN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final int memSize;
    private final EmbeddingImpl kEmb;
    private final EmbeddingImpl xEmb;
    private final Tensor Mk;
    private final Tensor Mv0;
    private final LinearImpl aEmbed;
    private final LinearImpl fLayer;
    private final LSTMImpl lstm;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl pLayer;

    public SKVMN(long numConcepts) {
        this(numConcepts, 64, 20, 0.2f, DeviceSupport.backend());
    }

    public SKVMN(long numConcepts, int embedDim, int memSize, float dropout, String device) {
        super("SKVMN");
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;
        this.memSize = memSize;

        this.kEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("k_emb", kEmb);

        this.xEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 2, embedDim));
        register_module("x_emb", xEmb);

        Tensor mkInit = torch.randn(
                new long[]{memSize, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar((float) Math.sqrt(1.0 / embedDim)));
        this.Mk = mkInit;
        register_buffer("Mk", Mk);

        Tensor mv0Init = torch.randn(
                new long[]{memSize, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        this.Mv0 = mv0Init;
        register_buffer("Mv0", Mv0);

        this.aEmbed = new LinearImpl(embedDim * 2L, embedDim);
        register_module("a_embed", aEmbed);

        this.fLayer = new LinearImpl(embedDim * 2L, embedDim);
        register_module("f_layer", fLayer);

        // Default LSTM: batch_first = false (matches Scala LSTMImpl(embedDim, embedDim))
        this.lstm = new LSTMImpl(embedDim, embedDim);
        register_module("lstm", lstm);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.pLayer = new LinearImpl(embedDim, 1);
        register_module("p_layer", pLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            kEmb.to(dev, false);
            xEmb.to(dev, false);
            aEmbed.to(dev, false);
            fLayer.to(dev, false);
            lstm.to(dev, false);
            pLayer.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor cLong = conceptIds.toType(ScalarType.Long);
        Tensor rLong = responses.toType(ScalarType.Long);

        Tensor conceptIdx = cLong.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)))
                .toType(ScalarType.Long);
        Tensor responseIdx = rLong.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)))
                .toType(ScalarType.Long);

        Tensor kEmbed = kEmb.forward(conceptIdx);
        Tensor interactionIds = conceptIdx.mul(new Scalar(2)).add(responseIdx);
        Tensor interactionEmbeds = xEmb.forward(interactionIds);

        Tensor memInit = Mv0.clone().unsqueeze(0);
        Tensor currentMem = memInit.repeat(batchSize, 1, 1);

        List<Tensor> fTList = new ArrayList<>();
        for (int i = 0; i < seqLen; i++) {
            Tensor q = kEmbed.select(1, i);

            Tensor mkT = Mk.t();
            Tensor scores = torch.mm(q, mkT);
            Tensor attention = scores.softmax(1);

            Tensor attnExp = attention.unsqueeze(2);
            Tensor readContent = currentMem.mul(attnExp).sum(1);

            Tensor concat = torch.cat(new TensorVector(readContent, q), 1);
            Tensor f = torch.tanh(fLayer.forward(concat));
            fTList.add(f);

            Tensor y = interactionEmbeds.select(1, i);
            Tensor writeInput = torch.cat(new TensorVector(f, y), 1);
            Tensor writeEmbed = aEmbed.forward(writeInput);

            Tensor eraseSignal = torch.sigmoid(writeEmbed);
            Tensor addSignal = torch.tanh(writeEmbed);

            Tensor eraseGate = eraseSignal.unsqueeze(1);
            Tensor addGate = addSignal.unsqueeze(1);

            Tensor eraseFactor = attnExp.mul(eraseGate).neg().add(new Scalar(1.0));
            currentMem = currentMem.mul(eraseFactor).add(attnExp.mul(addGate));
        }

        // Stack: (seqLen, batchSize, embedDim) — LSTM batch_first=false
        Tensor ft = torch.stack(new TensorVector(fTList.toArray(new Tensor[0])), 0);
        T_TensorT_TensorTensor_T_T lstmRet = lstm.forwardT_TensorT_TensorTensor_T_T(ft);
        Tensor lstmOut = lstmRet.get0();
        Tensor lastHidden = lstmOut.select(0, seqLen - 1);

        Tensor pred = pLayer.forward(dropoutLayer.forward(lastHidden));
        return pred.sigmoid();
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
