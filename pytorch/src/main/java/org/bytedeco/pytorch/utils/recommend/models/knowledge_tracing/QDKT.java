/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/QDKT.scala
 *
 * QDKT: Question-specific Deep Knowledge Tracing.
 * Interaction Embedding (question * 2 + response) → LSTM → per-question output.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LSTMImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class QDKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numQuestions;
    private final EmbeddingImpl interactionEmb;
    private final LSTMImpl lstm;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl outputLayer;

    public QDKT(long numQuestions, long numConcepts) {
        this(numQuestions, numConcepts, 64, 1, 0.2f, DeviceSupport.backend());
    }

    public QDKT(
            long numQuestions,
            long numConcepts,
            int embedDim,
            int numLayers,
            float dropout,
            String device) {
        super("QDKT");
        this.numQuestions = numQuestions;

        this.interactionEmb = new EmbeddingImpl(new EmbeddingOptions(numQuestions * 2, embedDim));
        register_module("interaction_emb", interactionEmb);

        LSTMOptions lstmOptions = new LSTMOptions(embedDim, embedDim);
        lstmOptions.num_layers().put(numLayers);
        lstmOptions.dropout().put((double) dropout);
        lstmOptions.batch_first().put(true);
        this.lstm = new LSTMImpl(lstmOptions);
        register_module("lstm", lstm);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.outputLayer = new LinearImpl(embedDim, 1);
        register_module("output", outputLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            interactionEmb.to(dev, false);
            lstm.to(dev, false);
            outputLayer.to(dev, false);
        }
    }

    /**
     * @param questionIds (batch, seqLen)
     * @param conceptIds  (batch, seqLen) — kept for API parity
     * @param responses   (batch, seqLen) 0/1
     * @return predictions (batch, seqLen)
     */
    public Tensor forward(Tensor questionIds, Tensor conceptIds, Tensor responses) {
        Tensor qIdsLong = questionIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) (numQuestions * 2 - 1))));
        Tensor rLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)));

        Tensor interactionIds = qIdsLong.mul(new Scalar(2)).add(rLong);
        Tensor emb = interactionEmb.forward(interactionIds.toType(ScalarType.Long));

        T_TensorT_TensorTensor_T_T lstmRet = lstm.forwardT_TensorT_TensorTensor_T_T(emb);
        Tensor lstmOut = lstmRet.get0();
        Tensor dropped = dropoutLayer.forward(lstmOut);
        Tensor logits = outputLayer.forward(dropped);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor questionIds, Tensor conceptIds, Tensor responses) {
        return forward(questionIds, conceptIds, responses);
    }
}
