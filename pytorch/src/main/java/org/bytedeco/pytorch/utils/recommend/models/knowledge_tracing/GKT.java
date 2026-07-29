/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/GKT.scala
 *
 * GKT: Graph-based Knowledge Tracing (Simplified) — LSTM instead of full GNN.
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
public class GKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final EmbeddingImpl interactionEmb;
    private final EmbeddingImpl conceptEmb;
    private final LSTMImpl lstm;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl predictLayer;

    public GKT(long numConcepts) {
        this(numConcepts, 64, 64, 0.5f, "dense", DeviceSupport.backend());
    }

    public GKT(
            long numConcepts,
            int embedDim,
            int hiddenDim,
            float dropout,
            String graphType,
            String device) {
        super("GKT");
        this.numConcepts = numConcepts;

        this.interactionEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 1, embedDim));
        register_module("interaction_emb", interactionEmb);

        this.conceptEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("concept_emb", conceptEmb);

        LSTMOptions lstmOpts = new LSTMOptions(embedDim, hiddenDim);
        lstmOpts.num_layers().put(1);
        lstmOpts.dropout().put((double) dropout);
        lstmOpts.batch_first().put(true);
        this.lstm = new LSTMImpl(lstmOpts);
        register_module("lstm", lstm);

        this.dropoutLayer = new DropoutImpl(dropout);
        this.predictLayer = new LinearImpl(hiddenDim, 1);
        register_module("predict", predictLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            this.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor conceptIdsLong = conceptIds.toType(ScalarType.Long);
        Tensor responsesLong = responses.toType(ScalarType.Long);

        Tensor conceptIdsClamped = conceptIdsLong.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(numConcepts)))
                .toType(ScalarType.Long);

        Tensor interactionIdsRaw = conceptIdsClamped.add(responsesLong.mul(new Scalar((double) numConcepts)));
        Tensor interactionIds = interactionIdsRaw.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(numConcepts * 2)))
                .toType(ScalarType.Long);

        Tensor xEmb = interactionEmb.forward(interactionIds);
        Tensor cEmb = conceptEmb.forward(conceptIdsClamped);
        Tensor combinedEmb = xEmb.add(cEmb);

        T_TensorT_TensorTensor_T_T lstmRet = lstm.forwardT_TensorT_TensorTensor_T_T(combinedEmb);
        Tensor lstmOut = lstmRet.get0();
        Tensor dropped = dropoutLayer.forward(lstmOut);
        Tensor logits = predictLayer.forward(dropped);
        return logits.sigmoid();
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
