/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/DKT.scala
 *
 * DKT: Deep Knowledge Tracing (Piech et al., NeurIPS 2015).
 * Architecture: Interaction Embedding (concept + response) → LSTM → Output Layer
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LSTMImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.InteractionEmbedding;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final InteractionEmbedding interactionEmb;
    private final LSTMImpl lstm;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl outputLayer;

    public DKT(long numConcepts) {
        this(numConcepts, 64, 1, 0.2f, DeviceSupport.backend());
    }

    public DKT(long numConcepts, int embedDim, int numLayers, float dropout, String device) {
        super("DKT");
        this.interactionEmb = new InteractionEmbedding(numConcepts, embedDim, device);
        register_module("interaction_emb", interactionEmb);

        LSTMOptions opt = new LSTMOptions(embedDim, embedDim);
        opt.num_layers().put(numLayers);
        opt.dropout().put((double) dropout);
        opt.batch_first().put(true);
        this.lstm = new LSTMImpl(opt);
        register_module("lstm", lstm);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.outputLayer = new LinearImpl(embedDim, numConcepts);
        register_module("output", outputLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            lstm.to(dev, false);
            outputLayer.to(dev, false);
        }
    }

    /**
     * @param conceptIds (batch, seqLen)
     * @param responses  (batch, seqLen) 0/1
     * @return logits (batch, seqLen, numConcepts)
     */
    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor interactionEmbOut = interactionEmb.forward(conceptIds, responses);
        T_TensorT_TensorTensor_T_T lstmRet = lstm.forwardT_TensorT_TensorTensor_T_T(interactionEmbOut);
        Tensor lstmOut = lstmRet.get0();
        Tensor dropped = dropoutLayer.forward(lstmOut);
        return outputLayer.forward(dropped);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses).sigmoid();
    }
}
