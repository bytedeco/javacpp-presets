/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/ATDKT.scala
 *
 * AT-DKT: Attention-based Deep Knowledge Tracing.
 * Interaction Embedding → LSTM → Self-Attention → MLP → sigmoid.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LSTMImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.InteractionEmbedding;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ATDKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final InteractionEmbedding interactionEmb;
    private final LSTMImpl lstm;
    private final SelfAttentionLayer selfAttn;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl fc1;
    private final LinearImpl fc2;
    private final LinearImpl outputLayer;

    public ATDKT(long numConcepts) {
        this(numConcepts, 64, 1, 4, 0.2f, DeviceSupport.backend());
    }

    public ATDKT(
            long numConcepts,
            int embedDim,
            int numLayers,
            int numHeads,
            float dropout,
            String device) {
        super("ATDKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        this.interactionEmb = new InteractionEmbedding(numConcepts, embedDim, device);
        register_module("interaction_emb", interactionEmb);

        LSTMOptions lstmOptions = new LSTMOptions(embedDim, embedDim);
        lstmOptions.num_layers().put(numLayers);
        lstmOptions.dropout().put((double) dropout);
        lstmOptions.batch_first().put(true);
        this.lstm = new LSTMImpl(lstmOptions);
        register_module("lstm", lstm);

        this.selfAttn = new SelfAttentionLayer(embedDim, numHeads, dropout, device);
        register_module("self_attn", selfAttn);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.fc1 = new LinearImpl(embedDim, embedDim);
        register_module("fc1", fc1);
        this.fc2 = new LinearImpl(embedDim, embedDim / 2L);
        register_module("fc2", fc2);
        this.outputLayer = new LinearImpl(embedDim / 2L, 1);
        register_module("output", outputLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            lstm.to(dev, false);
            fc1.to(dev, false);
            fc2.to(dev, false);
            outputLayer.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor interactionEmbOut = interactionEmb.forward(conceptIds, responses);
        T_TensorT_TensorTensor_T_T lstmRet = lstm.forwardT_TensorT_TensorTensor_T_T(interactionEmbOut);
        Tensor lstmOut = lstmRet.get0();
        Tensor attnOut = selfAttn.forward(lstmOut);
        Tensor combined = lstmOut.add(attnOut);
        Tensor dropped = dropoutLayer.forward(combined);
        Tensor h1 = torch.relu(fc1.forward(dropped));
        Tensor h2 = torch.relu(fc2.forward(h1));
        Tensor logits = outputLayer.forward(h2);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
