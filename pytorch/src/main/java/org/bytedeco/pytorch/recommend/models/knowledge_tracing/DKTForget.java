/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/DKTForget.scala
 *
 * DKT-Forget: Deep Knowledge Tracing with Forgetting Dynamics.
 * Interaction Embedding + Time Gap Integration → LSTM → Output.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorT_TensorTensor_T_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LSTMImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.InteractionEmbedding;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DKTForget extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String device;
    private final InteractionEmbedding interactionEmb;
    private final LinearImpl timeEmbed;
    private final LSTMImpl lstm;
    private final DropoutImpl dropoutLayer;
    private final LinearImpl outputLayer;

    public DKTForget(long numConcepts) {
        this(numConcepts, 64, 1, 0.2f, DeviceSupport.backend());
    }

    public DKTForget(long numConcepts, int embedDim, int numLayers, float dropout, String device) {
        super("DKTForget");
        this.device = device;

        this.interactionEmb = new InteractionEmbedding(numConcepts, embedDim, device);
        register_module("interaction_emb", interactionEmb);

        this.timeEmbed = new LinearImpl(embedDim + 3L, embedDim);
        register_module("time_embed", timeEmbed);

        LSTMOptions lstmOptions = new LSTMOptions(embedDim, embedDim);
        lstmOptions.num_layers().put(numLayers);
        lstmOptions.dropout().put((double) dropout);
        lstmOptions.batch_first().put(true);
        this.lstm = new LSTMImpl(lstmOptions);
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

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor interactionEmbOut = interactionEmb.forward(conceptIds, responses);

        // Simplified time gap features (fixed ones for benchmark, matches Scala)
        Tensor timeFeatures = torch.ones(
                new long[]{batchSize, seqLen, 3L},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (!"cpu".equals(device)) {
            timeFeatures = timeFeatures.to(new Device(device), ScalarType.Float);
        }

        Tensor combined = torch.cat(new TensorVector(interactionEmbOut, timeFeatures), 2);
        Tensor timeIntegrated = timeEmbed.forward(combined);
        Tensor activated = torch.tanh(timeIntegrated);

        T_TensorT_TensorTensor_T_T lstmRet = lstm.forwardT_TensorT_TensorTensor_T_T(activated);
        Tensor lstmOut = lstmRet.get0();
        Tensor dropped = dropoutLayer.forward(lstmOut);
        return outputLayer.forward(dropped);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses).sigmoid();
    }
}
