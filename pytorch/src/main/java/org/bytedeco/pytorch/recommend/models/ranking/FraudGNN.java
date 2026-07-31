/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/GNN.scala (FraudGNN)
 *
 * FraudGNN: multi-layer GCN stack for fraud detection.
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FraudGNN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final List<GraphConvolution> layers = new ArrayList<>();
    private final DropoutImpl dropoutLayer;
    private final ReLUImpl activation;
    private final int numLayers;

    public FraudGNN(int numFeatures) {
        this(numFeatures, 128, 2, 3, 0.3f, DeviceSupport.backend());
    }

    public FraudGNN(int numFeatures, int hiddenDim, int numClasses, int numLayers,
                    float dropout, String device) {
        super("FraudGNN");
        this.numLayers = numLayers;

        for (int i = 0; i < numLayers; i++) {
            int inDim = (i == 0) ? numFeatures : hiddenDim;
            int outDim = (i == numLayers - 1) ? numClasses : hiddenDim;
            GraphConvolution layer = new GraphConvolution(inDim, outDim, device);
            register_module("gc_" + i, layer);
            layers.add(layer);
        }

        this.dropoutLayer = new DropoutImpl(dropout);
        this.activation = new ReLUImpl();

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            for (GraphConvolution layer : layers) {
                layer.to(dev, false);
            }
        }
    }

    public Tensor forward(Tensor features, Tensor adj) {
        Tensor x = features;
        for (int i = 0; i < numLayers - 1; i++) {
            x = layers.get(i).forward(x, adj);
            x = activation.forward(x);
            x = dropoutLayer.forward(x);
        }
        return layers.get(numLayers - 1).forward(x, adj);
    }
}
