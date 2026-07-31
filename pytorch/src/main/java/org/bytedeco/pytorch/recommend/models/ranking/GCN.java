/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/GNN.scala
 * (GCN, GAT, GraphAttentionLayer, GraphSAGE, SAGEAggregator, FraudGNN)
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

/**
 * Graph Convolutional Network (GCN).
 * Reference: "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GCN extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final GraphConvolution gc1;
    private final GraphConvolution gc2;
    private final DropoutImpl dropoutLayer;
    private final ReLUImpl activation;

    public GCN(int numFeatures) {
        this(numFeatures, 64, 2, 0.5f, DeviceSupport.backend());
    }

    public GCN(int numFeatures, int hiddenDim, int numClasses, float dropout, String device) {
        super("GCN");
        this.gc1 = new GraphConvolution(numFeatures, hiddenDim, device);
        this.gc2 = new GraphConvolution(hiddenDim, numClasses, device);
        this.dropoutLayer = new DropoutImpl(dropout);
        this.activation = new ReLUImpl();

        register_module("gc1", gc1);
        register_module("gc2", gc2);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            gc1.to(dev, false);
            gc2.to(dev, false);
        }
    }

    public Tensor forward(Tensor features, Tensor adj) {
        Tensor x = gc1.forward(features, adj);
        x = activation.forward(x);
        x = dropoutLayer.forward(x);
        return gc2.forward(x, adj);
    }
}
