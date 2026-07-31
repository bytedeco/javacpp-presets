/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/GNN.scala
 * (GraphSAGE, FraudGNN)
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
 * GraphSAGE: Graph SAmpling and Aggregation.
 * Reference: "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GraphSAGE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final SAGEAggregator agg1;
    private final SAGEAggregator agg2;
    private final DropoutImpl dropoutLayer;
    private final ReLUImpl activation;

    public GraphSAGE(int numFeatures) {
        this(numFeatures, 64, 2, "mean", 0.5f, DeviceSupport.backend());
    }

    public GraphSAGE(int numFeatures, int hiddenDim, int numClasses, String aggregator,
                     float dropout, String device) {
        super("GraphSAGE");
        // Scala maps mean/pool/lstm all to SAGEAggregator currently
        this.agg1 = new SAGEAggregator(numFeatures, hiddenDim, device);
        this.agg2 = new SAGEAggregator(hiddenDim, numClasses, device);
        register_module("agg1", agg1);
        register_module("agg2", agg2);

        this.dropoutLayer = new DropoutImpl(dropout);
        this.activation = new ReLUImpl();

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            agg1.to(dev, false);
            agg2.to(dev, false);
        }
    }

    public Tensor forward(Tensor features, Tensor adj) {
        Tensor x = agg1.forward(features, adj);
        x = activation.forward(x);
        x = dropoutLayer.forward(x);
        return agg2.forward(x, adj);
    }
}
