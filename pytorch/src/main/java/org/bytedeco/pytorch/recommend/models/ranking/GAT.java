/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/GNN.scala
 * (GAT, SAGEAggregator, GraphSAGE, FraudGNN)
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * Graph Attention Network (GAT).
 * Reference: "Graph Attention Networks" (ICLR 2018)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GAT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final GraphAttentionLayer att1;
    private final GraphAttentionLayer att2;

    public GAT(int numFeatures) {
        this(numFeatures, 64, 2, 8, 0.5f, DeviceSupport.backend());
    }

    public GAT(int numFeatures, int hiddenDim, int numClasses, int numHeads,
               float dropout, String device) {
        super("GAT");
        this.att1 = new GraphAttentionLayer(numFeatures, hiddenDim, numHeads, dropout, device);
        this.att2 = new GraphAttentionLayer(hiddenDim, numClasses, 1, dropout, device);
        register_module("att1", att1);
        register_module("att2", att2);
    }

    public Tensor forward(Tensor features, Tensor adj) {
        Tensor x = att1.forward(features, adj);
        Tensor xActivated = torch.elu(x);
        return att2.forward(xActivated, adj);
    }
}
