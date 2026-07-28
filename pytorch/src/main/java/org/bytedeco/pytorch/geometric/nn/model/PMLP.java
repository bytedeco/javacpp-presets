package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

public class PMLP extends Module {
    private int K; // Propagation steps
    private SequentialImpl mlp;

    public PMLP(long inChannels, long hiddenChannels, long outChannels, int K, int numLayers) {
        this.K = K;

        this.mlp = new SequentialImpl();
        this.mlp.push_back(new LinearImpl(inChannels, hiddenChannels));
        this.mlp.push_back(new ReLUImpl());
        for(int i=0; i<numLayers-2; i++){
            this.mlp.push_back(new LinearImpl(hiddenChannels, hiddenChannels));
            this.mlp.push_back(new ReLUImpl());
        }
        this.mlp.push_back(new LinearImpl(hiddenChannels, outChannels));

        register_module("mlp", mlp);
    }

    public Tensor forward(Tensor x, Tensor edge_index) {
        Tensor h = x;
        long numNodes = x.size(0);
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // 1. Propagation Phase (Simplified GCN propagation: Mean or Norm-sum)
        // A^K X
        for (int k = 0; k < K; k++) {
            // Simple Mean org.bytedeco.pytorch.geometric.aggr.Aggregation
            Tensor msg = h.index_select(0, col);
            h = AggrUtils.scatter(msg, row, numNodes, "mean");
        }

        // 2. Transformation Phase (MLP)
        return mlp.forward(h);
    }
}