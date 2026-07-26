package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
public class ARLinkPredictor extends Module {
    private EmbeddingImpl nodeEmb;
    private double margin;

    public ARLinkPredictor(long numNodes, long dim, double margin) {
        this.margin = margin;
        this.nodeEmb = new EmbeddingImpl(numNodes, dim);
        register_module("nodeEmb", nodeEmb);
    }

    public Tensor forward(Tensor x) {
        return nodeEmb.forward(x);
    }

    /**
     * Attract-Repel Loss
     * @param edge_index Positive edges [2, E]
     * @param negedge_index Negative edges [2, E]
     */
    public Tensor loss(Tensor edge_index, Tensor negedge_index) {
        Tensor z = nodeEmb.weight();

        // 1. Attraction Loss: ||z_u - z_v||^2
        Tensor u = z.index_select(0, edge_index.select(0, 0));
        Tensor v = z.index_select(0, edge_index.select(0, 1));
        Tensor distPos = u.sub(v).pow(new Scalar(2)).sum(new long[]{1}, false,new ScalarTypeOptional());
        Tensor attractLoss = distPos.mean();

        // 2. Repulsion Loss: max(0, margin - ||z_u - z_v||^2)
        Tensor uNeg = z.index_select(0, negedge_index.select(0, 0));
        Tensor vNeg = z.index_select(0, negedge_index.select(0, 1));
        Tensor distNeg = uNeg.sub(vNeg).pow(new Scalar(2)).sum(new long[]{1}, false,new ScalarTypeOptional());
        Tensor repelLoss = torch.relu(torch.tensor(new Scalar(margin)).sub(distNeg)).mean();

        return attractLoss.add(repelLoss);
    }
}