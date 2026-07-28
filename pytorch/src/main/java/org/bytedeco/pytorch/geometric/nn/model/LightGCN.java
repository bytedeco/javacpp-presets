package org.bytedeco.pytorch.geometric.nn.model;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

public class LightGCN extends Module {
    private EmbeddingImpl embedding;
    private int numLayers;

    public LightGCN(long numNodes, long embeddingDim, int numLayers) {
        this.numLayers = numLayers;
        this.embedding = new EmbeddingImpl(numNodes, embeddingDim);
        // Xavier Init is important for LightGCN
        torch.xavier_normal_(embedding.weight());
        register_module("embedding", embedding);
    }

    public Tensor forward(Tensor edge_index) {
        // 1. Initial Embedding E0
        Tensor x = embedding.weight();

        TensorVector allEmbs = new TensorVector();
        allEmbs.put(x);

        // 2. Compute Normalization Coefficients (D^-0.5 * D^-0.5)
        long numNodes = x.size(0);
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        Tensor deg = AggrUtils.compute_degree(row, numNodes);
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));

        // Norm = deg[row] * deg[col]
        Tensor norm = degInvSqrt.index_select(0, row).mul(degInvSqrt.index_select(0, col));

        // 3. Propagation Layers
        for (int i = 0; i < numLayers; i++) {
            // E(k+1) = D^-0.5 A D^-0.5 E(k)
            // Message: norm * x_j
            Tensor x_j = x.index_select(0, col);
            Tensor msg = x_j.mul(norm.unsqueeze(1));

            // Aggregate
            x = AggrUtils.scatter(msg, row, numNodes, "sum");
            allEmbs.put(x);
        }

        // 4. Combine (Weighted Sum, LightGCN uses 1/(K+1) usually or just Mean)
        Tensor stack = torch.stack(allEmbs, 1); // [N, K+1, D]
        return stack.mean(new long[]{1}, false, new ScalarTypeOptional());
    }
}