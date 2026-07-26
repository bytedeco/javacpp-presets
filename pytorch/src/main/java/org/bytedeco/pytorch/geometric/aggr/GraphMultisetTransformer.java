package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * 11. org.bytedeco.pytorch.geometric.aggr.GraphMultisetTransformer (GMT) 的核心：PMA (Pooling by Multihead Attention)
 * 使用一组 Seed Vectors (可学习) 去 Query 图节点
 */
public class GraphMultisetTransformer extends Aggregation {
    private Tensor seed; // Seed vectors [k, dim]
    private LinearImpl linQ, linK, linV, linOut;
    private long numHeads;
    private long dim;

    public GraphMultisetTransformer(long inChannels, long outChannels, long numHeads, long numSeeds) {
        this.numHeads = numHeads;
        this.dim = outChannels;

        // Seed Vectors S: [K, F]
        this.seed = new Tensor(torch.randn(new long[]{numSeeds, outChannels}));
        register_parameter("seed", seed);

        // Attention Projections
        this.linQ = new LinearImpl(outChannels, outChannels); // Query (Seed)
        this.linK = new LinearImpl(inChannels, outChannels);  // Key (Node Feat)
        this.linV = new LinearImpl(inChannels, outChannels);  // Value (Node Feat)
        this.linOut = new LinearImpl(outChannels, outChannels);

        register_module("linQ", linQ);
        register_module("linK", linK);
        register_module("linV", linV);
        register_module("linOut", linOut);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // GMT 通常用于 Global Pooling (Graph Classification)
        // x: [N, In], index: [N] (Batch)

        // 1. Projections
        // Q = Seed [K, Out] -> 扩展到 [Batch, K, Out]
        long batchSize = dimSize;
        Tensor Q = seed.unsqueeze(0).expand(new long[]{batchSize, seed.size(0), seed.size(1)});

        // K, V from Graph Nodes x
        // 这是一个难点：X 是堆叠的 [N, In]，需要根据 index 变成 [Batch, MaxNodes, In] 并 mask
        // 或者使用 org.bytedeco.pytorch.geometric.utils.Scatter Attention。

        // 为了简化，我们使用 "Global Attention Pooling" 的思路 (Gate)
        // Softmax(Q * K^T) * V
        // 这里仅实现最简单的单 Seed (k=1) 版本，即 GlobalAttention

        Tensor gate = linK.forward(x); // [N, Out]
        Tensor feat = linV.forward(x); // [N, Out]

        // Attention Score: gate * seed
        // 假设 seed 只有一个 (Global Context)
        Tensor seedVec = seed.mean(new long[]{0}, false, new ScalarTypeOptional(torch.ScalarType.Float)); // [Out]
        Tensor scores = gate.matmul(seedVec); // [N]

        // org.bytedeco.pytorch.geometric.utils.Scatter Softmax
        Tensor alpha = AggrUtils.scatter_softmax(scores, index, dimSize); // [N]

        // Weighted Sum
        Tensor out = AggrUtils.scatter(feat.mul(alpha.unsqueeze(1)), index, dimSize, "sum");

        return linOut.forward(out);
    }
}