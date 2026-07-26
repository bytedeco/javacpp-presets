package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;


/**
 * Set Transformer org.bytedeco.pytorch.geometric.aggr.Aggregation (PMA)
 * Q = Seed, K = X, V = X
 */
public class SetTransformerAggregation extends Aggregation {
    private Tensor seed;
    private LinearImpl linQ, linK, linV, linOut;
    private long numHeads;
    private long outChannels;

    public SetTransformerAggregation(long inChannels, long outChannels, long numHeads, long numSeeds) {
        this.numHeads = numHeads;
        this.outChannels = outChannels;

        // Seed: [NumSeeds, Out]
        this.seed = new Tensor(torch.randn(new long[]{numSeeds, outChannels})); //Parameter
        register_parameter("seed", seed);

        this.linQ = new LinearImpl(outChannels, outChannels);
        this.linK = new LinearImpl(inChannels, outChannels);
        this.linV = new LinearImpl(inChannels, outChannels);
        this.linOut = new LinearImpl(outChannels, outChannels);

        register_module("linQ", linQ);
        register_module("linK", linK);
        register_module("linV", linV);
        register_module("linOut", linOut);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 注意：标准 SetTransformer 是全局的。
        // 在 GNN 中做局部聚合时，需要对每个子图（Target Node）单独应用 Attention。
        // 这在 org.bytedeco.pytorch.geometric.utils.Scatter 操作中比较难高效实现（需要 Segmented MatMul）。
        // 这里我们实现简化版：Global Pooling 模式，或者假设 numSeeds=1 且使用 org.bytedeco.pytorch.geometric.utils.Scatter Softmax 模拟。

        // 如果是 numSeeds=1，退化为 Global Attention Pooling (org.bytedeco.pytorch.geometric.aggr.AttentionalAggregation 的变体)
        // 下面实现 numSeeds=1 的高效 org.bytedeco.pytorch.geometric.utils.Scatter 版本。如果 numSeeds > 1，通常只能用于 Global Pooling。

        if (seed.size(0) != 1) {
            throw new UnsupportedOperationException("Local SetTransformer currently supports numSeeds=1 via scatter.");
        }

        // 1. Q = Seed [1, Out] -> 映射后
        Tensor Q = linQ.forward(seed); // [1, Out]

        // 2. K, V
        Tensor K = linK.forward(x); // [N, Out]
        Tensor V = linV.forward(x); // [N, Out]

        // 3. Attention Score = (K * Q^T) / sqrt(d)
        // [N, Out] * [Out, 1] -> [N, 1]
        Tensor scores = K.matmul(Q.t());
        scores = scores.mul(new Scalar(1.0 / Math.sqrt(outChannels)));

        // 4. Softmax
        Tensor alpha = AggrUtils.scatter_softmax(scores, index, dimSize);

        // 5. Weighted Sum
        Tensor agg = AggrUtils.scatter(V.mul(alpha), index, dimSize, "sum");

        return linOut.forward(agg);
    }
}