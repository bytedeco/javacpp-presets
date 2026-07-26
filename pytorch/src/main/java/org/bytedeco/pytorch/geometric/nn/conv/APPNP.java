package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;

/**
 * 实现 torch_geometric.nn.conv.APPNP
 * 特点：非训练层，基于迭代 PPR 算法平滑神经网络的预测。
 */
public class APPNP extends MessagePassing {
    private int K;
    private double alpha;
    private double dropout;
    private boolean addSelfLoops;
    private boolean normalize;

    public APPNP(int K, double alpha, double dropout, boolean addSelfLoops, boolean normalize) {
        super("add");
        this.K = K;
        this.alpha = alpha;
        this.dropout = dropout;
        this.addSelfLoops = addSelfLoops;
        this.normalize = normalize;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        long N = x.size(0);

        // 1. 归一化处理 (D^-1/2 * A_hat * D^-1/2)
        Tensor norm = null;
        if (normalize) {
            norm = compute_normalization(edge_index, edge_weight, N);
        }

        // 2. 迭代传播
        // x_0 是原始预测（通常是 MLP 的输出）
        Tensor x_0 = x;
        Tensor x_k = x;

        // 训练模式下可选的边 Dropout (PyG 标准实现通常在此处对 norm 应用 dropout)
        // if (this.training() && dropout > 0) { ... }

        for (int i = 0; i < K; i++) {
            // x_(k+1) = (1 - alpha) * (A_hat @ x_k) + alpha * x_0
            Tensor aggregated = propagate(edge_index, x_k, norm);
            x_k = aggregated.mul(new Scalar(1.0 - alpha)).add(x_0.mul(new Scalar(alpha)));
        }

        return x_k;
    }

    private Tensor compute_normalization(Tensor edge_index, Tensor edge_weight, long numNodes) {
        // 标准对称归一化逻辑
        if (edge_weight == null) {
            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
        }

        // 如果需要，可以在此处添加自环
        // if (addSelfLoops) { ... }

        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        Tensor deg = Scatter.scatter(edge_weight, row, numNodes, "add");
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));

        return degInvSqrt.index_select(0, row)
                .mul(edge_weight)
                .mul(degInvSqrt.index_select(0, col));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 将归一化系数应用到邻居特征
        if (edge_attr != null) {
            return x_j.mul(edge_attr.view(-1, 1));
        }
        return x_j;
    }
}