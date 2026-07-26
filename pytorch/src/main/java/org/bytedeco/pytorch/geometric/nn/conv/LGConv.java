package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.LGConv
 * 轻量化图卷积算子：去除线性变换和激活函数，仅保留传播。
 */
public class LGConv extends MessagePassing {
    private boolean normalize;

    public LGConv(boolean normalize) {
        super("add");
        this.normalize = normalize;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    /**
     * @param x          节点特征 [N, channels]
     * @param edge_index 边索引 [2, E]
     * @param edge_weight 边权重 (可选)
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        long N = x.size(0);

        // 1. 计算对称归一化因子: D^-0.5 * A * D^-0.5
        Tensor norm = null;
        if (normalize) {
            norm = compute_normalization(edge_index, edge_weight, N);
        }

        // 2. 消息传递：直接聚合邻居特征，没有任何 W 矩阵
        return propagate(edge_index, x, norm);
    }

    private Tensor compute_normalization(Tensor edge_index, Tensor edge_weight, long numNodes) {
        if (edge_weight == null) {
            edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());
        }

        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // 计算度 Degree
        Tensor deg = torch.zeros(new long[]{numNodes}, edge_weight.options());
        deg.scatter_add_(0, row, edge_weight);

        // D^-0.5
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));

        // 归一化系数: d_i^-0.5 * w_ij * d_j^-0.5
        return degInvSqrt.index_select(0, row)
                .mul(edge_weight)
                .mul(degInvSqrt.index_select(0, col));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 直接返回邻居特征乘以归一化系数
        return x_j.mul(edge_attr.view(-1, 1));
    }

//    @Override
    public void reset_parameters() {
        // 无参算子，无需重置
    }
}