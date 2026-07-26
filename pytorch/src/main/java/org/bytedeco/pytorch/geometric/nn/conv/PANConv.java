package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.PANConv
 * 基于路径积分的多阶邻域特征聚合算子。
 */
public class PANConv extends MessagePassing {
    private LinearImpl lin;         // 最终映射 W
    private Tensor weight;          // 路径权重 w_l [filter_size + 1]
    private int filterSize;

    public PANConv(long inChannels, long outChannels, int filterSize) {
        super("add");
        this.filterSize = filterSize;

        // 严格使用 LinearImpl
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin);

        // 路径权重 w: 对应从 0 阶到 L 阶路径的贡献
        this.weight = torch.randn(new long[]{filterSize + 1});
        register_parameter("weight", weight);
    }

    /**
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        long N = x.size(0);

        // 1. 计算归一化邻接矩阵 M (MET Matrix)
        // 简化实现：这里使用对称归一化的 A
        Tensor norm = compute_normalization(edge_index, N);

        // 2. 迭代计算路径积分: \sum_{l=0}^L w_l * M^l * X
        // l=0
        Tensor out = x.mul(weight.select(0, 0));

        Tensor x_l = x;
        for (int l = 1; l <= filterSize; l++) {
            // 计算 M^l * X = M * (M^{l-1} * X)
            x_l = propagate(edge_index, x_l, norm);
            out = out.add(x_l.mul(weight.select(0, l)));
        }

        // 3. 最终线性变换
        return lin.forward(out);
    }

    private Tensor compute_normalization(Tensor edge_index, long numNodes) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());

        Tensor deg = torch.zeros(new long[]{numNodes}, edge_index.options());
        deg.scatter_add_(0, row, edge_weight);

        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0.0));

        return degInvSqrt.index_select(0, row)
                .mul(degInvSqrt.index_select(0, col));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // edge_attr 存储的是对称归一化系数
        return x_j.mul(edge_attr.view(-1, 1));
    }
}