package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.WLConvContinuous
 * 用于连续特征的 Weisfeiler-Lehman 算子。
 * 该算子通过度缩放的均值聚合来细化连续特征。
 */
public class WLConvContinuous extends MessagePassing {

    public WLConvContinuous() {
        super("add"); // 基础聚合使用 add，后续手动进行度缩放
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    /**
     * @param x          节点连续特征 [N, channels]
     * @param edge_index 边索引 [2, E]
     * @param edge_weight 边权重 (可选)
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        long N = x.size(0);

        // 1. 聚合邻居特征: \sum_{j \in N(i)} w_ji * x_j
        Tensor out = propagate(edge_index, x, edge_weight);

        // 2. 计算度 (Degree) d_i
        Tensor deg = torch.zeros(new long[]{N}, x.options());
        Tensor row = edge_index.select(0, 1); // Target nodes

        if (edge_weight == null) {
            edge_weight = torch.ones(new long[]{edge_index.size(1)}, x.options());
        }
        deg.scatter_add_(0, row, edge_weight);

        // 3. 应用公式: x_i' = 0.5 * (x_i + (1/d_i) * \sum w_ji * x_j)
        // 也就是节点自身特征与邻域均值的等权融合
        Tensor invDeg = deg.pow(new Scalar(-1.0));
        invDeg.masked_fill_(invDeg.isinf(), new Scalar(0.0)); // 处理孤立点

        // 邻域均值
        Tensor meanAgg = out.mul(invDeg.view(-1, 1));

        return x.add(meanAgg).mul(new Scalar(0.5));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 如果有边权重，则加权邻居特征
        if (edge_attr != null) {
            return x_j.mul(edge_attr.view(-1, 1));
        }
        return x_j;
    }


//    @Override
    public void reset_parameters() {
        // 非参算子，无需操作
    }
}