package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.SimpleConv
 * 一个简单的非训练传播算子。
 */
public class SimpleConv extends MessagePassing {
    private String combineRoot; // "sum", "cat", "self_loop", null

    /**
     * @param aggr 聚合方式: "sum", "mean", "min", "max" 等
     * @param combineRoot 中心节点结合方式: "sum", "cat", "self_loop", null
     */
    public SimpleConv(String aggr, String combineRoot) {
        super(aggr);
        this.combineRoot = combineRoot;
    }

    public SimpleConv(String aggr) {
        this(aggr, null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        // 1. 基础传播: 聚合邻居消息
        // 如果有 edge_weight，将其作为 edge_attr 传入
        Tensor out = propagate(edge_index, x, edge_weight);

        // 2. 结合中心节点 (Root Combination)
        if (combineRoot == null) {
            return out;
        }

        switch (combineRoot.toLowerCase()) {
            case "sum":
                // x_i' = x_i + aggr(x_j)
                return out.add(x);
            case "cat":
                // x_i' = [x_i, aggr(x_j)]
                return torch.cat(new TensorVector(x, out), -1);
            case "self_loop":
                // 这种模式通常在外部处理 edge_index (添加自环)
                // 这里如果显式处理，可以用 add
                return out.add(x);
            default:
                return out;
        }
    }

    /**
     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr, numNodes)
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 如果有边权重，则进行加权
        if (edge_attr != null) {
            // edge_attr 形状可能是 [E] 或 [E, 1]
            return x_j.mul(edge_attr.view(-1, 1));
        }
        return x_j;
    }

    // 注意：aggregate 方法由父类 MessagePassing 根据构造函数传入的 aggr ("sum", "mean"等) 自动处理
}