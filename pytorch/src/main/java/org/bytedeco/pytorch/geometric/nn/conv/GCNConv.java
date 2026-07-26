package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops;
//import org.gnn.framework.nn.org.bytedeco.pytorch.geometric.nn.conv.MessagePassing;

/**
 * 实现 org.bytedeco.pytorch.geometric.nn.conv.GCNConv
 */
public class GCNConv extends MessagePassing {

    private LinearImpl lin;
    private long inChannels;
    private long outChannels;

    public GCNConv(org.bytedeco.javacpp.Pointer p) {
        super(p);
    }
    public GCNConv(long inChannels, long outChannels) {
        super("add"); // GCN 使用加法聚合
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        // 注册参数 Linear
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin);
    }

    // 前向传播
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 1. 线性变换 X * Theta
        // Shape: [N, outChannels]
        // 注意：在 JavaCPP 中需要确保 edge_index 的类型和设备一致
        edge_index = add_self_loops(edge_index, x.size(0));
        Tensor xTransformed = lin.forward(x);
// 3. 计算对称归一化系数 (D^-0.5 * A_hat * D^-0.5)
        Tensor norm = gcn_norm(edge_index, xTransformed.size(0), xTransformed.scalar_type());
        // 2. 计算归一化系数 (类似于 PyG 的 gcn_norm)
        // 这里简化：假设 edge_index 已经包含了自环
        // Degree Matrix Calculation:
//        Tensor row = edge_index.index_select(0, torch.tensor(0));
//        Tensor col = edge_index.index_select(0, torch.tensor(1));
//
//        // 计算度: degree[i] = sum(1 for neighbor)
//        Tensor ones = torch.ones(new long[]{row.size(0)}, x.options());
        // 注意：计算的是 source 节点的度还是 target？通常 GCN 是对称归一化
        // 此处简化实现，仅做示意，未实现完整的 D^-0.5 * D^-0.5

        // 3. 开始传播
        // GCN 的 message 阶段其实就是传递加权后的 x
        return propagate(edge_index, xTransformed, norm);
    }

    @Override //, Tensor edge_index
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 在标准的 GCN 中，这里应该乘以归一化系数 norm
        // msg = x_j * norm
        // x_j 是源节点的特征，norm 是预计算好的归一化系数
        // 将 norm 重塑为 [E, 1] 以便进行广播乘法
//        return norm.view(-1, 1).mul(x_j);
//        return x_j.mul(norm.view(-1, 1));
        // edge_attr 在这里就是 forward 传进来的 norm
        if (edge_attr != null) {
            // norm: [E] -> [E, 1]
            return x_j.mul(edge_attr.view(-1, 1));
        }
        return x_j;
    }

    /**
     * 辅助方法：计算归一化系数
     */
    private Tensor gcn_norm(Tensor edge_index, long numNodes, torch.ScalarType dtype) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // 计算度 (Degree)
        Tensor deg = torch.zeros(new long[]{numNodes}, edge_index.options().dtype(new ScalarTypeOptional(dtype)));
        Tensor ones = torch.ones(new long[]{row.size(0)}, edge_index.options().dtype(new ScalarTypeOptional(dtype)));
        deg.scatter_add_(0, col, ones); // 目标节点的度

        // 计算 D^-0.5
        Tensor deg_inv_sqrt = deg.pow(new Scalar(-0.5));
        // 处理无穷大（度为0的情况）
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt.isinf(), new Scalar(0));

        // 获取每个 edge 对应的 norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        Tensor norm = deg_inv_sqrt.index_select(0, row).mul(deg_inv_sqrt.index_select(0, col));
        return norm;
    }
}
