package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 修复后的稠密图卷积层（DenseGCNConv）
 * 继承 MessagePassing 符合 PyG 规范，适配稠密邻接矩阵
 * 输入：x [B, N, in_channels], adj [B, N, N], mask [B, N]（可选）
 * 输出：out [B, N, out_channels]
 */
public class DenseGCNConv extends MessagePassing {
    public LinearImpl lin;
    private boolean improved; // 是否添加 2 次自环（A + 2I）

    // 构造器：修正参数名拼写（improved 而非 improve）
    public DenseGCNConv(long inChannels, long outChannels, boolean improved) {
        super(); // 调用 MessagePassing 父类构造器
        this.improved = improved;
        this.lin = new LinearImpl(inChannels, outChannels);
        register_module("lin", lin); // 注册线性层为子模块
    }

    /**
     * 适配稀疏图的 forward（兼容基类，实际稠密场景不使用）
     * @param x 节点特征 [B, N, in]
     * @param edge_index 稀疏边索引（稠密场景传 null）
     * @return 卷积输出 [B, N, out]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        throw new UnsupportedOperationException("稠密 GCN 请调用 forward(x, adj, mask) 方法");
    }

    /**
     * 稠密 GCN 核心前向传播
     * @param x   节点特征 [B, N, in_channels]
     * @param adj 稠密邻接矩阵 [B, N, N] (0/1 矩阵)
     * @param mask 节点掩码 [B, N]（可选，用于屏蔽 padding 节点）
     * @return 卷积输出 [B, N, out_channels]
     */
    public Tensor forward(Tensor x, Tensor adj, TensorOptional mask) {
        // 输入维度校验
        if (x.dim() != 3) {
            throw new IllegalArgumentException("x 必须是 3 维张量 [B, N, in_channels]，当前维度：" + x.dim());
        }
        if (adj.dim() != 3) {
            throw new IllegalArgumentException("adj 必须是 3 维张量 [B, N, N]，当前维度：" + adj.dim());
        }

        long B = x.size(0);
        long N = x.size(1);

        // 1. 邻接矩阵添加自环（GCN 核心：A = A + I 或 A + 2I）
        // 生成适配批次维度的单位矩阵：[1, N, N] -> [B, N, N]
        Tensor eye = torch.eye(N, x.options()).unsqueeze(0).expand(new long[]{B, N, N});
        Tensor hatA = adj.add(eye);
        if (improved) {
            hatA = hatA.add(eye); // 改进版：添加 2 次自环
        }

        // 2. 度矩阵归一化（D^-0.5 * A * D^-0.5）
        // 计算度：sum(adj, dim=2) -> [B, N]
        Tensor deg = hatA.sum(new long[]{2}, false, new ScalarTypeOptional());
        // 度的负 0.5 次方（避免除 0）
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        // 处理 inf/NaN：替换为 0
        degInvSqrt.masked_fill_(torch.logical_or(degInvSqrt.isinf(),degInvSqrt.isnan()), new Scalar(0));

        // 归一化邻接矩阵：[B,N,N] * [B,N,1] * [B,1,N] -> [B,N,N]
        Tensor normAdj = hatA.mul(degInvSqrt.unsqueeze(2)).mul(degInvSqrt.unsqueeze(1));

        // 3. GCN 核心计算：normAdj @ (X @ W)
        Tensor xW = lin.forward(x); // [B, N, in] -> [B, N, out]
        Tensor out = normAdj.matmul(xW); // [B, N, N] @ [B, N, out] -> [B, N, out]

        // 4. 节点掩码（可选：屏蔽 padding 节点）
        if (mask.has_value()) {
            Tensor maskTensor = mask.get();
            if (maskTensor.dim() != 2 || maskTensor.size(0) != B || maskTensor.size(1) != N) {
                throw new IllegalArgumentException("mask 必须是 2 维张量 [B, N]，当前维度/形状：" + maskTensor);
            }
            out = out.mul(maskTensor.unsqueeze(2).to(x.options().dtype()));
        }

        // 释放临时张量（JavaCPP 内存管理）
        eye.close();
        hatA.close();
        deg.close();
        degInvSqrt.close();
        normAdj.close();

        return out;
    }

    /**
     * 覆写 MessagePassing 要求的 message 方法（稠密 GCN 无需实现，返回空）
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return null;
    }

    // 释放资源（避免内存泄漏）
    @Override
    public void close() {
        if (lin != null) lin.close();
        super.close();
    }
}

//public class DenseGCNConv extends MessagePassing {
//    private LinearImpl lin;
//    private boolean improve; // Add self-loops 2 times?
//
//    public DenseGCNConv(long inChannels, long outChannels, boolean improved) {
//        this.improve = improved;
//        this.lin = new LinearImpl(inChannels, outChannels);
//        register_module("lin", lin);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, new TensorOptional());
//    }
//    /**
//     * @param x   Node features [B, N, In]
//     * @param adj Adjacency matrix [B, N, N]
//     * @param mask Mask [B, N] (Optional, to mask out padding nodes)
//     */
//    public Tensor forward(Tensor x, Tensor adj, TensorOptional mask) {
//        // 1. Normalize Adj
//        // D^-0.5 * A * D^-0.5 (Usually pre-computed, but here we do it on fly)
//        // Add Self Loops: A = A + I
//        long N = x.size(1);
//        Tensor eye = torch.eye(N, x.options()).unsqueeze(0); // [1, N, N]
//        Tensor hatA = adj.add(eye);
//        if (improve) hatA = hatA.add(eye); // +2I
//
//        // Degree: sum(A, dim=2) -> [B, N]
//        Tensor deg = hatA.sum(new long[]{2}, false, new ScalarTypeOptional());
//        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
//        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));
//
//        // Norm = D^-0.5 @ A @ D^-0.5
//        // Be careful with broadcasting: [B, N, 1] * [B, N, N] * [B, 1, N]
//        Tensor normAdj = hatA.mul(degInvSqrt.unsqueeze(2)).mul(degInvSqrt.unsqueeze(1));
//
//        // 2. Convolution: A @ X @ W
//        // XW = lin(x) -> [B, N, Out]
//        Tensor xW = lin.forward(x);
//
//        // AXW = normAdj @ xW
//        Tensor out = normAdj.matmul(xW);
//
//        // 3. Masking
//        if (mask.has_value()) {
//            out = out.mul(mask.get().unsqueeze(2).to(torch.ScalarType.Float));
//        }
//
//        return out;
//    }
//    /**
//     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
//     * 哪怕 SAGE 只需要 x_j，参数也必须写全！
//     */
////    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // GraphSAGE 的 message 就是邻居特征本身
//        // 如果以后要支持带权重的 SAGE，可以在这里处理 edge_attr
//        return x_j;
//    }
//}