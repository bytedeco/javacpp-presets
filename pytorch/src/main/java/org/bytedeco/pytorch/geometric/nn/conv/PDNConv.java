package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.PDNConv
 * 基于路径发现机制，利用边特征对节点消息进行分维度过滤。
 */

public class PDNConv extends MessagePassing {
    private LinearImpl linEdge;
    private LinearImpl linRes;
    private boolean normalize;
    private Tensor bias;

    public PDNConv(long inChannels, long outChannels, int edgeDim, int hiddenChannels, boolean normalize, boolean hasBias) {
        super("sum");
        this.normalize = normalize;

        // 边特征处理: edgeDim -> inChannels (用于和 x_j 做 mask)
        this.linEdge = new LinearImpl(edgeDim, (int)inChannels);
        this.linRes = new LinearImpl(inChannels, outChannels);

        register_module("lin_edge", linEdge);
        register_module("lin_res", linRes);

        if (hasBias) {
            this.bias = torch.zeros(new long[]{outChannels});
            register_parameter("bias", bias);
        }
    }

    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        long N = x.size(0);
        Tensor norm = normalize ? compute_normalization(edge_index, N) : null;

        // 显式调用自定义传播，避开基类反射机制的歧义
        return propagate_pdn(edge_index, x, edge_attr, norm);
    }

    private Tensor propagate_pdn(Tensor edge_index, Tensor x, Tensor edge_attr, Tensor norm) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);

        // 准备消息
        Tensor x_j = x.index_select(0, row);

        // 计算 Edge Mask [E, inChannels]
        Tensor edge_mask = torch.sigmoid(linEdge.forward(edge_attr));

        // 消息融合: x_j * mask
        Tensor msg = x_j.mul(edge_mask);

        // 归一化
        if (norm != null) {
            msg = msg.mul(norm.view(-1, 1));
        }

        // 聚合
        Tensor out = aggregate(msg, col, x.size(0));

        // 最终线性变换
        out = linRes.forward(out);
        if (bias != null) out = out.add(bias);

        return out;
    }

        @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 覆盖基类防止其执行默认的 mul(edge_attr)
        return x_j;
    }

    private Tensor compute_normalization(Tensor edge_index, long numNodes) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor deg = torch.zeros(new long[]{numNodes}, edge_index.options());
        deg.scatter_add_(0, col, torch.ones(new long[]{edge_index.size(1)}, edge_index.options()));
        Tensor degInv = deg.pow(new Scalar(-1.0));
        degInv.masked_fill_(degInv.isinf(), new Scalar(0.0));
        return degInv.index_select(0, col); // 目标节点的度归一化
    }
}
//public class PDNConv extends MessagePassing {
//    private LinearImpl linEdge;     // 对应 MLP(e_ji)
//    private LinearImpl linRes;      // 对应最终的线性变换
//    private boolean normalize;
//    private Tensor bias;
//
//    public PDNConv(long inChannels, long outChannels, int edgeDim, int hiddenChannels, boolean normalize, boolean hasBias) {
//        super("sum"); // PDN 默认使用求和聚合
//        this.normalize = normalize;
//
//        // 1. 边特征处理网络 (严格使用 LinearImpl)
//        // 输入 edgeDim -> 隐藏层 -> 输出 inChannels (为了与 x_j 进行逐元素乘法)
//        // 这里简化为一层变换，实际论文通常建议使用两层 MLP
//        this.linEdge = new LinearImpl(edgeDim, inChannels);
//        register_module("lin_edge", linEdge);
//
//        // 2. 聚合后的输出变换
//        this.linRes = new LinearImpl(inChannels, outChannels);
//        register_module("lin_res", linRes);
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, (Tensor)null);
//    }
//    
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
//        if (edge_attr == null) {
//            throw new IllegalArgumentException("PDNConv requires edge_attr (edge features).");
//        }
//
//        long N = x.size(0);
//
//        // 1. 预计算归一化系数 (可选)
//        Tensor norm = null;
//        if (normalize) {
//            norm = compute_normalization(edge_index, N);
//        }
//
//        // 2. 消息传递与聚合
//        Tensor out = propagate(edge_index, x, edge_attr, norm);
//
//        // 3. 最终线性映射
////        out = linRes.forward(out);
////
////        if (bias != null) {
////            out = out.add(bias);
////        }
//
//        return out;
//    }
//
////    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, Tensor norm, long numNodes) {
//        // 1. 计算边掩码 (MLP 变换)
//        // edge_attr 形状 [15, edgeDim] -> linEdge -> [15, 16]
//        Tensor edge_mask = torch.sigmoid(linEdge.forward(edge_attr));
//
//        // 2. 执行节点特征与边掩码的逐元素乘法
//        // x_j: [15, 16], edge_mask: [15, 16] -> 结果: [15, 16]
//        Tensor msg = x_j.mul(edge_mask);
//
//        // 3. 应用归一化系数 (如果 norm 不为空)
//        if (norm != null && norm.defined() && norm.size(0) != 0) {
//            // norm 形状通常是 [15], 需要 unsqueeze(-1) 变成 [15, 1] 以便广播到 [15, 16]
//            msg = msg.mul(norm.view(new long[]{-1, 1}));
//        }
//
//        return msg;
//    }
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 1. 计算边掩码
//        // 假设 linEdge 将 edge_attr 映射到维度 1
//        Tensor edge_mask = torch.sigmoid(linEdge.forward(edge_attr));
//
//        // 2. 核心修正：确保 edge_mask 是 [15, 1] 而不是 [15]
//        // 只有 [15, 16] mul [15, 1] 才能正确广播
//        if (edge_mask.dim() == 1) {
//            edge_mask = edge_mask.unsqueeze(-1);
//        } else if (edge_mask.size(1) != 1 && edge_mask.size(1) != x_j.size(1)) {
//            // 如果 mask 既不是 1 维也不是全特征维，强制转为 [E, 1]
//            edge_mask = edge_mask.view(new long[]{-1, 1});
//        }
//
//        // 3. 执行乘法
//        // 此时 x_j: [15, 16], edge_mask: [15, 1] -> 结果: [15, 16]
//        return x_j.mul(edge_mask);
//    }
//
//
//    private Tensor compute_normalization(Tensor edge_index, long numNodes) {
//        // 标准的对称归一化 D^-0.5 * A * D^-0.5
//        Tensor row = edge_index.select(0, 0);
//        Tensor deg = torch.zeros(new long[]{numNodes}, edge_index.options());
//        deg.scatter_add_(0, row, torch.ones(new long[]{edge_index.size(1)}, edge_index.options()));
//        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
//        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0.0));
//        return degInvSqrt.index_select(0, row); // 返回源节点的归一化系数
//    }
//}



//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 核心逻辑: x_j * MLP(edge_attr)
//        // linEdge 将边特征转换为与节点特征维度相同的掩码
//        Tensor edge_mask = torch.sigmoid(linEdge.forward(edge_attr));
//        if (edge_mask.dim() == 1) {
//            edge_mask = edge_mask.unsqueeze(-1);
//        }
//        Tensor msg = x_j.mul(edge_mask);
//
//        // 应用对称归一化 (如果开启)
//        // 这里传递过来的 edge_attr 实际上在内部通过 propagate 机制处理
//        return msg;
//    }