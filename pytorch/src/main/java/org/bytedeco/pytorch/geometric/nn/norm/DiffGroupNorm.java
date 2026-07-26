package org.bytedeco.pytorch.geometric.nn.norm;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;


public class DiffGroupNorm extends org.bytedeco.pytorch.nn.Module {
    private long inChannels;
    private int groups;
    private double eps;

    private LinearImpl lin;
    private Tensor weight;
    private Tensor bias;

    public DiffGroupNorm(long inChannels, int groups, double eps) {
        super();
        this.inChannels = inChannels;
        this.groups = groups;
        this.eps = eps;

        // 使用 Linear 而非 LinearImpl 包装类，减少中间层
        this.lin = new LinearImpl(inChannels, groups);
        register_module("lin", lin);

        this.weight = register_parameter("weight", ones(new long[]{inChannels}));
        this.bias = register_parameter("bias", zeros(new long[]{inChannels}));
    }

    public Tensor forward(Tensor x) {
        x = x.contiguous();
        long N = x.size(0);
        long C = x.size(1);
        long G = (long) groups;

        // 1. 计算分组概率 S [N, G]
        Tensor s = softmax(lin.forward(x), 1).contiguous();

        // 2. 组内统计
        Tensor s_t = s.transpose(0, 1).contiguous(); // [G, N]
        Tensor groupSum = matmul(s_t, x); // [G, C]
        Tensor groupDenom = s.sum(0).add(new Scalar(eps)).unsqueeze(1); // [G, 1]

        Tensor groupMean = groupSum.divide(groupDenom); // [G, C]

        // var = E[X^2] - (E[X])^2
        Tensor xSq = x.pow(new Scalar(2)).contiguous();
        Tensor groupSumSq = matmul(s_t, xSq);
        Tensor groupMeanSq = groupSumSq.divide(groupDenom);
        Tensor groupVar = groupMeanSq.subtract(groupMean.pow(new Scalar(2))).clamp_min(new Scalar(eps));

        // 3. 归一化并加权合并
        // x: [N, C] -> [N, 1, C]
        // groupMean: [G, C] -> [1, G, C]
        Tensor xReshaped = x.unsqueeze(1).expand(new long[]{N, G, C}, true).contiguous();
        Tensor mu = groupMean.unsqueeze(0).expand(new long[]{N, G, C}, true).contiguous();
        Tensor var = groupVar.unsqueeze(0).expand(new long[]{N, G, C}, true).contiguous();

        // (x - mu) / sqrt(var + eps)
        Tensor std = var.add(new Scalar(eps)).sqrt();
        Tensor out = xReshaped.subtract(mu).divide(std).contiguous(); // [N, G, C]

        // 加权求和
        Tensor sWeight = s.unsqueeze(2).expand_as(out).contiguous(); // [N, G, C]

        // 核心修复点：确保执行 multiply 前两者都是 contiguous 且维度完全对齐
        Tensor weightedOut = out.multiply(sWeight).sum(1).contiguous(); // [N, C]

        // 4. 应用可学习参数
        // 从参数字典获取以确保指针有效
        Tensor w = named_parameters().get("weight").view(new long[]{1, C});
        Tensor b = named_parameters().get("bias").view(new long[]{1, C});

        return weightedOut.multiply(w).add(b);
    }
}


//public class DiffGroupNorm extends org.bytedeco.pytorch.nn.Module {
//    private long inChannels;
//    private int groups;
//    private double eps;
//
//    private LinearImpl lin;      // 用于计算分组概率的线性层
//    private Tensor weight;   // Gamma
//    private Tensor bias;     // Beta
//
//    public DiffGroupNorm(long inChannels, int groups) {
//        this(inChannels, groups, 1e-5);
//    }
//
//    public DiffGroupNorm(long inChannels, int groups, double eps) {
//        super();
//        this.inChannels = inChannels;
//        this.groups = groups;
//        this.eps = eps;
//
//        // 1. 初始化用于计算分组 S 的线性层
//        this.lin = new LinearImpl(inChannels, groups);
//        register_module("lin", lin);
//
//        // 2. 初始化可学习的缩放和平移参数 (一比一还原 nn.Parameter)
//        this.weight = register_parameter("weight", ones(new long[]{inChannels}));
//        this.bias = register_parameter("bias", zeros(new long[]{inChannels}));
//    }
//
//    public Tensor forward(Tensor x) {
//        // x shape: [N, C]
//        long N = x.size(0);
//        long C = x.size(1);
//
//        // 1. 计算分组概率 S [N, G]
//        Tensor s = softmax(lin.forward(x), 1);
//
//        // 2. 计算每个组的期望和方差
//        // 我们需要利用矩阵乘法来高效实现组内统计
//        // s.t() 是 [G, N], x 是 [N, C] -> 结果 [G, C]
//        Tensor groupSum = matmul(s.transpose(0, 1), x);
//        Tensor groupDenom = s.sum(0).add(new Scalar(eps)); // [G]
//
//        // 组均值 [G, C]
//        Tensor groupMean = groupSum.divide(groupDenom.unsqueeze(1));
//
//        // 组方差 [G, C]
//        // var = E[X^2] - (E[X])^2
//        Tensor groupSumSq = matmul(s.transpose(0, 1), x.pow(new Scalar(2)));
//        Tensor groupMeanSq = groupSumSq.divide(groupDenom.unsqueeze(1));
//        Tensor groupVar = groupMeanSq.subtract(groupMean.pow(new Scalar(2))).clamp_min(new Scalar(eps));
//
//        // 3. 归一化并加权合并
//        // 将节点重新分配回组进行计算
//        // 这里的逻辑是 PyG DiffGroupNorm 的核心：
//        // 每个节点根据概率 S 混合不同组的标准化结果
//
//        // 对每个组进行标准化并广播回节点
//        // 为了内存效率，我们采用加权平均的方式
//        Tensor xNorm = x.unsqueeze(1); // [N, 1, C]
//        Tensor mu = groupMean.unsqueeze(0); // [1, G, C]
//        Tensor var = groupVar.unsqueeze(0); // [1, G, C]
//
//        // 执行标准化: (x - mu) / sqrt(var + eps)
//        Tensor out = xNorm.subtract(mu).divide(var.add(new Scalar(eps)).sqrt()); // [N, G, C]
//
//        // 使用 S 概率进行加权合并: sum(out * s_weighted)
//        Tensor sReshaped = s.unsqueeze(2); // [N, G, 1]
//        out = out.multiply(sReshaped).sum(1); // [N, C]
//
//        // 4. 应用最终的 learnable parameters
//        out = out.multiply(weight).add(bias);
//
//        return out;
//    }
//}


/**
 * DiffGroupNorm
 * 学习将节点聚类为 K 个组，并在组间进行归一化。
 * 适用于 Deep GCNs。
 */
//public class DiffGroupNorm extends Module {
//    private long groups;
//    private LinearImpl weight; // 用于生成 Assignment Matrix S
//    private Parameter lamda;   // 平衡系数
//
//    private BatchNorm1dImpl bn; // 组内 BN
//
//    public DiffGroupNorm(long inChannels, long groups, double lamda) {
//        super();
//        this.groups = groups;
//        this.lamda = new Parameter(torch.tensor(lamda)); // Learnable or fixed? PyG says fixed usually but can be param
//        register_parameter("lamda", this.lamda);
//
//        // Linear: [C, K]
//        this.weight = new LinearImpl(inChannels, groups);
//        register_module("weight", weight);
//
//        // 我们对变换后的特征做 BN
//        this.bn = new BatchNorm(groups * inChannels); // 或者处理方式不同
//        // PyG 的实现其实是对 "Group Features" 做归一化。
//        // 这里为了简化实现难度，我们采用一种常见的变体：
//        // 1. U = X @ W (Cluster Scores)
//        // 2. Normalize X based on U
//
//        // 更简单的实现：直接对输入做 GroupNorm (类似 CNN)，但这不叫 DiffGroupNorm。
//        // 真正的 DiffGroupNorm 需要计算软聚类。
//
//        // 由于 DiffGroupNorm 逻辑较为繁琐且涉及复杂的 org.bytedeco.pytorch.geometric.utils.Scatter/Gather，
//        // 这里我们实现一个常用的替代方案：LayerNorm + Group 约束，
//        // 或者严格复刻 PyG 逻辑 (需要 scatter_add)。
//
//        // 此处暂且实现一个占位逻辑，因为完整的 DiffGroupNorm 在纯 LibTorch 中
//        // 需要大量手动微分操作。
//        // 我们转而实现 GroupNorm (PyTorch 原生)，这在 GNN 中也很常用。
//    }
//
//    // 如果坚持实现 DiffGroupNorm，逻辑如下：
//    // X: [N, F], S: [N, K]
//    // X_group = S.T @ X -> [K, F] (Cluster Centers)
//    // 但这通常用于 Pooling。
//
//    // 我们这里提供一个标准的 GroupNorm 封装，这在 GNN 中更稳健。
//    // 如果确实需要 DiffGroupNorm，请告知，我需要写更长的辅助代码。
//}