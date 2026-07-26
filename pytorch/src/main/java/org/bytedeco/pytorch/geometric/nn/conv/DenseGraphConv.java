package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;


public class DenseGraphConv extends MessagePassing {
    private LinearImpl linRel;    // 邻居特征映射层 (A·X → out_channels)
    private LinearImpl linRoot;   // 自环特征映射层 (X → out_channels)
    private long inChannels;      // 输入通道数（记录用于验证）
    private long outChannels;     // 输出通道数（记录用于验证）

    // 核心构造方法：初始化线性层 + 注册模块 + 初始化基类
    public DenseGraphConv(long inChannels, long outChannels) {
//        super(false); // 必须初始化 MessagePassing 基类（解决JNI崩溃）
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        // 初始化线性层（默认 float32，保证数值稳定性）
        this.linRel = new LinearImpl(inChannels, outChannels);
        this.linRoot = new LinearImpl(inChannels, outChannels);

        // 注册模块到基类（保证参数被优化器跟踪 + 资源统一管理）
        register_module("linRel", this.linRel);
        register_module("linRoot", this.linRoot);
    }

    /**
     * 前向传播（修正签名 + 设备对齐 + 核心逻辑）
     * @param x 节点特征 [B, N, in_channels]
     * @param adj 邻接矩阵 [B, N, N]（稠密矩阵）
     * @return 卷积输出 [B, N, out_channels]
     */
    @Override
    public Tensor forward(Tensor x, Tensor adj) {
        // 安全校验：输入维度必须匹配
        if (x.dim() != 3) {
            throw new IllegalArgumentException("x 必须是 3 维张量 [B, N, in_channels]，当前维度：" + x.dim());
        }
        if (adj.dim() != 3) {
            throw new IllegalArgumentException("adj 必须是 3 维张量 [B, N, N]，当前维度：" + adj.dim());
        }
        if (x.size(2) != this.inChannels) {
            throw new IllegalArgumentException("x 输入通道数不匹配：期望 " + this.inChannels + "，实际 " + x.size(2));
        }

        // 设备/类型对齐：保证线性层与输入 x 在同一设备/类型（避免JNI错误）
        this.linRel.to(x.device(), x.scalar_type(),false);
        this.linRoot.to(x.device(), x.scalar_type(),false);

        // GraphConv 核心计算：
        // 1. 邻居特征求和：A·X (adj是[B,N,N], x是[B,N,C] → [B,N,C])
        Tensor neighbor = adj.matmul(x);
        // 2. 邻居特征映射：linRel(A·X) → [B,N,out_channels]
        Tensor neighborOut = this.linRel.forward(neighbor);
        // 3. 自环特征映射：linRoot(X) → [B,N,out_channels]
        Tensor rootOut = this.linRoot.forward(x);
        // 4. 合并输出：邻居特征 + 自环特征
        Tensor out = neighborOut.add(rootOut);

        // 释放临时张量（避免内存泄漏）
        neighbor.close();
        neighborOut.close();
        rootOut.close();

        return out;
    }

    /**
     * 复写 MessagePassing 基类的 message 方法（必须实现）
     * 签名：(x_j, x_i, edge_index, edge_attr, numNodes)
     * 稠密图场景下仅保持签名兼容，实际逻辑在 forward 中实现
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j; // GraphConv 的 message 就是邻居特征本身
    }

    // ====== 辅助方法：获取参数/配置（用于测试验证） ======
    public LinearImpl getLinRel() {
        return this.linRel;
    }

    public LinearImpl getLinRoot() {
        return this.linRoot;
    }

    public long getInChannels() {
        return this.inChannels;
    }

    public long getOutChannels() {
        return this.outChannels;
    }

    // ====== 资源释放：避免JNI内存泄漏 ======
    @Override
    public void close() {
        // 先释放子类资源
        if (this.linRel != null) {
            this.linRel.close();
            this.linRel = null;
        }
        if (this.linRoot != null) {
            this.linRoot.close();
            this.linRoot = null;
        }
        // 再释放基类资源
        super.close();
    }

    // 防止GC时未手动close导致内存泄漏
//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            close();
//        } finally {
//            super.finalize();
//        }
//    }
}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//
//public class DenseGraphConv extends MessagePassing {
//    private LinearImpl linRel;
//    private LinearImpl linRoot;
//
//    public DenseGraphConv(long inChannels, long outChannels) {
//        this.linRel = new LinearImpl(inChannels, outChannels); // Neighbor
//        this.linRoot = new LinearImpl(inChannels, outChannels); // Self
//        register_module("linRel", linRel);
//        register_module("linRoot", linRoot);
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor adj) {
//        // Neighbor: A @ X
//        Tensor neighbor = adj.matmul(x);
//
//        return linRel.forward(neighbor).add(linRoot.forward(x));
//    }
//    /**
//     * 必须匹配基类签名：(x_j, x_i, edge_index, edge_attr)
//     * 哪怕 SAGE 只需要 x_j，参数也必须写全！
//     */
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // GraphSAGE 的 message 就是邻居特征本身
//        // 如果以后要支持带权重的 SAGE，可以在这里处理 edge_attr
//        return x_j;
//    }
//}