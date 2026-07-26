package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 稠密图 GraphSAGE 卷积层（继承 MessagePassing 基类）
 * 公式：out = W_rel * Mean(Neighbors) + W_root * X
 * 支持可选 L2 归一化，适配稠密张量输入 [B, N, C]
 */
public class DenseSAGEConv extends MessagePassing {
    private LinearImpl linRel;     // 邻居特征映射层 (Mean(Neighbors) → out_channels)
    private LinearImpl linRoot;    // 自环特征映射层 (X → out_channels)
    private boolean normalize;     // 是否开启输出L2归一化
    private long inChannels;       // 输入通道数
    private long outChannels;      // 输出通道数

    // 核心构造方法：支持归一化开关
    public DenseSAGEConv(long inChannels, long outChannels, boolean normalize) {
//        super(false); // 关键修复：初始化基类，false=稠密图模式（无流控制）
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.normalize = normalize;

        // 初始化线性层（默认float32，保证数值稳定性）
        this.linRel = new LinearImpl(inChannels, outChannels);
        this.linRoot = new LinearImpl(inChannels, outChannels);

        // 注册模块到基类（保证参数被优化器跟踪 + 资源统一管理）
        register_module("lin_rel", this.linRel);
        register_module("lin_root", this.linRoot);
    }

    // 简化构造方法：默认关闭归一化
    public DenseSAGEConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, false);
    }

    /**
     * 前向传播（修正度计算 + 归一化 + 设备对齐）
     * @param x   节点特征 [Batch, Nodes, inChannels]
     * @param adj 邻接矩阵 [Batch, Nodes, Nodes]（稠密矩阵）
     * @return 卷积输出 [Batch, Nodes, outChannels]
     */
    @Override
    public Tensor forward(Tensor x, Tensor adj) {
        // 安全校验：输入维度/通道数合规性
        if (x.dim() != 3) {
            throw new IllegalArgumentException("x 必须是3维张量 [B, N, C]，当前维度：" + x.dim());
        }
        if (adj.dim() != 3) {
            throw new IllegalArgumentException("adj 必须是3维张量 [B, N, N]，当前维度：" + adj.dim());
        }
        if (x.size(2) != this.inChannels) {
            throw new IllegalArgumentException("x 输入通道数不匹配：期望 " + inChannels + "，实际 " + x.size(2));
        }

        // 设备/类型对齐：保证线性层与输入x在同一设备/类型（避免JNI错误）
        this.linRel.to(x.device(), x.scalar_type(),false);
        this.linRoot.to(x.device(), x.scalar_type(),false);

        // ========== GraphSAGE 核心逻辑 ==========
        // 1. 邻居特征求和：A·X → [B, N, C]
        Tensor neighborSum = adj.matmul(x);

        // 2. 计算节点度（入度）：sum(adj, dim=2, keepdim=True) → [B, N, 1]
        //    修复：固定dim=2（列维度）求和，保留维度避免广播错误
        Tensor deg = adj.sum(new long[]{2}, true, new ScalarTypeOptional(torch.kFloat())); // 确保度是float类型，避免整数除法问题

        // 3. 邻居特征均值化：Mean(Neighbors) = Sum(Neighbors) / (Degree + 1e-6)
        //    加1e-6避免除零错误
        Tensor neighborMean = neighborSum.div(deg.add(new Scalar(1e-6)));

        // 4. 线性变换：邻居特征 + 自环特征
        Tensor relOut = this.linRel.forward(neighborMean);  // 邻居特征映射
        Tensor rootOut = this.linRoot.forward(x);           // 自环特征映射
        Tensor out = relOut.add(rootOut);                   // 合并输出

        // 5. 可选L2归一化：沿最后一维（通道维）归一化
        if (this.normalize) {
            // 修复：norm参数正确配置（p=2，dim=-1，keepdim=True）
            Tensor norm = out.norm(
                    new ScalarOptional(new Scalar(2.0)),  // p=2（L2范数）
                    new long[]{-1},                       // 沿最后一维归一化
                    true                                  // 保留维度，避免广播错误
            );
            // clamp_min避免除以0，保证数值稳定性
            out = out.div(norm.clamp_min(new Scalar(1e-12)));
            norm.close(); // 释放临时张量
        }

        // 释放临时张量（避免内存泄漏）
        neighborSum.close();
        deg.close();
        neighborMean.close();
        relOut.close();
        rootOut.close();

        return out;
    }

    /**
     * 复写 MessagePassing 基类的 message 方法（必须实现）
     * 签名：(x_j, x_i, edge_index, edge_attr, numNodes)
     * 稠密图场景下仅保持签名兼容，实际逻辑在forward中实现
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j; // GraphSAGE的message就是邻居特征本身
    }

    // ====== 辅助方法：获取参数/配置（用于测试验证） ======
    public LinearImpl getLinRel() {
        return this.linRel;
    }

    public LinearImpl getLinRoot() {
        return this.linRoot;
    }

    public boolean isNormalize() {
        return this.normalize;
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
//public class DenseSAGEConv extends MessagePassing {
//    private LinearImpl linRel;  // W1 (Self)
//    private LinearImpl linRoot; // W2 (Neighbor)
//    private boolean normalize;
//    
////    public DenseSAGEConv(long inChannels, long outChannels) {
////        DenseSAGEConv(long inChannels, long outChannels, boolean normalize)
////        this.linRel = new LinearImpl(inChannels, outChannels);
////        this.linRoot = new LinearImpl(inChannels, outChannels);
////        register_module("linRel", linRel);
////        register_module("linRoot", linRoot);
////    }
//    public DenseSAGEConv(long inChannels, long outChannels, boolean normalize) {
//        super();
////        super();
//        this.normalize = normalize;
//
//        // 稠密层的线性变换通常作用在最后一个维度
//        this.linRel = new LinearImpl(inChannels, outChannels);
//        this.linRoot = new LinearImpl(inChannels, outChannels);
//
//        register_module("lin_rel", linRel);
//        register_module("lin_root", linRoot);
//    }
//
//    /**
//     * @param x   特征张量 [Batch, Nodes, inChannels]
//     * @param adj 邻接矩阵 [Batch, Nodes, Nodes]
//     */
//    @Override
//    public Tensor forward(Tensor x, Tensor adj) {
//        // 1. 计算邻居平均特征
//        // adj @ x -> [B, N, N] @ [B, N, C] = [B, N, C] (邻居求和)
//        Tensor out = adj.matmul(x);
//
//        // 计算度 (Degree) 进行均值化
//        // 在 dim=2 (列) 求和得到每个节点的入度
//        Tensor deg = adj.sum(new long[]{-1}, true,new ScalarTypeOptional());
//        out = out.div(deg.add(new Scalar(1e-6))); // 避免除零
//
//        // 2. 线性变换
//        // 标准 SAGE: out = W_rel * aggr + W_root * x
//        Tensor relOut = linRel.forward(out);
//        Tensor rootOut = linRoot.forward(x);
//        out = relOut.add(rootOut);
//
//        // 3. 可选的 L2 归一化
//        if (this.normalize) {
//            out = out.div(out.norm(new ScalarOptional(new Scalar(2)), new long[]{-1}, true).clamp_min(new Scalar(1e-12)));
//        }
//
//        return out;
//    }
//    public Tensor forward2(Tensor x, Tensor adj) {
//        // 1. Neighbor org.bytedeco.pytorch.geometric.aggr.Aggregation: Mean(Neighbors)
//        // A @ X -> Sum neighbors. To get Mean, divide by degree.
//        Tensor deg = adj.sum(new long[]{2}, true,new ScalarTypeOptional()); // [B, N, 1]
//        Tensor aggr = adj.matmul(x).div(deg.add(new Scalar(1e-6)));
//
//        // 2. Update
//        // Out = W_self * X + W_neighbor * Aggr
//        return linRel.forward(aggr).add(linRoot.forward(x));
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