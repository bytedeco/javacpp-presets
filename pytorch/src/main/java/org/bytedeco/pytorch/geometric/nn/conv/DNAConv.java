package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

/**
 * 动态邻居聚合卷积层（DNAConv）
 * 核心逻辑：跨层注意力机制 + 图拓扑平滑，输入维度 [N, L, C]
 * N: 节点数, L: 层数, C: 通道数
 */
public class DNAConv extends MessagePassing {
    private long channels;          // 输入/输出通道数
    private int heads;              // 注意力头数
    private int groups;             // 分组数
    public LinearImpl linQ;
    public LinearImpl linK;
    private LinearImpl linV; // Q/K/V 投影层
    private Parameter bias;         // 偏置参数（使用Parameter封装，支持训练）
    private long d_k;               // 每个注意力头的维度 = channels / heads

    /**
     * 构造方法：初始化DNAConv层
     * @param channels 输入/输出通道数（必须能被 heads 整除）
     * @param heads 注意力头数
     * @param groups 分组数
     * @param hasBias 是否使用偏置
     */
    public DNAConv(long channels, int heads, int groups, boolean hasBias) {
        super("add"); // 初始化MessagePassing基类，聚合方式为add
        // 输入校验
        if (channels % heads != 0) {
            throw new IllegalArgumentException("channels 必须能被 heads 整除：channels=" + channels + ", heads=" + heads);
        }
        if (heads <= 0 || groups <= 0) {
            throw new IllegalArgumentException("heads 和 groups 必须大于0：heads=" + heads + ", groups=" + groups);
        }

        this.channels = channels;
        this.heads = heads;
        this.groups = groups;
        this.d_k = channels / heads; // 每个注意力头的维度

        // 初始化Q/K/V线性层：组投影，输出维度=heads*d_k=channels
        this.linQ = new LinearImpl(channels, heads * this.d_k);
        this.linK = new LinearImpl(channels, heads * this.d_k);
        this.linV = new LinearImpl(channels, heads * this.d_k);

        // 注册模块到基类（支持参数优化）
        register_module("lin_q", linQ);
        register_module("lin_k", linK);
        register_module("lin_v", linV);

        // 初始化并注册偏置参数
        if (hasBias) {
            Tensor biasTensor = torch.zeros(new long[]{channels}).to(torch.kFloat());
            this.bias = new Parameter(biasTensor); // 用Parameter封装，支持梯度更新
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    /**
     * 前向传播（核心逻辑）
     * @param x [N, L, C] - 节点数N，层数L，通道数C
     * @param edge_index [2, E] - 边索引
     * @param edge_weight 边权重 [E]（可选，用于GCN式归一化）
     * @return 聚合后特征 [N, C]
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        // ========== 输入校验 ==========
        if (x.dim() != 3) {
            throw new IllegalArgumentException("x 必须是3维张量 [N, L, C]，当前维度：" + x.dim());
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index 必须是 [2, E] 张量，当前维度：" + edge_index.dim());
        }
        long N = x.size(0); // 节点数
        long L = x.size(1); // 层数
        long C = x.size(2); // 通道数
        if (C != this.channels) {
            throw new IllegalArgumentException("x 通道数不匹配：期望 " + this.channels + "，实际 " + C);
        }

        // ========== 设备/类型对齐 ==========
        this.linQ.to(x.device(), x.scalar_type(),false);
        this.linK.to(x.device(), x.scalar_type(),false);
        this.linV.to(x.device(), x.scalar_type(),false);
        if (this.bias != null) {
            this.bias.to(x.device(), x.scalar_type());
        }

        // ========== 1. Q/K/V投影 ==========
        // 展平 [N, L, C] → [N*L, C]，投影后还原维度
        Tensor x_flat = x.view(-1, C);
        Tensor Q_flat = linQ.forward(x_flat);
        Tensor K_flat = linK.forward(x_flat);
        Tensor V_flat = linV.forward(x_flat);

        // 重塑为 [N, L, heads, d_k]
        Tensor Q = Q_flat.view(N, L, heads, this.d_k);
        Tensor K = K_flat.view(N, L, heads, this.d_k);
        Tensor V = V_flat.view(N, L, heads, this.d_k);

        // ========== 2. 图拓扑平滑（跨层聚合） ==========
        Tensor K_hat = torch.zeros_like(K);
        Tensor V_hat = torch.zeros_like(V);
        // 对每一层特征分别进行传播
        for (int l = 0; l < L; l++) {
            Tensor K_l = K.select(1, l); // [N, heads, d_k]
            Tensor V_l = V.select(1, l); // [N, heads, d_k]
            // 调用propagate进行邻居聚合（补全numNodes参数）
            Tensor K_hat_l = propagate(edge_index, K_l, K_l, edge_weight, N);
            Tensor V_hat_l = propagate(edge_index, V_l, V_l, edge_weight, N);
            // 将聚合结果赋值到对应层
            K_hat.select(1, l).copy_(K_hat_l);
            V_hat.select(1, l).copy_(V_hat_l);
            // 释放临时张量
            K_l.close();
            V_l.close();
            K_hat_l.close();
            V_hat_l.close();
        }

        // ========== 3. 动态跨层注意力计算 ==========
        // Query取最后一层特征：[N, heads, d_k] → [N, 1, heads, d_k]
        Tensor query = Q.select(1, (int) (L - 1)).unsqueeze(1);
        // 点积注意力得分：[N,1,H,d] * [N,L,H,d] → sum(d) → [N,L,H]
        Tensor attn = query.mul(K_hat).sum(-1);
        // 缩放注意力得分（除以√d_k）
        attn = attn.div(new Scalar(Math.sqrt(this.d_k)));
        // 对层维度（dim=1）做softmax归一化
        attn = torch.softmax(attn, 1);

        // ========== 4. 加权聚合Value ==========
        // [N,L,H] → [N,L,H,1] * [N,L,H,d] → sum(L) → [N,H,d]
        Tensor out = attn.unsqueeze(-1).mul(V_hat).sum(1);
        // 重塑为 [N, H*d] = [N, C]
        Tensor res = out.view(N, this.channels);

        // ========== 5. 加偏置 ==========
        if (this.bias != null) {
            res = res.add(this.bias.data());
        }

        // ========== 释放所有临时张量 ==========
        x_flat.close();
        Q_flat.close();
        K_flat.close();
        V_flat.close();
        Q.close();
        K.close();
        V.close();
        K_hat.close();
        V_hat.close();
        query.close();
        attn.close();
        out.close();

        return res;
    }

    /**
     * 简化forward调用（无edge_weight）
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return ((DNAConv)this).forward(x, edge_index, (Tensor)null);
    }

    /**
     * MessagePassing的message方法：处理边权重
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j: [E, heads, d_k]，edge_attr: [E]（边权重）
        if (edge_attr != null) {
            // 边权重广播到 [E,1,1]，与x_j相乘
            return x_j.mul(edge_attr.view(-1, 1, 1));
        }
        return x_j;
    }

    /**
     * 重载propagate方法：补全numNodes参数
     */
    public Tensor propagate(Tensor edge_index, Tensor x, Tensor edge_weight, long numNodes) {
        return super.propagate(edge_index, x, edge_weight, numNodes);
    }

    /**
     * 重置参数（支持训练初始化）
     */
//    @Override
    public void reset_parameters() {
        if (linQ != null) linQ.reset_parameters();
        if (linK != null) linK.reset_parameters();
        if (linV != null) linV.reset_parameters();
        if (bias != null) {
            torch.zeros_(bias.data());
        }
    }

    /**
     * 资源释放：避免JNI内存泄漏
     */
    @Override
    public void close() {
        if (linQ != null) {
            linQ.close();
            linQ = null;
        }
        if (linK != null) {
            linK.close();
            linK = null;
        }
        if (linV != null) {
            linV.close();
            linV = null;
        }
        if (bias != null) {
//            bias.close();
            bias = null;
        }
        super.close();
    }

    // 辅助方法（测试用）
    public long getChannels() { return channels; }
    public int getHeads() { return heads; }
    public long getDk() { return d_k; }
}

//package org.bytedeco.pytorch.geometric.nn.conv;
//
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.global.torch;
//
///**
// * 实现 torch_geometric.nn.conv.DNAConv
// * 动态邻居聚合算子，支持跨层注意力机制。
// * 输入维度预期为 [numNodes, numLayers, channels]
// */
//public class DNAConv extends MessagePassing {
//    private long channels;
//    private int heads;
//    private int groups;
//    private LinearImpl linQ, linK, linV;
//    private Tensor bias;
//
//    public DNAConv(long channels, int heads, int groups, boolean hasBias) {
//        super("add");
//        this.channels = channels;
//        this.heads = heads;
//        this.groups = groups;
//
//        // DNA 使用组投影 (Grouped Projections)
//        // 实际上是处理 [N, L, C] 的注意力
//        this.linQ = new LinearImpl(channels, heads * (channels / heads));
//        this.linK = new LinearImpl(channels, heads * (channels / heads));
//        this.linV = new LinearImpl(channels, heads * (channels / heads));
//
//        register_module("lin_q", linQ);
//        register_module("lin_k", linK);
//        register_module("lin_v", linV);
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{channels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        return forward(x, edge_index, null);
//    }
//    /**
//     * @param x [N, L, C] - N个节点，L层特征，每层C个通道
//     * @param edge_index [2, E]
//     * @param edge_weight 归一化系数 (类似于 GCN)
//     */
//    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
//        long N = x.size(0);
//        long L = x.size(1);
//        long C = x.size(2);
//
//        // 1. 投影 Query, Key, Value
//        // 将 [N, L, C] 展平处理再还原
//        Tensor x_flat = x.view(-1, C);
//        Tensor Q = linQ.forward(x_flat).view(N, L, heads, -1);
//        Tensor K = linK.forward(x_flat).view(N, L, heads, -1);
//        Tensor V = linV.forward(x_flat).view(N, L, heads, -1);
//
//        // 2. 邻居特征聚合 (使用类似 GCN 的非参数化传播)
//        // 这里的 K_hat 和 V_hat 是经过图拓扑平滑后的信息
//        // 我们需要对每一层特征分别进行 propagate
//        Tensor K_hat = torch.zeros_like(K);
//        Tensor V_hat = torch.zeros_like(V);
//
//        for (int l = 0; l < L; l++) {
//            K_hat.select(1, l).copy_(propagate(edge_index, K.select(1, l), edge_weight));
//            V_hat.select(1, l).copy_(propagate(edge_index, V.select(1, l), edge_weight));
//        }
//
//        // 3. 计算动态注意力 (Dynamic Attention)
//        // Query 取自当前层（通常是最后一层 L-1）
//        Tensor query = Q.select(1, L - 1).unsqueeze(1); // [N, 1, heads, d]
//
//        // 计算点积得分: [N, 1, heads, d] * [N, L, heads, d] -> [N, L, heads]
//        Tensor attn = (query.mul(K_hat)).sum(-1);
//        attn = attn.div(new Scalar(Math.sqrt(channels / (double)heads)));
//        attn = torch.softmax(attn, 1); // 对层维度 L 做归一化
//
//        // 4. 加权聚合 Value: [N, L, heads, 1] * [N, L, heads, d] -> [N, heads, d]
//        Tensor out = (attn.unsqueeze(-1).mul(V_hat)).sum(1);
//
//        // 5. 还原维度
//        Tensor res = out.view(N, channels);
//
//        if (bias != null) {
//            res = res.add(bias);
//        }
//
//        return res;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        if (edge_attr != null) {
//            return x_j.mul(edge_attr.view(-1, 1, 1));
//        }
//        return x_j;
//    }
//}