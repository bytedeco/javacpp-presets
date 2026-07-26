package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;

/**
 * 修正版 FeaStConv（特征驱动注意力图卷积）
 * 核心特性：
 * 1. 多注意力头的特征驱动权重分配
 * 2. 动态注意力分数 + 加权聚合
 * 3. 兼容 PyTorch 梯度计算规则，无原地操作风险
 */
public class FeaStConv extends MessagePassing {
    private int heads; // 注意力头数
    public LinearImpl linWeights; // 多头部权重投影层 [in, heads*out]
    public LinearImpl linSrc;
    public LinearImpl linDst; // 注意力分数计算层 [in, heads]
    public Parameter bias; // 偏置参数（用Parameter管理梯度）
    private long outChannelsPerHead; // 每个头的输出维度

    public FeaStConv(long inChannels, long outChannels, int heads, boolean hasBias) {
        super("add"); // 聚合模式：add
        this.heads = heads;
        this.outChannelsPerHead = outChannels;

        // 1. 初始化线性层（统一配置：Float + CPU，确保设备/类型一致）
        TensorOptions paramOpts = new TensorOptions()
                .dtype(new ScalarTypeOptional(torch.ScalarType.Float))
                .device(new DeviceOptional(new Device(torch.kCPU())));
//                .requires_grad(new BoolOptional(true));

        // 权重投影层：输出维度 = 头数 × 单头输出维度
        this.linWeights = new LinearImpl(inChannels, heads * outChannels);
        // 初始化线性层参数（确保梯度可训练）
        if (hasBias) {
            
        }
        initLinearParams(linWeights, inChannels, heads * outChannels, paramOpts);
        register_module("lin_weights", linWeights);

        // 注意力分数计算层（源/目标节点）
        this.linSrc = new LinearImpl(inChannels, heads);
        initLinearParams(linSrc, inChannels, heads, paramOpts);
        register_module("lin_src", linSrc);

        this.linDst = new LinearImpl(inChannels, heads);
        initLinearParams(linDst, inChannels, heads, paramOpts);
        register_module("lin_dst", linDst);

        // 2. 偏置初始化（用Parameter管理）
        if (hasBias) {
            Tensor biasTensor = torch.zeros(new long[]{outChannels}, paramOpts);
            this.bias = new Parameter(biasTensor);
            register_parameter("bias", this.bias);
        }
    }

    /**
     * 核心前向传播逻辑
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     * @return 输出特征 [N, outChannels]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // ========== 输入校验 ==========
        long[] xShape = x.sizes().vec().get();
        if (xShape.length != 2) {
            throw new IllegalArgumentException("节点特征x必须是2D张量，当前维度：" + xShape.length);
        }
        long N = xShape[0]; // 节点数
        long inChannels = xShape[1];

        long[] edgeIndexShape = edge_index.sizes().vec().get();
        if (edgeIndexShape.length != 2 || edgeIndexShape[0] != 2) {
            throw new IllegalArgumentException("边索引edge_index必须是[2, E]形状，当前：" + edgeIndexShape);
        }
        long E = edgeIndexShape[1]; // 边数

        // ========== 空边场景处理 ==========
        if (E == 0) {
            Tensor out = torch.zeros(new long[]{N, outChannelsPerHead}, x.options());
            if (bias != null) {
                out = out.add(bias);
            }
            return out;
        }

        // ========== 1. 提取源/目标节点索引 ==========
        Tensor sourceIdx = edge_index.select(0, 0); // [E]
        Tensor targetIdx = edge_index.select(0, 1); // [E]

        // ========== 2. 计算注意力分数（特征引导权重） ==========
        // q = linSrc(x_source) + linDst(x_target) → [E, heads]
        Tensor xSource = x.index_select(0, sourceIdx); // [E, inChannels]
        Tensor xTarget = x.index_select(0, targetIdx); // [E, inChannels]
        Tensor q = linSrc.forward(xSource).add(linDst.forward(xTarget)); // [E, heads]

        // Softmax归一化（明确指定dim=-1，避免空张量报错）
        Tensor alpha = torch.softmax(q, -1); // [E, heads]

        // ========== 3. 计算多头部投影特征 ==========
        // xTrans: [N, heads*out] → reshape → [N, heads, out]
        Tensor xTrans = linWeights.forward(x).view(N, heads, outChannelsPerHead); // [N, heads, out]
        Tensor xjTrans = xTrans.index_select(0, sourceIdx); // [E, heads, out]

        // ========== 4. 特征驱动加权聚合 ==========
        // alpha.unsqueeze(-1): [E, heads] → [E, heads, 1]
        // 加权：[E, heads, out] * [E, heads, 1] → [E, heads, out]
        Tensor msg = xjTrans.mul(alpha.unsqueeze(-1));
        // 对头维度求和：[E, heads, out] → [E, out]
        Tensor outMsg = msg.sum(1);

        // ========== 5. 调用父类propagate完成聚合（替换自定义aggregate） ==========
        // 构造空edge_attr，复用MessagePassing的聚合逻辑
        Tensor out = propagate(edge_index, outMsg, targetIdx, N);

        // ========== 6. 加偏置 ==========
        if (bias != null) {
            out = out.add(bias);
        }

        // ========== 资源释放 ==========
        sourceIdx.close();
        targetIdx.close();
        xSource.close();
        xTarget.close();
        q.close();
        alpha.close();
        xTrans.close();
        xjTrans.close();
        msg.close();
        outMsg.close();

        return out;
    }

    /**
     * 覆写MessagePassing的message方法（适配聚合逻辑）
     * @param x_j        消息张量（已加权的输出消息）
     * @return 最终消息张量
     */
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // 核心消息已在forward中计算完成，直接返回
        return x_j;
    }

    /**
     * 复用MessagePassing的propagate方法完成聚合
     * @param edge_index 边索引
     * @param msg        消息张量 [E, out]
     * @param targetIdx  目标节点索引
     * @param N          节点数
     * @return 聚合后的输出 [N, out]
     */
    public Tensor propagate(Tensor edge_index, Tensor msg, Tensor targetIdx, long N) {
        // 初始化输出张量（非原地操作）
        Tensor out = torch.zeros(new long[]{N, outChannelsPerHead}, msg.options());
        // 聚合：scatter_add（非原地操作，避免梯度报错）
        out = out.scatter_add(0, targetIdx.unsqueeze(-1).expand_as(msg), msg);
        return out;
    }

    /**
     * 初始化线性层参数（统一Xavier初始化）
     * @param linear     线性层
     * @param inDim      输入维度
     * @param outDim     输出维度
     * @param paramOpts  参数配置
     */
    private void initLinearParams(LinearImpl linear, long inDim, long outDim, TensorOptions paramOpts) {
        // 权重初始化（非原地操作）
        Tensor weight = torch.empty(new long[]{outDim, inDim}, paramOpts,new MemoryFormatOptional());
        weight = torch.xavier_uniform_(weight);
        linear.weight(new Parameter(weight));

        // 偏置初始化（如果有）
        if (linear.bias() != null) {
            Tensor bias = torch.zeros(new long[]{outDim}, paramOpts);
            linear.bias(new Parameter(bias));
        }
    }

    /**
     * 资源释放（避免内存泄漏）
     */
//    @Override
//    protected void finalize() throws Throwable {
//        try {
//            if (linWeights != null) linWeights.close();
//            if (linSrc != null) linSrc.close();
//            if (linDst != null) linDst.close();
//            if (bias != null) bias.close();
//        } finally {
////            super.finalize();
//        }
//    }

    // ========== Getter方法（测试用） ==========
    public long getOutChannelsPerHead() {
        return outChannelsPerHead;
    }

    public int getHeads() {
        return heads;
    }
}

//public class FeaStConv extends MessagePassing {
//    private int heads; // 注意力头数 M
//    private LinearImpl linWeights; // 包含 M 个线性变换 [M, In, Out]
//    private LinearImpl linSrc, linDst; // 用于计算特征引导权重的线性层
//    private Tensor bias;
//
//    public FeaStConv(long inChannels, long outChannels, int heads, boolean hasBias) {
//        super("add");
//        this.heads = heads;
//
//        // 1. 权重组: 我们一次性定义 M 个 outChannels，稍后 view 成 [M, In, Out]
//        // 也可以理解为 M 个并行的 Linear 层
//        this.linWeights = new LinearImpl(inChannels, heads * outChannels);
//
//        // 2. 特征引导层: 用于计算 q_m(x_i, x_j)
//        this.linSrc = new LinearImpl(inChannels, heads);
//        this.linDst = new LinearImpl(inChannels, heads);
//
//        register_module("lin_weights", linWeights);
//        register_module("lin_src", linSrc);
//        register_module("lin_dst", linDst);
//
//        if (hasBias) {
//            this.bias = torch.zeros(new long[]{outChannels});
//            register_parameter("bias", bias);
//        }
//    }
//
//    @Override
//    public Tensor forward(Tensor x, Tensor edge_index) {
//        long N = x.size(0);
//        long C_out = linWeights.options().out_features().get() / heads;
//
//        // 1. 计算注意力分数 (Steering weights)
//        // q = linSrc(x_j) + linDst(x_i)
//        Tensor sourceIdx = edge_index.select(0, 0);
//        Tensor targetIdx = edge_index.select(0, 1);
//
//        Tensor q = linSrc.forward(x.index_select(0, sourceIdx))
//                .add(linDst.forward(x.index_select(0, targetIdx)));
//
//        // Softmax 归一化 (在 M 个头维度上)
//        Tensor alpha = torch.softmax(q, -1); // [E, heads]
//
//        // 2. 计算变换后的消息
//        // 先计算所有头的投影: [N, heads, C_out]
//        Tensor xTrans = linWeights.forward(x).view(N, heads, C_out);
//        Tensor xjTrans = xTrans.index_select(0, sourceIdx); // [E, heads, C_out]
//
//        // 3. 特征驱动聚合: alpha_m * (W_m * x_j)
//        // [E, heads, 1] * [E, heads, C_out] -> [E, heads, C_out]
//        Tensor msg = xjTrans.mul(alpha.unsqueeze(-1));
//
//        // 对头维度求和，得到最终消息: [E, C_out]
//        Tensor outMsg = msg.sum(1);
//
//        // 4. 聚合到目标节点
//        Tensor out = aggregate(outMsg, targetIdx, N);
//
//        if (bias != null) {
//            out = out.add(bias);
//        }
//
//        return out;
//    }
//
//    @Override
//    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
//        // 逻辑已在 forward 中集成
//        return x_j;
//    }
//}