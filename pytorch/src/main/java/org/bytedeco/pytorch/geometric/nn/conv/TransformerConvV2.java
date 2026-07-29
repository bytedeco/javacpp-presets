package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 修复版 TransformerConv（适配 bytedeco-pytorch 原生 API）
 * 核心：修正所有 API 调用错误，完善维度处理，补充异常校验
 */
public class TransformerConvV2 extends MessagePassing {
    private LinearImpl linKey, linQuery, linValue;
    private LinearImpl linSkip;
    private long heads;
    private long outChannels;
    private long inChannels;

    // 构造函数：增加输入校验，初始化所有线性层
    public TransformerConvV2(long inChannels, long outChannels, long heads) {
        super("add"); // 使用加法聚合
        // 输入校验
        if (inChannels <= 0 || outChannels <= 0 || heads <= 0) {
            throw new IllegalArgumentException("通道数和头数必须大于0");
        }
        this.inChannels = inChannels;
        this.heads = heads;
        this.outChannels = outChannels;

        var linearOptions = new LinearOptions(inChannels, heads * outChannels); 
        linearOptions.bias().put(false);
        linKey = new LinearImpl(linearOptions);
        linQuery = new LinearImpl(linearOptions);
        linValue = new LinearImpl(linearOptions);
        linSkip = new LinearImpl(linearOptions);

        // 注册模块（保证梯度传播）
        register_module("lin_query", linQuery);
        register_module("lin_key", linKey);
        register_module("lin_value", linValue);
        register_module("lin_skip", linSkip);
    }

    // 核心前向传播（统一入口）
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // 输入校验
        if (x == null || edge_index == null) {
            throw new NullPointerException("输入张量不能为空");
        }
        if (x.dim() != 2 || edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("输入维度错误：x应为2维，edge_index应为2xN维");
        }

        long numNodes = x.size(0);

        // 1. 计算 Q, K, V [N, in] -> [N, H*C] -> [N, H, C]
        Tensor q = linQuery.forward(x).view(numNodes, heads, outChannels);
        Tensor k = linKey.forward(x).view(numNodes, heads, outChannels);
        Tensor v = linValue.forward(x).view(numNodes, heads, outChannels);

        // 2. 自定义传播逻辑
        Tensor out = propagate_transformer(edge_index, q, k, v, x);

        // 3. 释放临时张量（避免内存泄漏）
        q.close();
        k.close();
        v.close();

        return out;
    }

    // 自定义传播逻辑（处理多张量）
    private Tensor propagate_transformer(Tensor edge_index, Tensor q, Tensor k, Tensor v, Tensor xOriginal) {
        Tensor sourceIdx = edge_index.select(0, 0); // 源节点索引 [E]
        Tensor targetIdx = edge_index.select(0, 1); // 目标节点索引 [E]
        long numNodes = q.size(0);

        // 1. 按索引提取边对应的 Q_i, K_j, V_j [E, H, C]
        Tensor q_i = q.index_select(0, targetIdx);
        Tensor k_j = k.index_select(0, sourceIdx);
        Tensor v_j = v.index_select(0, sourceIdx);

        // 2. 计算注意力消息
        Tensor msg = message_transformer(q_i, k_j, v_j, edge_index, numNodes);

        // 3. 聚合消息（修复 aggregate 实现）
        Tensor out = aggregate(msg, targetIdx, numNodes);

        // 4. 残差连接
        Tensor skip = linSkip.forward(xOriginal).view(numNodes, heads, outChannels);
        out = out.add(skip);

        // 5. 维度变换 [N, H, C] -> [N, H*C]
        out = out.view(numNodes, heads * outChannels);

        // 6. 释放临时张量
        sourceIdx.close();
        targetIdx.close();
        q_i.close();
        k_j.close();
        v_j.close();
        msg.close();
        skip.close();

        return out;
    }

    // 注意力消息计算（修复所有 API 调用）
    private Tensor message_transformer(Tensor q_i, Tensor k_j, Tensor v_j, Tensor edge_index, long numNodes) {
        // 1. 计算注意力分数 (q_i * k_j).sum(-1) / sqrt(C)
        // 修复 sum 调用：指定维度为 long 类型，禁用 keepdim
        Tensor alpha = q_i.mul(k_j).sum(new long[]{-1}, false, new ScalarTypeOptional(torch.kFloat()));
        // 修复 div 调用：使用 Scalar 正确初始化
        alpha = alpha.div(new Scalar(Math.sqrt(outChannels)));

        // 2. 按目标节点做 scatter softmax（核心修复）
        Tensor targetIdx = edge_index.select(0, 1);
        alpha = scatter_softmax(alpha, targetIdx, numNodes);

        // 3. 权重化 V_j：[E, H, C] * [E, H, 1]
        // 修复 unsqueeze：指定维度为 long 类型
        alpha = alpha.unsqueeze(-1);
        Tensor msg = v_j.mul(alpha);

        // 4. 释放临时张量
        targetIdx.close();
        alpha.close();

        return msg;
    }

    // 修复 scatter_softmax 实现（适配 bytedeco-pytorch）
    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        if (src == null || index == null) {
            throw new NullPointerException("src 和 index 不能为空");
        }

        // 1. 按 index 计算每个节点的最大值（数值稳定）
        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
        // 2. 广播最大值到边维度
        Tensor maxValExpanded = maxVal.index_select(0, index);
        // 3. 计算 exp(x - max(x))
        Tensor out = src.sub(maxValExpanded).exp();
        // 4. 按 index 求和
        Tensor sum = Scatter.scatter(out, index, dimSize, "add");
        // 5. 广播求和结果
        Tensor sumExpanded = sum.index_select(0, index);
        // 6. 归一化（加小常数避免除零）
        out = out.div(sumExpanded.add(new Scalar(1e-16)));

        // 7. 释放临时张量
        maxVal.close();
        maxValExpanded.close();
        sum.close();
        sumExpanded.close();

        return out;
    }

    @Override
    public Tensor aggregate(Tensor msg, Tensor index, long dimSize) {
        return Scatter.scatter(msg, index, dimSize, "add");
    }

    // 基类空实现（防止 NPE）
    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        return inputs;
    }

    // 释放资源
    public void close() {
        if (linKey != null) linKey.close();
        if (linQuery != null) linQuery.close();
        if (linValue != null) linValue.close();
        if (linSkip != null) linSkip.close();
    }
}

