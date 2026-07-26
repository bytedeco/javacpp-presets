package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 严格使用 LinearImpl 实现 torch_geometric.nn.conv.GeneralConv
 * 高度模块化的通用 GNN 层，支持注意力、边特征和残差连接。
 */
public class GeneralConv extends MessagePassing {
    private long inChannels, outChannels;
    private int heads;
    private boolean useAttention;

    // 核心线性层
    private LinearImpl linMsg;      // 消息投影 W_msg
    private LinearImpl linEdge;     // 边特征投影 W_edge (可选)
    private LinearImpl linSkip;     // 跳跃连接投影 W_skip (可选)

    // 注意力参数
    private LinearImpl linAtt;      // 用于 additive 注意力
    private Tensor attVector;       // 注意力向量 a

    private boolean l2Normalize;

    public GeneralConv(long inChannels, long outChannels, Integer inEdgeChannels,
                       int heads, boolean attention, String attentionType,
                       boolean skipLinear, boolean l2Normalize, boolean hasBias) {
        super("add");
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.heads = heads;
        this.useAttention = attention;
        this.l2Normalize = l2Normalize;

        // 1. 消息投影层 (LinearImpl)
        this.linMsg = new LinearImpl(inChannels, heads * outChannels);
        register_module("lin_msg", linMsg);

        // 2. 边特征处理 (LinearImpl)
        if (inEdgeChannels != null) {
            this.linEdge = new LinearImpl(inEdgeChannels, heads * outChannels);
            register_module("lin_edge", linEdge);
        }

        // 3. 注意力机制实现 (LinearImpl)
        if (attention) {
            if (attentionType.equals("additive")) {
                this.linAtt = new LinearImpl(outChannels, outChannels);
                register_module("lin_att", linAtt);
                this.attVector = torch.randn(new long[]{1, heads, outChannels});
                register_parameter("att_vector", attVector);
            }
        }

        // 4. 跳跃连接 (LinearImpl)
        if (skipLinear) {
            this.linSkip = new LinearImpl(inChannels, heads * outChannels);
            register_module("lin_skip", linSkip);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        // A. 基础消息准备
        Tensor xMsg = linMsg.forward(x).view(-1, heads, outChannels);

        // B. 消息传递与聚合
        Tensor out = propagate(edge_index, xMsg, edge_attr);

        // C. 跳跃连接 (Skip Connection)
        if (linSkip != null) {
            out = out.add(linSkip.forward(x).view(-1, heads, outChannels));
        }

        // D. L2 归一化
        if (l2Normalize) {
            NormalizeFuncOptions options = new NormalizeFuncOptions();
            options.dim().put(-1);
            options.p().put(2.0);
            options.eps().put(1e-12);
            out = torch.normalize(out, options);
        }

        // 合并多头并输出
        return out.view(-1, heads * outChannels);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        Tensor msg = x_j; // x_j 已经是投影后的 [E, heads, outChannels]

        // 注入边信息
        if (edge_attr != null && linEdge != null) {
            Tensor eMsg = linEdge.forward(edge_attr).view(-1, heads, outChannels);
            msg = msg.add(eMsg);
        }

        // 注意力加权
        if (useAttention && attVector != null) {
            // 简化版 Additive Attention
            Tensor alpha = torch.tanh(msg).mul(attVector).sum(-1);
            alpha = torch.softmax(alpha, 0); // 在邻域内归一化
            msg = msg.mul(alpha.unsqueeze(-1));
        }

        return msg;
    }
}
