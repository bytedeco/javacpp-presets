package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.Scatter;

/**
 * 实现 torch_geometric.nn.conv.GATv2Conv
 * 修复了标准 GAT 的静态注意力问题，使每个节点都能真正关注到任何其他节点。
 */
public class GATv2Conv extends MessagePassing {
    private LinearImpl linSrc, linDst, linEdge;
    private Tensor att; // 注意力向量 a
    private long heads;
    private long outChannels;
    private boolean concat;
    private double negativeSlope;
    private Tensor bias;

    public GATv2Conv(long inChannels, long outChannels, long heads, boolean concat, double negativeSlope, Integer edgeDim, boolean hasBias) {
        super("add");
        this.heads = heads;
        this.outChannels = outChannels;
        this.concat = concat;
        this.negativeSlope = negativeSlope;

        // GATv2 的核心：源节点和目标节点拥有独立的线性变换
        this.linSrc = new LinearImpl(inChannels, heads * outChannels);
        this.linDst = new LinearImpl(inChannels, heads * outChannels);

        // 注意力权重向量 a: [1, heads, outChannels]
        this.att = torch.randn(new long[]{1, heads, outChannels});
        torch.xavier_uniform_(this.att);

        register_module("lin_src", linSrc);
        register_module("lin_dst", linDst);
        register_parameter("att", att);

        if (edgeDim != null) {
            this.linEdge = new LinearImpl(edgeDim, heads * outChannels);
            register_module("lin_edge", linEdge);
        }

        if (hasBias) {
            long biasDim = concat ? heads * outChannels : outChannels;
            this.bias = torch.zeros(new long[]{biasDim});
            register_parameter("bias", bias);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        long N = x.size(0);

        // 1. 分别计算源和目标节点的投影: [N, H, C]
        Tensor xSrc = linSrc.forward(x).view(N, heads, outChannels);
        Tensor xDst = linDst.forward(x).view(N, heads, outChannels);

        // 2. 传播逻辑
        return propagate(edge_index, xSrc, xDst, edge_attr);
    }

    /**
     * 重载 propagate 以支持 GATv2 的双线性输入
     */
    public Tensor propagate(Tensor edge_index, Tensor xSrc, Tensor xDst, Tensor edge_attr) {
        long N = xDst.size(0);
        Tensor sourceIdx = edge_index.select(0, 0);
        Tensor targetIdx = edge_index.select(0, 1);

        // Lift
        Tensor xj = xSrc.index_select(0, sourceIdx);
        Tensor xi = xDst.index_select(0, targetIdx);

        // 计算消息 (包含注意力计算)
        Tensor msg = message_v2(xj, xi, edge_attr, targetIdx, N);

        // 聚合
        Tensor out = aggregate(msg, targetIdx, N);

        // 更新与多头合并
        return update_v2(out);
    }

    private Tensor message_v2(Tensor xj, Tensor xi, Tensor edge_attr, Tensor targetIdx, long numNodes) {
        // 公式: alpha = a^T * LeakyReLU(Wh_i + Wh_j)
        Tensor out = xi.add(xj);

        if (edge_attr != null && linEdge != null) {
            Tensor e = linEdge.forward(edge_attr).view(-1, heads, outChannels);
            out = out.add(e);
        }

        out = torch.leaky_relu(out, new Scalar(negativeSlope));

        // 计算注意力分数: [E, H]
        Tensor alpha = (out.mul(att)).sum(-1);

        // Softmax 归一化
        alpha = scatter_softmax(alpha, targetIdx, numNodes);

        // 加权特征: [E, H, C] * [E, H, 1]
        return xj.mul(alpha.unsqueeze(-1));
    }

    private Tensor update_v2(Tensor out) {
        long N = out.size(0);
        if (concat) {
            out = out.view(N, heads * outChannels);
        } else {
            out = out.mean(1);
        }

        if (bias != null) {
            out = out.add(bias);
        }
        return out;
    }

    private Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        Tensor maxVal = Scatter.scatter(src, index, dimSize, "max");
        Tensor out = src.sub(maxVal.index_select(0, index)).exp();
        Tensor sum = Scatter.scatter(out, index, dimSize, "add");
        return out.div(sum.index_select(0, index).add(new Scalar(1e-16)));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j; // 基类占位
    }
}