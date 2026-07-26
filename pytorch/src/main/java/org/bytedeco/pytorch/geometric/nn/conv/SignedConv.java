package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.SignedConv
 * 处理带有正负边属性的图卷积，通过两套权重捕捉结构平衡。
 */
public class SignedConv extends MessagePassing {
    private long inChannels, outChannels;
    private boolean firstAggr;
    private LinearImpl linPos, linNeg;
    private Tensor biasPos, biasNeg;

    public SignedConv(long inChannels, long outChannels, boolean firstAggr, boolean hasBias) {
        super("mean"); // 使用均值聚合
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.firstAggr = firstAggr;

        // 正向与负向变换矩阵
        // 如果是第一层，inChannels 是原始维度；否则是拼接后的 2 * inChannels
        long actualIn = firstAggr ? inChannels : inChannels * 2;
        this.linPos = new LinearImpl(actualIn *2, outChannels);
        this.linNeg = new LinearImpl(actualIn * 2, outChannels);

        register_module("lin_pos", linPos);
        register_module("lin_neg", linNeg);

        if (hasBias) {
            this.biasPos = torch.zeros(new long[]{outChannels});
            this.biasNeg = torch.zeros(new long[]{outChannels});
            register_parameter("bias_pos", biasPos);
            register_parameter("bias_neg", biasNeg);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    /**
     * @param x              节点特征
     * @param pos_edge_index 正边索引 [2, E_pos]
     * @param neg_edge_index 负边索引 [2, E_neg]
     */
    public Tensor forward(Tensor x, Tensor pos_edge_index, Tensor neg_edge_index) {
        if (firstAggr) {
            // --- 第一层聚合逻辑 ---
            // 1. 聚合正向邻居
            Tensor outPos = propagate(pos_edge_index, x, new long[]{x.size(0), x.size(0)});
            outPos = torch.cat(new TensorVector(outPos, x), -1);
            outPos = linPos.forward(outPos);

            // 2. 聚合负向邻居
            Tensor outNeg = propagate(neg_edge_index, x, new long[]{x.size(0), x.size(0)});
            outNeg = torch.cat(new TensorVector(outNeg, x), -1);
            outNeg = linNeg.forward(outNeg);

            // 返回拼接结果 [N, 2 * outChannels]
            Tensor res = torch.cat(new TensorVector(outPos, outNeg), -1);
            return applyBias(res);
        } else {
            // --- 后续层聚合逻辑 ---
            // x 被分为 x_pos 和 x_neg
            Tensor xPos = x.narrow(-1, 0, inChannels);
            Tensor xNeg = x.narrow(-1, inChannels, inChannels);

            // 1. 正向输出: 聚合(pos_edge 的 x_pos) + 聚合(neg_edge 的 x_neg)
            Tensor a = propagate(pos_edge_index, xPos, new long[]{x.size(0), x.size(0)});
            Tensor b = propagate(neg_edge_index, xNeg, new long[]{x.size(0), x.size(0)});
            Tensor outPos = torch.cat(new TensorVector(a, b, xPos), -1);
            outPos = linPos.forward(outPos);

            // 2. 负向输出: 聚合(pos_edge 的 x_neg) + 聚合(neg_edge 的 x_pos)
            Tensor c = propagate(pos_edge_index, xNeg, new long[]{x.size(0), x.size(0)});
            Tensor d = propagate(neg_edge_index, xPos, new long[]{x.size(0), x.size(0)});
            Tensor outNeg = torch.cat(new TensorVector(c, d, xNeg), -1);
            outNeg = linNeg.forward(outNeg);

            Tensor res = torch.cat(new TensorVector(outPos, outNeg), -1);
            return applyBias(res);
        }
    }

    private Tensor applyBias(Tensor x) {
        if (biasPos == null) return x;
        // 偏置也需要分为两段应用
        Tensor b = torch.cat(new TensorVector(biasPos, biasNeg), -1);
        return x.add(b);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j; // 简单的均值聚合消息
    }
}