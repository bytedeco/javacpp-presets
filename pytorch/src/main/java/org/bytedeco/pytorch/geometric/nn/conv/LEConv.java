package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.LEConv
 * 特点：利用差分算子识别局部极值，是 ASAP 池化层的基础。
 */
public class LEConv extends MessagePassing {
    private LinearImpl lin1, lin2;
    private Tensor bias;

    public LEConv(long inChannels, long outChannels, boolean hasBias) {
        super("add");

        // lin1 处理中心节点特征 x_i
        this.lin1 = new LinearImpl(inChannels, outChannels);
        // lin2 处理邻居差分特征 (x_i - x_j)
        this.lin2 = new LinearImpl(inChannels, outChannels);

        register_module("lin1", lin1);
        register_module("lin2", lin2);

        if (hasBias) {
            this.bias = torch.zeros(new long[]{outChannels});
            register_parameter("bias", bias);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor)null);
    }
    
    /**
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     * @param edge_weight 边权重 [E] (默认为 1.0)
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        long N = x.size(0);

        // 1. 邻居差分聚合：sum_j w_ij * (x_i - x_j)
        // 这一步在 propagate 内部通过 x_i - x_j 实现
        Tensor out = propagate(edge_index, x, edge_weight);

        // 2. 最终公式: x_i' = lin1(x_i) + lin2(out)
        Tensor res = lin1.forward(x).add(lin2.forward(out));

        if (bias != null) {
            res = res.add(bias);
        }

        return res;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_i 是目标节点，x_j 是源节点
        // 计算差值: (x_i - x_j)
        Tensor msg = x_i.sub(x_j);

        // 如果存在边权重 w_ij，进行缩放
        if (edge_attr != null) {
            msg = msg.mul(edge_attr.view(-1, 1));
        }

        return msg;
    }
}