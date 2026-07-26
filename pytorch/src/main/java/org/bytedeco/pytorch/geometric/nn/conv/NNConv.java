package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.NNConv (Edge-Conditioned Convolution)
 * 利用 MLP 根据边特征动态生成消息传递的权重矩阵。
 */
public class NNConv extends MessagePassing {
    private long inChannels;
    private long outChannels;
    private Module nn; // 动态生成权重的 MLP
    private LinearImpl linRoot;
    private Tensor bias;

    public NNConv(long inChannels, long outChannels, Module nn, String aggr, boolean rootWeight, boolean hasBias) {
        super(aggr);
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.nn = nn;

        register_module("nn", nn);

        if (rootWeight) {
            this.linRoot = new LinearImpl(inChannels, outChannels);
            register_module("lin_root", linRoot);
        }

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
     * @param x [N, inChannels] 节点特征
     * @param edge_index [2, E] 边索引
     * @param edge_attr [E, numEdgeFeatures] 边特征
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (edge_attr == null) {
            throw new IllegalArgumentException("NNConv requires edge_attr to be provided.");
        }

        long N = x.size(0);

        // 1. 消息传递逻辑
        Tensor out = propagate(edge_index, x, edge_attr);

        // 2. 处理中心节点 (Root Weight)
        if (linRoot != null) {
            out = out.add(linRoot.forward(x));
        }

        if (bias != null) {
            out = out.add(bias);
        }

        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j shape: [E, inChannels]
        // edge_attr shape: [E, numEdgeFeatures]

        // 1. 通过 MLP 生成权重矩阵 W(e_ij)
        // MLP 输出维度应为 inChannels * outChannels
        Tensor weight = nn.asSequential().forward(edge_attr);
        weight = weight.view(new long[]{-1, inChannels, outChannels}); // [E, In, Out]

        // 2. 执行矩阵乘法: x_j @ W(e_ij)
        // x_j.unsqueeze(1) -> [E, 1, In]
        // [E, 1, In] @ [E, In, Out] -> [E, 1, Out] -> [E, Out]
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1);
    }
}