package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.List;
import java.util.ArrayList;

/**
 * 严格使用 LinearImpl 规范实现 torch_geometric.nn.conv.MixHopConv
 * 特点：在一层内混合不同阶数的邻域信息。
 */
public class MixHopConv extends MessagePassing {
    private List<Integer> powers;
    private List<LinearImpl> lins; // 每一阶对应一个独立的 LinearImpl
    private boolean normalize;

    public MixHopConv(long inChannels, long outChannels, List<Integer> powers, boolean normalize) {
        super("add");
//        this.powers = (powers != null) ? powers : List.of(0, 1, 2);
        if (powers == null) {
            this.powers = new ArrayList<>();
            this.powers.add(0);
            this.powers.add(1);
            this.powers.add(2);
        } else {
            this.powers = new ArrayList<>(powers);
        }
        this.normalize = normalize;
        this.lins = new ArrayList<>();

        // 1. 为每个幂次注册独立的 LinearImpl
        for (int i = 0; i < this.powers.size(); i++) {
            LinearImpl lin = new LinearImpl(inChannels, outChannels);
            lins.add(lin);
            register_module("lin_" + i, lin);
        }
    }

    /**
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        long N = x.size(0);
        List<Tensor> outputs = new ArrayList<>();

        // 1. 预计算归一化系数 (D^-0.5 * A * D^-0.5)
        Tensor norm = null;
        if (normalize) {
            norm = compute_normalization(edge_index, N);
        }

        // 2. 针对每一个 power 进行特征提取
        for (int i = 0; i < powers.size(); i++) {
            int p = powers.get(i);
            Tensor x_p = x;

            // 迭代执行 p 次邻域传播，模拟 A^p
            for (int j = 0; j < p; j++) {
                x_p = propagate(edge_index, x_p, norm);
            }

            // 应用该阶数特有的线性变换 (Strictly LinearImpl)
            outputs.add(lins.get(i).forward(x_p));
        }

        // 3. 将所有阶数的输出在特征维度拼接
        return torch.cat(new TensorVector(outputs.toArray(new Tensor[0])), -1);
    }

    private Tensor compute_normalization(Tensor edge_index, long numNodes) {
        Tensor row = edge_index.select(0, 0);
        Tensor col = edge_index.select(0, 1);
        Tensor edge_weight = torch.ones(new long[]{edge_index.size(1)}, edge_index.options());

        Tensor deg = torch.zeros(new long[]{numNodes}, edge_index.options());
        deg.scatter_add_(0, row, edge_weight);

        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0));

        return degInvSqrt.index_select(0, row)
                .mul(degInvSqrt.index_select(0, col));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // edge_attr 存储归一化系数
        return x_j.mul(edge_attr.view(-1, 1));
    }
}