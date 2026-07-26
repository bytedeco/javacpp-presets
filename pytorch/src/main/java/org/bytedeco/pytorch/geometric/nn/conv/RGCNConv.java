package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.RGCNConv
 * 关系图卷积算子，支持多种边类型。
 */
public class RGCNConv extends MessagePassing {
    private long inChannels;
    private long outChannels;
    private int numRelations;
    private LinearImpl[] lins; // 为每种关系分配一个 W_r
    private LinearImpl linRoot; // 自环/根节点权重 W_root
    private Tensor bias;

    public RGCNConv(long inChannels, long outChannels, int numRelations, boolean rootWeight, boolean hasBias) {
        super("sum"); // 聚合通常使用 sum，遵循论文公式
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.numRelations = numRelations;

        // 1. 初始化关系特定的线性层 [numRelations, inChannels, outChannels]
        this.lins = new LinearImpl[numRelations];
        for (int i = 0; i < numRelations; i++) {
            lins[i] = new LinearImpl(inChannels, outChannels);
            register_module("lin_rel_" + i, lins[i]);
        }

        // 2. 根节点权重
        if (rootWeight) {
            this.linRoot = new LinearImpl(inChannels, outChannels);
            register_module("lin_root", linRoot);
        }

        // 3. 偏置
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
     * @param x         节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     * @param edge_type  关系类型 [E]，范围 [0, numRelations - 1]
     */
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_type) {
        long N = x.size(0);
        Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());

        // 核心逻辑：按关系类型迭代，实现显存效率最大化
        for (int r = 0; r < numRelations; r++) {
            // 找到所有属于关系 r 的边索引
            Tensor mask = edge_type.eq(new Scalar(r));
            if (mask.any().item_bool()) {
                // 提取局部 edge_index [2, E_r]
                Tensor edge_index_r = edge_index.masked_select(mask.unsqueeze(0).expand(new long[]{2, edge_type.size(0)}))
                        .view(2, -1);

                // 执行该关系的消息传递: W_r * aggregated(x_j)
                Tensor res = propagate(edge_index_r, x, new long[]{x.size(0), x.size(0)});
                out = out.add(lins[r].forward(res));
            }
        }

        // 加上根节点自身的变换
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
        return x_j; // 简单的聚合，变换已在 forward 中完成
    }
}