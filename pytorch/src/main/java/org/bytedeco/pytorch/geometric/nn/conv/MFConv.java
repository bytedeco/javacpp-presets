package org.bytedeco.pytorch.geometric.nn.conv;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * 实现 torch_geometric.nn.conv.MFConv
 * 根据节点的入度选择不同的线性变换权重。
 */
public class MFConv extends MessagePassing {
    private LinearImpl[] lins; // 为每个可能的度分配一个线性层
    private int maxDegree;
    private Tensor bias;

    public MFConv(long inChannels, long outChannels, int maxDegree, boolean hasBias) {
        super("add");
        this.maxDegree = maxDegree;

        // 初始化权重列表：从 degree=0 到 degree=maxDegree
        this.lins = new LinearImpl[maxDegree + 1];
        for (int i = 0; i <= maxDegree; i++) {
            lins[i] = new LinearImpl(inChannels, outChannels);
            register_module("lin_" + i, lins[i]);
        }

        if (hasBias) {
            this.bias = torch.zeros(new long[]{outChannels});
            register_parameter("bias", bias);
        }
    }

    /**
     * @param x          节点特征 [N, inChannels]
     * @param edge_index 边索引 [2, E]
     */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        long N = x.size(0);

        // 1. 邻居聚合：这里直接使用加法聚合邻居特征
        Tensor aggregated = propagate(edge_index, x, new long[]{x.size(0), x.size(0)});

        // 2. 计算每个节点的度 (In-degree)
        // row: source, col: target. 聚合到 target，所以计算的是 target 的度。
        Tensor targetIdx = edge_index.select(0, 1);
        Tensor deg = torch.zeros(new long[]{N}, edge_index.options());
        deg.scatter_add_(0, targetIdx, torch.ones(new long[]{targetIdx.size(0)}, edge_index.options()));

        // 3. 根据度进行分类处理 (Degree-specific transformation)
        // 生产环境优化：使用 mask 批量处理相同的度，而不是循环每个节点
        Tensor out = torch.zeros(new long[]{N, lins[0].options().out_features().get()}, x.options());

        for (int d = 0; d <= maxDegree; d++) {
            // 找到所有度为 d 的节点索引
            Tensor mask = deg.eq(new Scalar(d));
            if (mask.any().item_bool()) {
                // 提取这些节点的特征并应用对应的线性层
                Tensor nodesWithDegreeD = aggregated.masked_select(mask.unsqueeze(-1)).view(-1, x.size(1));
                Tensor transformed = lins[d].forward(nodesWithDegreeD);

                // 将结果写回 (由于 masked_scatter 在某些版本有性能限制，我们手动处理索引)
                out.masked_scatter_(mask.unsqueeze(-1), transformed);
            }
        }

        // 处理超过最大度的节点：统一使用最后一层权重
        Tensor maskOver = deg.gt(new Scalar(maxDegree));
        if (maskOver.any().item_bool()) {
            Tensor nodesOver = aggregated.masked_select(maskOver.unsqueeze(-1)).view(-1, x.size(1));
            out.masked_scatter_(maskOver.unsqueeze(-1), lins[maxDegree].forward(nodesOver));
        }

        if (bias != null) {
            out = out.add(bias);
        }

        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }
}