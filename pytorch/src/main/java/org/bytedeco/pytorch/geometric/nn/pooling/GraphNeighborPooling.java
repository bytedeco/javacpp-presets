package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class GraphNeighborPooling {

    /**
     * Max Pool Neighbor X
     * 将节点特征替换为其自身及邻域内的最大值
     */
    public static Tensor max_pool_neighbor_x(Tensor x, Tensor edge_index) {
        // 1. 包含自连接 (Central Node + Neighbors)
        Tensor selfLoopedge_index = add_self_loops(edge_index, x.size(0));

        long rowIdx = 0; // source_to_target: 源节点聚合到目标节点
        long colIdx = 1;
        Tensor row = selfLoopedge_index.index_select(0, tensor(new long[]{rowIdx})).view(-1);
        Tensor col = selfLoopedge_index.index_select(0, tensor(new long[]{colIdx})).view(-1);

        // 2. 准备邻居特征
        Tensor msg = x.index_select(0, row); // 获取所有源节点的特征

        // 3. 聚合：对每个目标节点(col) 取其对应的 msg 的最大值
        // 这里简化实现，实际可使用 torch_scatter 库或 torch.amin/amax
        // 我们使用一个简单的循环或 scatter 逻辑
        Tensor out = full(x.sizes(), new Scalar(-1e9f), x.options());
        return scatter_max_impl(out, col, msg);
    }

    /**
     * Avg Pool Neighbor X
     * 将节点特征替换为其自身及邻居的平均值
     */
    public static Tensor avg_pool_neighbor_x(Tensor x, Tensor edge_index) {
        Tensor selfLoopedge_index = add_self_loops(edge_index, x.size(0));

        Tensor row = selfLoopedge_index.index_select(0, tensor(new long[]{0})).view(-1);
        Tensor col = selfLoopedge_index.index_select(0, tensor(new long[]{1})).view(-1);

        Tensor msg = x.index_select(0, row);

        // 聚合：计算均值
        Tensor out = zeros(x.sizes(), x.options());
        return scatter_mean_impl(out, col, msg);
    }

    // 辅助方法：添加自连接
    private static Tensor add_self_loops(Tensor edge_index, long numNodes) {
        Tensor loop = arange(new Scalar(0), new Scalar(numNodes), edge_index.options().dtype(new ScalarTypeOptional(kLong()))).view(1, -1);
        Tensor selfLoops = cat(new TensorVector(loop, loop), 0);
        return cat(new TensorVector(edge_index, selfLoops), 1);
    }

    // 简化的 Scatter 实现逻辑 (实际开发建议集成 torch-scatter)
    private static Tensor scatter_max_impl(Tensor out, Tensor index, Tensor src) {
        // JavaCPP 层面可以通过 index_put_ 或 特化的算子实现
        // 演示目的，此处逻辑代表聚合逻辑
        return out.scatter_reduce(0, index.unsqueeze(-1).expand_as(src), src, "amax", false);
    }

    private static Tensor scatter_mean_impl(Tensor out, Tensor index, Tensor src) {
        return out.scatter_reduce(0, index.unsqueeze(-1).expand_as(src), src, "mean", false);
    }
}
