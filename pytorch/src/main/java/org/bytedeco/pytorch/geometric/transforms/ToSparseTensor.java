package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.sparse_coo_tensor;

/**
 * ToSparseTensor: 将 edge_index 转换为稀疏矩阵表示 adj_t
 * 在处理超大规模图时，SparseTensor 的内存效率远高于稠密矩阵
 */
public class ToSparseTensor implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        long numNodes = data.x.size(0);
        // 在 JavaCPP-PyTorch 中，我们构造一个标准的 torch.sparse_coo_tensor
        // 形状为 [N, N]
        Tensor values = ones(new long[]{data.edge_index.size(1)}, data.x.options());
        data.put("adj_t", sparse_coo_tensor(data.edge_index, values, new long[]{numNodes, numNodes}));

        // 通常还需要将稀疏矩阵转为 CSR 格式以加速计算
        data.put("adj_t", data.get("adj_t").coalesce());
        return data;
    }
}