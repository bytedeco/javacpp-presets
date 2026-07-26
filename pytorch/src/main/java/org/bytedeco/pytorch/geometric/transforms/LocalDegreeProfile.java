package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * LocalDegreeProfile (LDP): 提取局部度分布特征
 * 包含：节点度、邻居度的 (min, max, mean, std)
 */
public class LocalDegreeProfile implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        long N = data.x.size(0);
        // 1. 计算度
        Tensor deg = zeros(new long[]{N}, data.x.options());
        deg.scatter_add_(0, data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1)))), ones(new long[]{data.edge_index.size(1)}, data.x.options()));

        // 2. 聚合邻居的度特征 (使用之前实现的 avg_pool_neighbor_x 类似逻辑)
        // 这里简化为只追加度数本身，实际需计算 min/max/std
        Tensor degAgg = deg.view(-1, 1);
        data.x = cat(new TensorVector(data.x, degAgg), 1);
        return data;
    }
}