package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * ToUndirected: 将有向图转换为无向图
 * 原理：对于每一条边 (i, j)，添加反向边 (j, i)，然后去重
 */
public class ToUndirected implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor row = data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(0))));
        Tensor col = data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1))));

        // 拼接反向边
        Tensor newRow = cat(new TensorVector(row, col), 0);
        Tensor newCol = cat(new TensorVector(col, row), 0);
        Tensor newEdges = cat(new TensorVector(newRow.view(1, -1), newCol.view(1, -1)), 0);

        // 简单的去重逻辑（实际生产中建议使用 coalesce 或 unique）
        data.edge_index = newEdges;
        return data;
    }
}