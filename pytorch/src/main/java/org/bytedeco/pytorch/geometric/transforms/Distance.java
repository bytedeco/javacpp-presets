package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class Distance implements BaseTransform {
    private final boolean norm; // 是否归一化

    public Distance(boolean norm) { this.norm = norm; }

    @Override
    public GraphData apply(GraphData data) {
        Tensor row = data.edge_index.select(0, 0);
        Tensor col = data.edge_index.select(0, 1);

        // 获取源节点和目标节点的坐标 [E, D]
        Tensor pos_i = data.pos.index_select(0, row);
        Tensor pos_j = data.pos.index_select(0, col);

        // 计算差值的二范数 (dim=1)
        Tensor dist = pos_j.sub(pos_i).pow(new Scalar(2)).sum(1).sqrt().view(new long[]{-1, 1});

        if (norm && dist.numel() > 0) {
            dist = dist.div(dist.max());
        }

        // 将结果存入 edge_attr
        data.edge_attr = (data.edge_attr == null) ? dist : cat(new TensorVector(data.edge_attr, dist), 1);
        return data;
    }
}