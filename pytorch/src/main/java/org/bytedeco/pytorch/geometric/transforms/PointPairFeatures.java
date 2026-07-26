package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class PointPairFeatures implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        if (data.pos == null || data.get("norm") == null) {
            throw new RuntimeException("PointPairFeatures requires both 'pos' and 'norm' attributes.");
        }
        Tensor row = data.edge_index.select(0, 0);
        Tensor col = data.edge_index.select(0, 1);

        Tensor pos_i = data.pos.index_select(0, row);
        Tensor pos_j = data.pos.index_select(0, col);
        Tensor norm_i = data.get("norm").index_select(0, row);
        Tensor norm_j = data.get("norm").index_select(0, col);

        Tensor d = pos_j.sub(pos_i);
        // 1. 距离
        Tensor dist = d.pow(new Scalar(2)).sum(1).sqrt().view(new long[]{-1, 1});
        d = d.div(dist.add(new Scalar(1e-7))); // 归一化向量

        // 2-4. 计算角度 (利用点积)
        Tensor a1 = (norm_i.mul(d)).sum(1).view(new long[]{-1, 1});
        Tensor a2 = (norm_j.mul(d)).sum(1).view(new long[]{-1, 1});
        Tensor a3 = (norm_i.mul(norm_j)).sum(1).view(new long[]{-1, 1});

        Tensor ppf = cat(new TensorVector(dist, a1, a2, a3), 1);
        data.edge_attr = (data.edge_attr == null) ? ppf : cat(new TensorVector(data.edge_attr, ppf), 1);
        return data;
    }
}