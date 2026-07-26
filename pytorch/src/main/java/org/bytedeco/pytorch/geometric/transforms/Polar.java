package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class Polar implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor row = data.edge_index.select(0, 0);
        Tensor col = data.edge_index.select(0, 1);

        Tensor v = data.pos.index_select(0, col).sub(data.pos.index_select(0, row));

        // r = sqrt(x^2 + y^2)
        Tensor r = v.pow(new Scalar(2)).sum(1).sqrt();
        // theta = atan2(y, x)
        Tensor theta = atan2(v.select(1, 1), v.select(1, 0));

        // 归一化角度从 [-pi, pi] 到 [0, 1]
        theta = theta.div(new Scalar(2 * Math.PI)).add(new Scalar(0.5));

        Tensor polar = stack(new TensorVector(r, theta), 1);

        data.edge_attr = (data.edge_attr == null) ? polar : cat(new TensorVector(data.edge_attr, polar), 1);
        return data;
    }
}