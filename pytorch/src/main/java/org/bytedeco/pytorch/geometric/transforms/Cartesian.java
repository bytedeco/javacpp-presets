package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class Cartesian implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor row = data.edge_index.select(0, 0);
        Tensor col = data.edge_index.select(0, 1);

        Tensor pos_i = data.pos.index_select(0, row);
        Tensor pos_j = data.pos.index_select(0, col);

        // 计算相对偏移 [E, D]
        Tensor cart = pos_j.sub(pos_i);

        // 归一化到 [0, 1] 之间 (通常用于处理图像像素坐标)
        cart = cart.div(data.pos.max().mul(new Scalar(2))).add(new Scalar(0.5));

        data.edge_attr = (data.edge_attr == null) ? cart : cat(new TensorVector(data.edge_attr, cart), 1);
        return data;
    }
}