package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class LocalCartesian implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor row = data.edge_index.select(0, 0);
        Tensor col = data.edge_index.select(0, 1);

        Tensor pos_i = data.pos.index_select(0, row);
        Tensor pos_j = data.pos.index_select(0, col);

        // 局部化：pos_j 相对于 pos_i 的偏移
        Tensor local_cart = pos_j.sub(pos_i);

        // 缩放：按全局最大位移缩放，保持相对比例
        local_cart = local_cart.div(data.pos.max());

        data.edge_attr = (data.edge_attr == null) ? local_cart : cat(new TensorVector(data.edge_attr, local_cart), 1);
        return data;
    }
}