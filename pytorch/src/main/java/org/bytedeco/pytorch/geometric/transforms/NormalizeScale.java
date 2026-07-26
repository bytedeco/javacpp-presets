package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class NormalizeScale implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        // 先中心化
        data.pos = data.pos.sub(data.pos.mean(new long[]{0}, true, new ScalarTypeOptional(kFloat())));
        // 计算最大绝对值
        Tensor max_val = data.pos.abs().max();
        data.pos = data.pos.div(max_val.add(new Scalar(1e-7)));
        return data;
    }
}