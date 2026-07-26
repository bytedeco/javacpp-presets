package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class Center implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        // 计算 mean，形状为 [1, D]
        Tensor mean = data.pos.mean(new long[]{0}, true,new ScalarTypeOptional(kFloat()));
        data.pos = data.pos.sub(mean);
        return data;
    }
}
