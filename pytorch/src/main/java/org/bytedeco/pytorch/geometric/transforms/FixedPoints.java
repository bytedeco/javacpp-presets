package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class FixedPoints implements BaseTransform {
    private final int numPoints;

    public FixedPoints(int numPoints) { this.numPoints = numPoints; }

    @Override
    public GraphData apply(GraphData data) {
        long N = data.numNodes();
        Tensor idx;
        if (N >= numPoints) {
            idx = randperm(N, data.x.options().dtype(new ScalarTypeOptional(kLong()))).slice(0, new LongOptional(0), new LongOptional(numPoints),1);
        } else {
            // 点数不够，随机重复填充
            idx = randint(0, N, new long[]{numPoints}, data.x.options().dtype(new ScalarTypeOptional(kLong())));
        }

        data.x = data.x.index_select(0, idx);
        data.pos = data.pos.index_select(0, idx);
        // 注意：edge_index 在此变换后通常失效，建议重新构建 KNN
        data.edge_index = null;
        return data;
    }
}