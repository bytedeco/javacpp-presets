package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;
/**
 * AddSelfLoops: 为图添加自连接 (i, i)
 */
public class AddSelfLoops implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        long numNodes = data.x.size(0);
        // 创建 [0, 1, ..., N-1]
        Tensor loop = arange(new Scalar(0), new Scalar(numNodes), data.edge_index.options());
        // 构造 [2, N] 的自循环边
        Tensor edgeLoop = cat(new TensorVector(loop.view(1, -1), loop.view(1, -1)), 0);
        // 合并原有边
        data.edge_index = cat(new TensorVector(data.edge_index, edgeLoop), 1);
        return data;
    }
}