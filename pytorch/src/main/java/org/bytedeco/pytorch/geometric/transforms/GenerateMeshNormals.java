package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class GenerateMeshNormals implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor face = data.get("face"); // [3, num_faces]
        Tensor pos = data.pos;

        // 计算每个面的法向量 (叉乘)
        Tensor v1 = pos.index_select(0, face.select(0, 0));
        Tensor v2 = pos.index_select(0, face.select(0, 1));
        Tensor v3 = pos.index_select(0, face.select(0, 2));

//        Tensor faceNormals = cross(v2.sub(v1), v3.sub(v1), 1);
        Tensor faceNormals = cross(v2.sub(v1), v3.sub(v1), new LongOptional(1));

        // 累加到各个顶点
        Tensor nodeNormals = zeros_like(pos);
        for (int i = 0; i < 3; i++) {
            nodeNormals.scatter_add_(0, face.select(0, i).unsqueeze(-1).expand_as(faceNormals), faceNormals);
        }

        // 归一化
        data.put("norm", nodeNormals.div(nodeNormals.norm(new ScalarOptional(new Scalar(2)), new long[]{1}, true).add(new Scalar(1e-7))));
        return data;
    }
}