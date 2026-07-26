package org.bytedeco.pytorch.geometric.transforms;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class FaceToEdge implements BaseTransform {
    private final boolean removeSelfLoops;

    public FaceToEdge(boolean removeSelfLoops) { this.removeSelfLoops = removeSelfLoops; }

    public GraphData apply(GraphData data) {
        Tensor face = data.get("face"); // 形状 [3, num_faces]

        // 1. 提取三角形的三条边
        // e1: [2, num_faces] (顶点0和1)
        Tensor e1 = stack(new TensorVector(face.select(0, 0), face.select(0, 1)), 0);
        // e2: [2, num_faces] (顶点1和2)
        Tensor e2 = stack(new TensorVector(face.select(0, 1), face.select(0, 2)), 0);
        // e3: [2, num_faces] (顶点2和0)
        Tensor e3 = stack(new TensorVector(face.select(0, 2), face.select(0, 0)), 0);

        // 2. 拼接所有边 [2, 3 * num_faces]
        Tensor edgeIndex = cat(new TensorVector(e1, e2, e3), 1);

        // 3. 保证双向（无向图）：拼接反向边 [2, 6 * num_faces]
        edgeIndex = cat(new TensorVector(edgeIndex, edgeIndex.flip(0)), 1);

        // 4. 正确调用 unique_dim 进行全局去重
        // 参数：input, dim=1, sorted=true, return_inverse=false, return_counts=false
        T_TensorTensorTensor_T result = unique_dim(edgeIndex, 1, true, false, false);

        data.edge_index = result.get0(); // 拿到去重后的 [2, E]

        return data;
    }
//    @Override
//    public GraphData apply(GraphData data) {
//        Tensor face = data.get("face"); // [3, num_faces]
//
//        // 提取三条边
//        Tensor e1 = face.index_select(0, tensor(new long[]{0, 1}, face.options())); // (i,j)
//        Tensor e2 = face.index_select(0, tensor(new long[]{1, 2}, face.options())); // (j,k)
//        Tensor e3 = face.index_select(0, tensor(new long[]{2, 0}, face.options())); // (k,i)
//
//        // 拼接并去重
//        Tensor edgeIndex = cat(new TensorVector(e1, e2, e3), 1);
//        // 转换为无向图 (包含反向边)
//        edgeIndex = cat(new TensorVector(edgeIndex, edgeIndex.flip(0)), 1);
//
//        // 调用我们之前的 unique 逻辑去重
//        data.edge_index = unique_consecutive(edgeIndex, true, false, false).get0();
//        return data;
//    }
}
