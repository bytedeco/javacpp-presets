package org.bytedeco.pytorch.geometric.transforms;

//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.opencv.opencv_imgproc.*;
//import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class Delaunay implements BaseTransform {

    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = data.pos; // [N, D]
        long numNodes = pos.size(0);

        // 1. 计算所有点对之间的距离平方矩阵 [N, N]
        // dist_mat_ij = ||pos_i - pos_j||^2
        Tensor distMat = pos.unsqueeze(1).sub(pos.unsqueeze(0)).pow(new Scalar(2)).sum(2);

        // 2. 找到每个点对 (i, j) 的中点坐标 [N, N, D]
        Tensor midPoints = pos.unsqueeze(1).add(pos.unsqueeze(0)).div(new Scalar(2.0));

        // 3. 计算所有点 k 到每条潜在边 (i, j) 中点的距离平方 [N, N, N]
        // 这里使用广播机制：(N, N, 1, D) - (1, 1, N, D) -> (N, N, N, D)
        Tensor distToMid = midPoints.unsqueeze(2).sub(pos.unsqueeze(0).unsqueeze(0))
                .pow(new Scalar(2)).sum(3);

        // 4. 计算每条潜在边 (i, j) 的半径平方 (距离的一半的平方)
        Tensor radiusSq = distMat.div(new Scalar(4.0));

        // 5. Gabriel Graph 判断标准：
        // 对于边 (i, j)，如果不存在任何点 k 使得 distToMid(i,j,k) < radiusSq(i,j)
        // 注意排除 i 和 j 自身
        Tensor isWithin = distToMid.lt(radiusSq.unsqueeze(2).sub(new Scalar(1e-5)));

        // 统计有多少个点在圆内
        Tensor countWithin = isWithin.sum(2);

        // 如果 countWithin == 0，则该边有效
        Tensor adjMask = countWithin.eq(new Scalar(0));

        // 排除自环
        adjMask.fill_diagonal_(new Scalar(0)).to(ScalarType.Bool);

        // 6. 将邻接矩阵转换为 edge_index [2, E]
        Tensor edgeIndex = adjMask.nonzero().t().to(kLong());

        data.edge_index = edgeIndex;
        return data;
    }
//    public GraphData apply(GraphData data) {
//        Tensor pos = data.pos.cpu(); // 必须在 CPU 上处理
//        float[] coords = (float[]) pos.getValues(); // 假设是 Float32
//        int numPoints = (int) pos.size(0);
//
//        // 1. 定义边界矩形 (OpenCV Subdiv2D 需要)
//        float minX = pos.select(1, 0).min().item_float();
//        float maxX = pos.select(1, 0).max().item_float();
//        float minY = pos.select(1, 1).min().item_float();
//        float maxY = pos.select(1, 1).max().item_float();
//        Rect rect = new Rect((int)minX - 1, (int)minY - 1, (int)(maxX - minX) + 2, (int)(maxY - minY) + 2);
//
//        Subdiv2D subdiv = new Subdiv2D(rect);
//
//        // 2. 插入点并建立旧索引到 Subdiv 内部 ID 的映射
//        for (int i = 0; i < numPoints; i++) {
//            subdiv.insert(new Point2f(coords[i * 2], coords[i * 2 + 1]));
//        }
//
//        // 3. 提取三角形
//        FloatVector triangles = new FloatVector();
//        subdiv.getTriangleList(triangles);
//
//        // 4. 将三角形顶点坐标转换回节点索引 (通过最近邻或哈希映射)
//        // 注意：这步通常比较复杂，需要将 OpenCV 返回的坐标匹配回原始 data.pos 的索引
//        // 匹配成功后，构造 face 属性 [3, num_faces]
//        // data.put("face", faceTensor);
//
//        return data;
//    }
}