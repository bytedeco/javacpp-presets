package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public  class ToSLIC implements BaseTransform {
    private final int k; // 预期的超像素数量
    private final float m; // 紧凑度参数 (平衡颜色和空间距离)

    public ToSLIC(int k, float m) {
        this.k = k;
        this.m = m;
    }

    @Override
    public GraphData apply(GraphData data) {
        // 假设输入 data.x 是图像像素 [N, 3], data.pos 是像素坐标 [N, 2]
        long numPixels = data.x.size(0);
        long numFeatures = data.x.size(1); // 颜色维度
        float s = (float) Math.sqrt(numPixels / (double) k); // 步长

        // 1. 初始化聚类中心 (均匀采样)
        // 简单起见，这里直接随机挑选 k 个点作为初始中心
        Tensor indices = randperm(numPixels, data.x.options().dtype(new ScalarTypeOptional(kLong()))).slice(0,new LongOptional(0) , new LongOptional(k),1);
        Tensor centersPos = data.pos.index_select(0, indices); // [k, 2]
        Tensor centersCol = data.x.index_select(0, indices);   // [k, 3]

        Tensor labels = zeros(new long[]{numPixels}, data.x.options().dtype(new ScalarTypeOptional(kLong())));

        // 2. 迭代优化 (SLIC 核心迭代)
        for (int iter = 0; iter < 5; iter++) {
            // 计算每个像素到所有中心的 5D 距离
            // dist = sqrt( ||col_i - col_j||^2 + (m/s)^2 * ||pos_i - pos_j||^2 )

            // 空间距离平方 [N, k]
            Tensor d_space = data.pos.unsqueeze(1).sub(centersPos.unsqueeze(0)).pow(new Scalar(2)).sum(2);
            // 颜色距离平方 [N, k]
            Tensor d_color = data.x.unsqueeze(1).sub(centersCol.unsqueeze(0)).pow(new Scalar(2)).sum(2);

            // 组合距离
            Tensor dists = d_color.add(d_space.mul( new Scalar(Math.pow(m / s, 2))));

            // 分配标签：找到距离最近的中心
            labels = dists.argmin(new LongOptional(1),false);

            // 更新中心 (Scatter Mean)
            // 这里我们需要计算每个标签下像素的平均值
            centersPos = scatter_mean(data.pos, labels, 0, k);
            centersCol = scatter_mean(data.x, labels, 0, k);
        }

        // 3. 构建超像素图数据
        GraphData superpixelGraph = new GraphData(centersCol, null);
        superpixelGraph.pos = centersPos;

        // 4. 建立边索引 (可选：通过 KNN 或 寻找相邻像素的 label 差异)
        // 此处建议使用 KNN(k=6) 来建立超像素间的邻接关系
        superpixelGraph.edge_index = computeKNN(centersPos, k);

        return superpixelGraph;
    }
    public static Tensor computeKNN(Tensor pos, int k) {
        long n = pos.size(0);

        // 安全检查：如果总点数不足以支撑 k 个邻居，则动态减小 k
        // 实际能取的最大邻居数是 n - 1
        int actualK = (int) Math.min(k, n - 1);

        if (actualK <= 0) {
            // 如果只有一个点，无法构建边，返回空的 edge_index [2, 0]
            return empty(new long[]{2, 0}, pos.options().dtype(new ScalarTypeOptional(kLong())),new MemoryFormatOptional());
        }

        // 计算距离矩阵...
        Tensor distInner = pos.pow(new Scalar(2)).sum(new long[]{1}, true,new ScalarTypeOptional());
        Tensor distMat = distInner.add(distInner.t()).addmm(pos, pos.t(), new Scalar(1.0),new Scalar(-2.0) );

        // 请求 actualK + 1 个点
        T_TensorTensor_T topkResult = distMat.neg().topk(actualK + 1, 1, true, true);
        Tensor indices = topkResult.get1();

        // 移除自身，取剩余的 actualK 个
//        indices = indices.slice(1, 1, actualK + 1);
        indices = indices.slice(1, new LongOptional(1),new LongOptional(k+1) , 1);

        // 构造 row 和 col
        Tensor row = arange(new Scalar(n), pos.options().dtype(new ScalarTypeOptional(kLong())))
                .view(-1, 1).expand(new long[]{n, actualK}).reshape(-1);
        Tensor col = indices.reshape(-1);

        return stack(new TensorVector(row, col), 0);
    }
    public static Tensor computeKNN2(Tensor pos, int k) {
        // pos 形状: [N, D]
        long n = pos.size(0);

        // 1. 计算范数平方 ||a||^2: [N, 1]
        Tensor distInner = pos.pow(new Scalar(2)).sum(new long[]{1}, true, new ScalarTypeOptional());

        // 2. 计算距离矩阵: ||a||^2 + ||b||^2 - 2 * a @ b.T
        // 利用广播机制得到 [N, N] 矩阵
        Tensor distMat = distInner.add(distInner.t())
                .addmm(pos, pos.t(), new Scalar(1.0), new Scalar(-2.0));

        // 3. 取得每个点的最近 k 个索引
        // 注意：第 0 个通常是点自身（距离为 0），所以我们取 topk(k + 1)
        // topk 默认返回最大的值，我们要最小的距离，所以对距离矩阵取负号，或者使用专门的 topk 参数
        T_TensorTensor_T topkResult = distMat.neg().topk(k + 1, 1, true, true);
        Tensor indices = topkResult.get1(); // [N, k+1]

        // 4. 移除第一个索引（自身），并重塑为 edge_index 格式 [2, N * k]
        // 选取的范围是索引列的 1 到 k+1
        indices = indices.slice(1, new LongOptional(1),new LongOptional(k+1) , 1);

        // 5. 构造 edge_index
        // 源节点 (row): [0,0...1,1...N,N]
        Tensor row = arange( new Scalar(n), pos.options().dtype(new ScalarTypeOptional(kLong())))
                .view(-1, 1)
                .expand(new long[]{n, k})
                .reshape(-1);
        // 目标节点 (col): flatten indices
        Tensor col = indices.reshape(-1);

        return stack(new TensorVector(row, col), 0);
    }

    // 辅助方法：基于 Tensor 的 scatter_mean 实现
    private Tensor scatter_mean(Tensor src, Tensor index, int dim, int num_clusters) {
        long featDim = src.size(1);
        Tensor out = zeros(new long[]{num_clusters, featDim}, src.options());
        Tensor count = zeros(new long[]{num_clusters, featDim}, src.options());

        Tensor expandIdx = index.unsqueeze(1).expand_as(src);
        out.scatter_add_(dim, expandIdx, src);
        count.scatter_add_(dim, expandIdx, ones_like(src));

        return out.div(count.add(new Scalar(1e-7)));
    }
}
//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.opencv.opencv_ximgproc.*;
//import static org.bytedeco.opencv.global.opencv_ximgproc.*;
//import org.bytedeco.pytorch.*;
//import org.bytedeco.pytorch.geometric.data.GraphData;
//
//import static org.bytedeco.pytorch.global.torch.*;
//
//public static class ToSLIC implements BaseTransform {
//    private final int numSegments;
//    private final float compactness;
//
//    public ToSLIC(int numSegments, float compactness) {
//        this.numSegments = numSegments;
//        this.compactness = compactness;
//    }
//
//    @Override
//    public GraphData apply(GraphData data) {
//        // 假设 data.x 此时存储的是原始图像张量 [C, H, W]
//        // 1. 转换为 OpenCV Mat
//        Mat img = tensorToMat(data.x);
//
//        // 2. 创建 SLIC 对象
//        SuperpixelSLIC slic = createSuperpixelSLIC(img, SLIC, numSegments, compactness);
//        slic.iterate(10); // 迭代 10 次优化
//
//        int numActualSegments = slic.getNumberOfSuperpixels();
//        Mat labels = new Mat();
//        slic.getLabels(labels); // 获取每个像素所属的超像素 ID [H, W]
//
//        // 3. 计算每个超像素的中心和平均颜色
//        // 利用 labels 矩阵进行聚合运算 (类似于 scatter_mean)
//        // 生成最终的 data.pos [numActualSegments, 2] 和 data.x [numActualSegments, C]
//
//        // 4. 构建邻接关系 (RAG: Region Adjacency Graph)
//        // 遍历 labels，检查相邻像素是否属于不同超像素，若是则在 edge_index 中连边
//
//        return data;
//    }
//
//    // 辅助工具：将 Tensor 转换为 OpenCV Mat
//    private Mat tensorToMat(Tensor t) {
//        // 实现张量到 Mat 的内存转换逻辑
//        return new Mat();
//    }
//}
