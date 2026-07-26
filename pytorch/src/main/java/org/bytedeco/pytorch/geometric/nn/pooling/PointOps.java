package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

public class PointOps {

    /**
     * Voxel Grid Sampling
     * @param pos   点坐标 [N, 3]
     * @param batch 批次索引 [N] (可选)
     * @param size  体素大小 (scalar)
     * @return Tensor[] {pos_pooled, batch_pooled}
     */
//    public static Tensor[] voxel_grid(Tensor pos, Tensor batch, float size) {
//        long N = pos.size(0);
//
//        // 1. 计算 Grid Index
//        // idx = floor(pos / size)
//        Tensor gridIdx = pos.div(new Scalar(size)).floor().to(torch.ScalarType.Long);
//
//        // 2. 将 3D (或D维) 坐标压缩为 1D Hash Key 以便排序去重
//        // 简单策略：Coordinate hashing
//        // key = x * d1 + y * d2 + z
//        // 为了避免冲突，需要计算每维的 span。
//        Tensor minVal = pos.min(0).get0() ; //.values();
//        Tensor maxVal = pos.max(0).get0();//.values();
//        Tensor dims = maxVal.sub(minVal).div(new Scalar(size)).ceil().to(torch.ScalarType.Long); // Grid dimensions
//
//        long dimX = dims.dataAsLongArray()[0];
//        long dimY = dims.dataAsLongArray()[1];
//        // z 不用乘系数
//
//        Tensor xIdx = gridIdx.select(1, 0).sub(minVal.select(0, 0).div(new Scalar(size)).floor().to(torch.ScalarType.Long));
//        Tensor yIdx = gridIdx.select(1, 1).sub(minVal.select(0, 1).div(new Scalar(size)).floor().to(torch.ScalarType.Long));
//        Tensor zIdx = gridIdx.select(1, 2).sub(minVal.select(0, 2).div(new Scalar(size)).floor().to(torch.ScalarType.Long));    
//
//        // 引入 Batch 维度参与 Hash，避免不同 Batch 的点混在一起
//        Tensor key = xIdx.mul(new Scalar(dimY)).add(yIdx); // 2D key
//        // 3D key ( + zIdx ) ... 这里简化逻辑，直接利用 unique 的 dim=0 特性
//
//        // 实际上，更稳健的方法是利用 torch.unique(grid_idx_with_batch, dim=0)
//        // [N, 4] -> [x_i, y_i, z_i, batch_i]
//        Tensor keys;
//        if (batch != null) {
//            keys = torch.cat(new TensorVector(gridIdx, batch.unsqueeze(1)), 1);
//        } else {
//            keys = gridIdx;
//        }
//
//        // 3. Unique -> Cluster Assignment
//        // return_inverse=True: inverse_indices 就是我们要的 cluster 索引
//        TensorVector uniqueRet = torch.unique_consecutive(keys, false, true, false, 0);
//        // uniqueRet: (unique_elements, inverse_indices)
//        // 注意：unique_consecutive 需要先 sort，这里假设 unique 能处理或者我们先 sort
//        // 标准 unique 比较慢但不需要 sort。为了稳健使用 unique (不是 consecutive)
//
//        // 使用 inverse indices 进行聚合
//        uniqueRet = torch.unique(keys, false, true, false, 0);
//        Tensor cluster = uniqueRet.get(1); // [N]
//
//        long numClusters = uniqueRet.get(0).size(0);
//
//        // 4. org.bytedeco.pytorch.geometric.utils.Scatter Mean (Compute Centroids)
//        // 利用我们之前写的 org.bytedeco.pytorch.geometric.utils.AggrUtils
//        Tensor posPooled = AggrUtils.scatter(pos, cluster, numClusters, "mean");
//
//        Tensor batchPooled = null;
//        if (batch != null) {
//            // Batch 也是同理，取 max 或 mean (结果是一样的，因为同一个 voxel 肯定在同一个 batch)
//            batchPooled = AggrUtils.scatter(batch.to(torch.ScalarType.Float), cluster, numClusters, "max").to(torch.ScalarType.Long);
//        }
//
//        return new Tensor[]{posPooled, batchPooled};
//    }


    // --- KNN Operations ---

    /**
     * Exact K-Nearest Neighbors
     *
     * @param x 查询点 [N, D]
     * @param y 数据库点 [M, D] (若为null则 y=x)
     * @param k 邻居数
     * @return Tensor indices [N, k]
     */
    public static Tensor knn(Tensor x, Tensor y, int k, Tensor batchX, Tensor batchY) {
        if (y == null) y = x;

        // 1. 计算距离矩阵 [N, M]
        // cdist 计算 L2 距离
        Tensor dist = torch.cdist(x, y);

        // 2. 处理 Batch (Masking)
        if (batchX != null && batchY != null) {
            // mask[i, j] = True if batchX[i] != batchY[j]
            Tensor mask = batchX.unsqueeze(1).ne(batchY.unsqueeze(0));
            // 将不同 batch 的距离设为无穷大
            dist.masked_fill_(mask, new Scalar(Float.POSITIVE_INFINITY));
        }

        // 3. TopK (smallest)
        // largest=false -> smallest
        Tensor indices = torch.topk(dist, k, 1, false, true).get1();
        return indices;
    }

    /**
     * KNN Graph
     * 构建 KNN 图
     *
     * @return edge_index [2, N*k]
     */
    public static Tensor knn_graph(Tensor x, int k, Tensor batch) {
        long N = x.size(0);

        // 1. Get Indices [N, k]
        Tensor col = knn(x, null, k, batch, batch);

        // 2. Create Row indices [N, k]
        // 0,0,0... 1,1,1...
        Tensor row = torch.arange(new Scalar(N), x.options()).unsqueeze(1).expand(new long[]{N, k});

        // 3. Flatten & Stack
        row = row.reshape(N * k);
        col = col.reshape(N * k);

        return torch.stack(new TensorVector(row, col), 0);
    }

    // --- Farthest Point Sampling ---

    /**
     * Farthest Point Sampling (Batched)
     *
     * @param x          [B, N, D] (注意：FPS通常需要 batch first 格式以并行化)
     * @param numSamples 采样数 (M)
     * @return indices [B, M]
     */
    public static Tensor fps(Tensor x, int numSamples) {
        long B = x.size(0);
        long N = x.size(1);
        long D = x.size(2);

        Device device = x.device();

        // 存储结果索引 [B, M]
        Tensor centroids = torch.zeros(new long[]{B, numSamples}, x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 存储每个点到“已选集合”的最短距离 [B, N]
        // 初始为无穷大
        Tensor distance = torch.full(new long[]{B, N}, new Scalar(1e20), x.options());

        // 随机选择第一个点 (或者选 index 0)
        // 这里选 index 0 的点作为起始点 (Random 0~N-1)
        Tensor farthest = torch.randint(N, new long[]{B}, x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        // 生成 Batch 索引 [0, 1, ..., B-1]
        Tensor batchIndices = torch.arange(new Scalar(B), x.options().dtype(new ScalarTypeOptional(torch.ScalarType.Long)));

        for (int i = 0; i < numSamples; i++) {
            // 1. 记录当前选出的最远点 torch.all() 表示对所有 batch 都操作//
            centroids.index_put_(new TensorIndexVector(new TensorIndex(new Slice(i)), new TensorIndex(new Slice(i))), farthest);

            // 2. 获取当前最远点的坐标
            // x[b, farthest[b], :]
            // index_select 比较麻烦，使用 gather 或 advanced indexing
            // centroid: [B, 1, D]
            Tensor centroid = x.index(new TensorIndexVector(new TensorIndex(batchIndices), new TensorIndex(farthest))).unsqueeze(1);

            // 3. 计算当前点到所有点的距离
            // dist: [B, N]
            // (x - centroid)^2 -> sum -> val
            Tensor dist = x.sub(centroid).pow(new Scalar(2)).sum(new long[]{2}, false, new ScalarTypeOptional(torch.ScalarType.Float));

            // 4. 更新全局最短距离
            // distance = min(distance, current_dist)
            Tensor mask = dist.lt(distance);
            distance.index_put_(new TensorIndexVector(mask), dist.index(new TensorIndexVector(mask)));
            // 或者直接: distance = torch.min(distance, dist);

            // 5. 选出下一个最远点 (argmax)
            farthest = distance.max(1).get1(); // [B]
        }

        return centroids;
    }

    // --- Approx KNN ---

    /**
     * Approx KNN (Random Projection)
     * 将 D 维投影到 projDim 维，然后在低维做 exact knn。
     */
    public static Tensor approx_knn(Tensor x, int k, Tensor batch, int projDim) {
        long D = x.size(1);

        // 1. 生成随机投影矩阵 [D, projDim]
        // 复用 org.bytedeco.pytorch.geometric.utils.AttentionUtils 中的逻辑或直接生成
        Tensor projMat = torch.randn(new long[]{D, projDim}, x.options());

        // 2. 投影
        Tensor xProj = x.matmul(projMat);

        // 3. 在低维空间做 KNN
        return knn(xProj, null, k, batch, batch);
    }

    public static Tensor approx_knn_graph(Tensor x, int k, Tensor batch, int projDim) {
        long N = x.size(0);
        Tensor col = approx_knn(x, k, batch, projDim);

        Tensor row = torch.arange(new Scalar(N), x.options()).unsqueeze(1).expand(new long[]{N, k});
        row = row.reshape(N * k);
        col = col.reshape(N * k);

        return torch.stack(new TensorVector(row, col), 0);
    }
}
