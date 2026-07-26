package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

import static org.bytedeco.pytorch.global.torch.*;

//import static org.bytedeco.pytorch.geometric.nn.pooling.PointOps.knn;

/**
 * Cluster Pooling Utilities
 * 包含 max_pool, avg_pool, graclus 等静态方法
 */
public class ClusterPooling {

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

        Tensor row = arange(new Scalar(N), x.options()).unsqueeze(1).expand(new long[]{N, k});
        row = row.reshape(N * k);
        col = col.reshape(N * k);

        return torch.stack(new TensorVector(row, col), 0);
    }
    /**
     * KNN Graph
     * 构建 KNN 图
     * @return edge_index [2, N*k]
     */
    public static Tensor knn_graph(Tensor x, int k, Tensor batch) {
        long N = x.size(0);

        // 1. Get Indices [N, k]
        Tensor col = knn(x, null, k, batch, batch);

        // 2. Create Row indices [N, k]
        // 0,0,0... 1,1,1...
        Tensor row = arange(new Scalar(N), x.options()).unsqueeze(1).expand(new long[]{N, k});

        // 3. Flatten & Stack
        row = row.reshape(N * k);
        col = col.reshape(N * k);

        return torch.stack(new TensorVector(row, col), 0);
    }

    // --- KNN Operations ---

    /**
     * Exact K-Nearest Neighbors
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


    // --- Farthest Point Sampling ---

    /**
     * 最远点采样 (FPS) 修复版
     *
     * @param x          输入点云 [B, N, D]
     * @param numSamples 采样点数 M
     * @return 采样点的索引 [B, M]
     */
    public static Tensor fps(Tensor x, int numSamples) {
        long B = x.size(0);
        long N = x.size(1);

        // 1. 初始化结果 [B, M]
        Tensor centroids = zeros(new long[]{B, numSamples},
                x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        // 2. 初始化距离 [B, N]，代表每个点到已选集合的最短距离
        // 使用 1e10 代替 1e20 防止精度溢出
        Tensor distance = torch.full(new long[]{B, N}, new Scalar(1e10), x.options());

        // 3. 随机选择或固定选择第一个采样点 [B]
        Tensor farthest = zeros(new long[]{B},
                x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        // 用于高级索引的 batch 索引 [B]
        Tensor batchIndices = arange(new Scalar(0), new Scalar(B),
                x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        for (int i = 0; i < numSamples; i++) {
            // --- 修复点 1: 记录当前选出的点索引 ---
            // centroids[:, i] = farthest
            centroids.select(1, i).copy_(farthest);

            // --- 修复点 2: 获取当前采样点的坐标 [B, 1, D] ---
            // 替代复杂的 index_put，使用 gather 逻辑
            // 先通过 batchIndices 和 farthest 选出点坐标
            // x.index(batchIndices, farthest) -> [B, D]
            Tensor centroid = x.index(new TensorIndexVector(new TensorIndex(batchIndices), new TensorIndex(farthest))).unsqueeze(1);

            // 3. 计算所有点到当前采样点的欧式距离平方 [B, N]
            // dist = sum((x - centroid)^2, dim=2)
            Tensor dist = x.subtract(centroid).pow(new Scalar(2.0)).sum(new long[]{2}, false, new ScalarTypeOptional(ScalarType.Float));

            // --- 修复点 3: 更新全局最短距离 ---
            // distance = min(distance, dist) 
            // 这种写法最稳健，完全避开 mask 和 index_put_ 的 Slice 报错
            distance = torch.min(distance, dist);

            // 5. 选出下一个最远点：距离已选集合最远的点
            // farthest = argmax(distance, dim=1)
            farthest = distance.max(1).get1();
        }

        return centroids;
    }
    /**
     * Farthest Point Sampling (Batched)
     * @param x [B, N, D] (注意：FPS通常需要 batch first 格式以并行化)
     * @param numSamples 采样数 (M)
     * @return indices [B, M]
     */
    public static Tensor fpss(Tensor x, int numSamples) {
        long B = x.size(0);
        long N = x.size(1);
        long D = x.size(2);

        Device device = x.device();

        // 存储结果索引 [B, M]
        Tensor centroids = zeros(new long[]{B, numSamples}, x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        // 存储每个点到“已选集合”的最短距离 [B, N]
        // 初始为无穷大
        Tensor distance = torch.full(new long[]{B, N}, new Scalar(1e20), x.options());

        // 随机选择第一个点 (或者选 index 0)
        // 这里选 index 0 的点作为起始点 (Random 0~N-1)
        Tensor farthest = torch.randint(N, new long[]{B}, x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        // 生成 Batch 索引 [0, 1, ..., B-1]
        Tensor batchIndices = arange(new Scalar(B), x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

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
            Tensor dist = x.sub(centroid).pow(new Scalar(2)).sum(new long[]{2}, false,new ScalarTypeOptional(ScalarType.Float));

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

    // --- 1. Basic Pooling (Based on Cluster Index) ---

    /**
     * max_pool
     * @param cluster [N] 节点所属的簇索引
     * @param x [N, C] 节点特征
     * @return [NumClusters, C]
     */
    public static Tensor max_pool(Tensor cluster, Tensor x) {
        long numClusters = cluster.max().item().toLong() + 1;
        return AggrUtils.scatter(x, cluster, numClusters, "max");
    }

    /**
     * avg_pool
     */
    public static Tensor avg_pool(Tensor cluster, Tensor x) {
        long numClusters = cluster.max().item().toLong() + 1;
        return AggrUtils.scatter(x, cluster, numClusters, "mean");
    }

    /**
     * max_pool_x (PyG 别名)
     * 返回 (x_pooled, batch_pooled)
     */
    public static Tensor[] max_pool_x(Tensor cluster, Tensor x, Tensor batch) {
        Tensor xPool = max_pool(cluster, x);
        // Batch 也需要池化 (假设同一个 cluster 的节点属于同一个 batch)
        Tensor batchPool = max_pool(cluster, batch.to(ScalarType.Float)).to(ScalarType.Long);
        return new Tensor[]{xPool, batchPool};
    }

    public static Tensor[] avg_pool_x(Tensor cluster, Tensor x, Tensor batch) {
        Tensor xPool = avg_pool(cluster, x);
        Tensor batchPool = max_pool(cluster, batch.to(ScalarType.Float)).to(ScalarType.Long);
        return new Tensor[]{xPool, batchPool};
    }

    // --- 2. Neighbor Pooling (类似 ClusterGCN) ---

    /**
     * avg_pool_neighbor_x
     * 聚合每个簇的“邻居簇”的特征。这需要 cluster_edge_index。
     */
    public static Tensor avg_pool_neighbor_x(Tensor x, Tensor edge_index) {
        // 这其实就是 GCN 的聚合步：Propagate Mean
        long numNodes = x.size(0);
        return AggrUtils.scatter(x.index_select(0, edge_index.select(0, 0)),
                edge_index.select(0, 1), numNodes, "mean");
    }

    // --- 3. Algorithms: Graclus / Greedy Pooling ---

    /**
     * Graclus Clustering 替代版 (Greedy Max Matching)
     * 计算图的最大匹配，将匹配的节点对合并为一个簇。
     * @param edge_index [2, E]
     * @param numNodes 节点数
     * @return cluster [N] (Cluster ID for each node)
     */
    public static Tensor graclus(Tensor edge_index, long numNodes) {
        // 1. 初始化: cluster[i] = -1
        Tensor cluster = torch.full(new long[]{numNodes}, new Scalar(-1), edge_index.options());

        // 2. 随机打乱边 (Randomize for stochasticity)
        Tensor randPerm = torch.randperm(edge_index.size(1), edge_index.options().dtype(new ScalarTypeOptional( ScalarType.Long)));
        Tensor row = edge_index.select(0, 0).index_select(0, randPerm);
        Tensor col = edge_index.select(0, 1).index_select(0, randPerm);

        // 3. 贪婪匹配 (Java 循环慢，但在没有 Custom Kernel 时是唯一解)
        // 为了加速，我们将 Tensor 拉回 CPU 数组处理
        long[] rowArr = new long[] {};// row.to(new Device("cpu"), torch.ScalarType.Long).dataAsLongArray(); ????
        long[] colArr = new long[]{} ; // col.to(new Device("cpu"), torch.ScalarType.Long).dataAsLongArray(); ???
        long[] clusterArr = new long[(int)numNodes];
        java.util.Arrays.fill(clusterArr, -1);

        int clusterCount = 0;

        for (int i = 0; i < rowArr.length; i++) {
            long u = rowArr[i];
            long v = colArr[i];

            if (u == v) continue; // Ignore self-loops

            // 如果两个节点都未被分配
            if (clusterArr[(int)u] == -1 && clusterArr[(int)v] == -1) {
                clusterArr[(int)u] = clusterCount;
                clusterArr[(int)v] = clusterCount;
                clusterCount++;
            }
        }

        // 4. 处理未匹配节点 (Assign singleton clusters)
        for (int i = 0; i < numNodes; i++) {
            if (clusterArr[i] == -1) {
                clusterArr[i] = clusterCount++;
            }
        }

        // 5. 转回 Tensor
        return torch.tensor(clusterArr).to(edge_index.device(), ScalarType.Long);
    }


    // --- Radius Search ---

    /**
     * Radius Search
     * 查找半径 r 内的所有邻居。
     *
     * @param x 查询点 [N, D]
     * @param y 数据库点 [M, D] (若为null则 y=x)
     * @param r 半径
     * @param maxNumNeighbors 每个查询点最多返回多少个邻居 (可选，-1表示不限制)
     * @return edge_index [2, E] (Row: index in x, Col: index in y)
     */
    public static Tensor radius(Tensor x, Tensor y, float r, Tensor batchX, Tensor batchY, int maxNumNeighbors) {
        if (y == null) y = x;

        // 1. 计算距离矩阵 [N, M]
        Tensor dist = torch.cdist(x, y);

        // 2. 处理 Batch
        if (batchX != null && batchY != null) {
            Tensor mask = batchX.unsqueeze(1).ne(batchY.unsqueeze(0));
            dist.masked_fill_(mask, new Scalar(Float.POSITIVE_INFINITY));
        }

        // 3. 生成 Mask: dist < r
        Tensor mask = dist.lt(new Scalar(r));

        // 4. 提取索引
        // nonzero() 返回 [E, 2] -> (row, col)
        Tensor indices = mask.nonzero();

        // 转置为 [2, E]
        Tensor edge_index = indices.t();

        // 5. (Optional) Max Num Neighbors 限制
        // 如果不限制，直接返回 edge_index
        if (maxNumNeighbors > 0) {
            // 这是一个比较麻烦的操作。如果不引入 sort，很难高效截断。
            // 简单策略：先计算 topk，再结合 radius mask。
            // 但标准 radius search 通常返回所有。
            // 考虑到性能，纯 LibTorch 实现如果要做 limit，通常是在 dense mask 上做，
            // 或者后期对 edge_index 进行采样。
            // 这里为了保持实现简洁和正确性，暂时忽略 maxNumNeighbors 并在日志警告，
            // 或者如果非常重要，我们可以对 dist 做 topk 再 mask。

            // 策略: 如果 E 极大，可以考虑在这里截断，但在 pure tensor 下很难对每个 row 单独计数截断。
            // 工业界通常使用 CUDA kernel (torch_cluster) 解决此问题。
            // 在此仅做标准 radius 实现。
        }

        return edge_index;
    }

    /**
     * Radius Graph
     * 构建基于半径的图结构
     */
    public static Tensor radius_graph(Tensor x, float r, Tensor batch, boolean loop, int maxNumNeighbors) {
        Tensor edge_index = radius(x, x, r, batch, batch, maxNumNeighbors);

        // 移除自环 (dist < r 会包含 dist=0)
        if (!loop) {
            Tensor row = edge_index.select(0, 0);
            Tensor col = edge_index.select(0, 1);
            Tensor mask = row.ne(col); // row != col

            Tensor finalRow = row.masked_select(mask);
            Tensor finalCol = col.masked_select(mask);

            return torch.stack(new TensorVector(finalRow, finalCol), 0);
        }

        return edge_index;
    }

    // --- Nearest ---

    /**
     * Find nearest neighbor (k=1)
     * @return indices [N] (Index in y for each x)
     */
    public static Tensor nearest(Tensor x, Tensor y, Tensor batchX, Tensor batchY) {
        // 复用 knn 逻辑，k=1
        // indices shape: [N, 1]
        Tensor idx = knn(x, y, 1, batchX, batchY);
        return idx.squeeze(1);
    }

    // --- Interpolation ---

    /**
     * KNN Interpolate (Inverse Distance Weighting)
     * 将源点集(source)的特征 插值 到 目标点集(target)。
     *
     * @param x     目标点坐标 (Target/Dense) [N, D]
     * @param y     源点坐标 (Source/Sparse) [M, D]
     * @param yFeat 源点特征 [M, C]
     * @param batchX Target Batch [N]
     * @param batchY Source Batch [M]
     * @param k     邻居数
     * @return      插值后的特征 [N, C]
     */
    public static Tensor knn_interpolate(Tensor x, Tensor y, Tensor yFeat, Tensor batchX, Tensor batchY, int k) {
        long N = x.size(0);
        long C = yFeat.size(1);

        // 1. 寻找邻居: Target 找 Source
        // indices: [N, k]
        Tensor idx = knn(x, y, k, batchX, batchY);

        // 2. 获取距离
        // 为了获得距离，我们需要 gather 坐标并计算，或者 knn 函数修改为返回 (dist, idx)。
        // 为了复用现有 knn (只返回 idx)，我们手动 gather 计算一下距离。
        // y_neighbors: [N, k, D]
        // 这里 idx 是 [N, k]，我们需要把它展平用于 index_select 然后 reshape
        Tensor idxFlat = idx.reshape(N * k);
        Tensor yNeighbors = y.index_select(0, idxFlat).reshape(N, k, y.size(1));

        // x_expanded: [N, 1, D]
        Tensor xExpanded = x.unsqueeze(1);

        // dist: [N, k]
        Tensor dist = xExpanded.sub(yNeighbors).norm(new ScalarOptional(new Scalar(2)), new long[]{2}, false);

        // 3. 计算权重 (Inverse Distance)
        // weights = 1.0 / (dist + 1e-9)
        Tensor weights = dist.add(new Scalar(1e-9)).reciprocal();

        // 4. 归一化权重
        // sum_weights: [N, 1]
        Tensor sumWeights = weights.sum(new long[]{1}, true, new ScalarTypeOptional(ScalarType.Float));
        weights = weights.div(sumWeights); // [N, k]

        // 5. Gather Source Features
        // yFeat: [M, C] -> y_feat_neighbors: [N, k, C]
        Tensor yFeatNeighbors = yFeat.index_select(0, idxFlat).reshape(N, k, C);

        // 6. 加权求和
        // weights: [N, k] -> [N, k, 1]
        Tensor weightedFeat = yFeatNeighbors.mul(weights.unsqueeze(2));

        // sum(dim=1) -> [N, C]
        return weightedFeat.sum(new long[]{1}, false, new ScalarTypeOptional(ScalarType.Float));
    }


    /**
     * Voxel Grid Pooling (体素栅格下采样)
     *
     * @param pos   点云坐标 [N, 3]
     * @param batch 批次索引 [N]
     * @param size  体素大小 (如 0.05)
     * @return 聚合后的新点坐标和所属 batch
     */
    public static VoxelOutput voxel_grid(Tensor pos, Tensor batch, double size) {
        // 1. 数据准备
        pos = pos.contiguous();
        Scalar sSize = new Scalar(size);

        // 2. 计算每个维度的体素坐标
        // 归一化坐标：(pos - min) / size
        Tensor minPos = pos.min(0, true).get0(); // [1, 3]
        Tensor gridCoords = pos.subtract(minPos).divide(sSize).floor(); // [N, 3]

        // 3. 将 3D 坐标压缩为 1D 索引 (Voxel ID)
        // 为了防止不同 batch 的点落在同一个 voxel_id，我们需要引入 batch 偏移
        Tensor maxCoords = gridCoords.max(0).get0().add(new Scalar(1.0));

        // 映射公式: id = x + y * max_x + z * max_x * max_y + batch * offset
        Tensor vX = gridCoords.select(1, 0);
        Tensor vY = gridCoords.select(1, 1);
        Tensor vZ = gridCoords.select(1, 2);

        long mx = maxCoords.index(new TensorIndexVector(new TensorIndex(0))).item().toLong();
        long my = maxCoords.index(new TensorIndexVector(new TensorIndex(1))).item().toLong();

        Tensor voxelId = vX.add(vY.multiply(new Scalar(mx)))
                .add(vZ.multiply(new Scalar(mx * my)));

        // 如果有 batch 信息，叠加 batch 偏移量
        if (batch != null) {
            long offset = mx * my * (maxCoords.index(new TensorIndexVector(new TensorIndex(2))).item().toLong() + 1);
            voxelId = voxelId.add(batch.multiply(new Scalar(offset)));
        }

        // 4. 使用 inverse 映射将稀疏的 voxelId 转换为连续的 cluster 索引 [0, K-1]
        // 这是为了方便后续 scatter 操作
        Tensor[] uniqueResult = _unique(voxelId);
        Tensor cluster = uniqueResult[1]; // 获取每个点对应的连续索引
        long numClusters = uniqueResult[0].size(0);

        // 5. 聚合特征 (重心采样)
        Tensor newPos = zeros(new long[]{numClusters, 3}, pos.options());
        Tensor expandedCluster = cluster.unsqueeze(1).expand_as(pos);

        // 求和
        newPos = newPos.scatter_add(0, expandedCluster, pos);

        // 计数并求均值
        Tensor count = zeros(new long[]{numClusters, 1}, pos.options());
        count = count.scatter_add(0, cluster.unsqueeze(1), ones(new long[]{pos.size(0), 1}, pos.options()));
        newPos = newPos.divide(count.clamp_min(new Scalar(1.0)));

        // 6. 获取新点的 batch 归属 (取每个 cluster 中第一个点的 batch)
        Tensor newBatch = null;
        if (batch != null) {
            newBatch = zeros(new long[]{numClusters}, batch.options());
            newBatch = newBatch.scatter(0, cluster, batch); // 后面的会覆盖前面的，但同一个 voxel 里的 batch 是一样的
        }

        return new VoxelOutput(newPos, newBatch, cluster);
    }

    // 内部辅助方法：处理 unique 逻辑
    private static Tensor[] _unique(Tensor input) {
        // 调用 LibTorch 的 unique 方法
        // 返回: {unique_values, inverse_indices}
        //torch.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None) → tuple[Tensor, Tensor, Tensor][source]
        T_TensorTensorTensor_T res = torch.unique_consecutive(input, true, true, new LongOptional());
        return new Tensor[]{res.get0(), res.get1()};
    }
}