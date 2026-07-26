package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;

public class EdgePooling extends org.bytedeco.pytorch.nn.Module {
    private long inChannels;
    private LinearImpl lin;

    public EdgePooling(long inChannels) {
        super();
        this.inChannels = inChannels;
        // PyG 逻辑: 拼接两个节点的特征 [2 * inChannels] 映射到 1 个分数
        this.lin = new LinearImpl(2 * inChannels, 1);
        register_module("lin", lin);
    }


    public EdgePoolingOutput edgePool(Tensor x, Tensor edge_index) {
        x = x.contiguous();
        long numNodes = x.size(0);
        long numEdges = edge_index.size(1);

        // 1. 计算边得分
        Tensor x_j = x.index_select(0, edge_index.select(0, 0));
        Tensor x_i = x.index_select(0, edge_index.select(0, 1));
        Tensor score = lin.forward(cat(new TensorVector(x_i, x_j), 1)).view(new long[]{-1});
        score = softmax(score, 0);

        // 2. 准备贪婪匹配的数据 (拉回 CPU 连续内存)
        Tensor sortedIndices = score.argsort(0, true).cpu().contiguous();
        Tensor edge_indexCPU = edge_index.cpu().contiguous();

        // 获取原生指针，替代报错的 .index(new Slice...)
        LongPointer sortedIdxPtr = new LongPointer(sortedIndices.data_ptr());
        LongPointer edgeIdxPtr = new LongPointer(edge_indexCPU.data_ptr());

        // 使用 Java 数组管理状态，速度比 Tensor 快 100 倍且不会报 Slice 错误
        boolean[] matched = new boolean[(int) numNodes];
        long[] cluster = new long[(int) numNodes];
        Arrays.fill(cluster, -1);
        long newNumNodes = 0;

        // --- 核心贪婪逻辑 (修复崩溃点) ---
        for (int i = 0; i < numEdges; i++) {
            // 获取排好序的边 ID
            long idx = sortedIdxPtr.get(i);

            // 获取该边连接的两个节点 u, v
            // edge_index 形状 [2, E]，row-major 存储
            long u = edgeIdxPtr.get(idx);              // 行 0, 列 idx
            long v = edgeIdxPtr.get(numEdges + idx);   // 行 1, 列 idx

            if (!matched[(int) u] && !matched[(int) v]) {
                matched[(int) u] = true;
                matched[(int) v] = true;
                cluster[(int) u] = newNumNodes;
                cluster[(int) v] = newNumNodes;
                newNumNodes++;
            }
        }

        // 处理未匹配的孤立节点
        for (int i = 0; i < numNodes; i++) {
            if (cluster[i] == -1) {
                cluster[i] = newNumNodes;
                newNumNodes++;
            }
        }

        // 3. 构建聚类张量并池化 (转换 cluster 为 Long 类型)
        Tensor clusterTensor = tensor(cluster, x.options().dtype(new ScalarTypeOptional(ScalarType.Long)));

        // 聚合节点特征 (Mean Pooling)
        Tensor newX = zeros(new long[]{newNumNodes, inChannels}, x.options());
        Tensor expandedCluster = clusterTensor.unsqueeze(1).expand_as(x);
        newX = newX.scatter_add(0, expandedCluster, x);

        // 计算每个 cluster 的节点数用于求均值
        Tensor count = zeros(new long[]{newNumNodes, 1}, x.options());
        count = count.scatter_add(0, clusterTensor.unsqueeze(1), ones(new long[]{numNodes, 1}, x.options()));
        newX = newX.divide(count.clamp_min(new Scalar(1.0)));

        return new EdgePoolingOutput(newX, clusterTensor);
    }

    /**
     * @param x         节点特征 [N, C]
     * @param edge_index 边索引 [2, E]
     *                  //     * @param batch 批次索引 [N]
     */
    public EdgePoolingOutput forward2(Tensor x, Tensor edge_index) {
        x = x.contiguous();
        // 1. 计算边得分
        // 提取每条边对应的源节点和目标节点特征
        Tensor x_j = x.index_select(0, edge_index.select(0, 0));
        Tensor x_i = x.index_select(0, edge_index.select(0, 1));

        // 拼接并计算 score [E, 1]
        Tensor score = lin.forward(cat(new TensorVector(x_i, x_j), 1)).view(new long[]{-1});
        score = softmax(score, 0); // 在全图边上做 softmax

        // 2. 贪婪边合并逻辑 (Greedy Matching)
        // 注意：由于 JavaCPP 访问数据较慢，我们尽量在 C++ 层完成
        // 按照分数从大到小排列索引
        Tensor sortedIndices = score.argsort(0, true);

        // 获取原生数据指针 (替代 dataAsLongArray)
        // 使用 LongIndexer 访问索引
        long numNodes = x.size(0);
        long numEdges = edge_index.size(1);

        // 创建一个用于标记节点是否被占用的数组
        Tensor matched = zeros(new long[]{numNodes}, x.options().dtype(new ScalarTypeOptional(ScalarType.Bool)));
        List<Long> clusterList = new ArrayList<>();

        // --- 核心贪婪逻辑 ---
        // 为了性能，我们这里使用底层的 IndexSelect
        // 我们需要找到不冲突的边进行合并
        // 这里提供一个向量化模拟逻辑或低频次 item() 访问

        // 注意：在实际大规模图中，此处建议调用 C++ 自定义算子
        // 这里展示符合 PyG 逻辑的 Java 实现：
        long[] cluster = new long[(int) numNodes];
        for (int i = 0; i < numNodes; i++) cluster[i] = -1;
        long newNumNodes = 0;

        // 这里必须小心处理，避免频繁跨 JNI // new TensorIndexVector(new TensorIndex(new Slice(i)), new TensorIndex(new Slice(i)))
        for (int i = 0; i < numEdges; i++) {
            long idx = sortedIndices.index(new TensorIndexVector(new TensorIndex(new Slice(i)))).item().toLong();
            long u = edge_index.index(new TensorIndexVector(new TensorIndex(new Slice(0)), new TensorIndex(new Slice(idx)))).item().toLong();
            long v = edge_index.index(new TensorIndexVector(new TensorIndex(new Slice(1)), new TensorIndex(new Slice(idx)))).item().toLong();

            if (!matched.index(new TensorIndexVector(new TensorIndex(new Slice(u)))).item().toBool() &&
                    !matched.index(new TensorIndexVector(new TensorIndex(new Slice(v)))).item().toBool()) {

                matched.index(new TensorIndexVector(new TensorIndex(new Slice(u)))).fill_(torch.tensor(true));
                matched.index(new TensorIndexVector(new TensorIndex(new Slice(v)))).fill_(torch.tensor(true));
                cluster[(int) u] = newNumNodes;
                cluster[(int) v] = newNumNodes;
                newNumNodes++;
            }
        }

        // 处理未匹配的孤立节点
        for (int i = 0; i < numNodes; i++) {
            if (cluster[i] == -1) {
                cluster[i] = newNumNodes;
                newNumNodes++;
            }
        }

        // 3. 构建聚类张量并池化
        Tensor clusterTensor = tensor(cluster, x.options().dtype(new ScalarTypeOptional(ScalarType.Bool)));

        // 聚合节点特征 (Mean Pooling)
        Tensor newX = zeros(new long[]{newNumNodes, inChannels}, x.options());
        Tensor expandedCluster = clusterTensor.unsqueeze(1).expand_as(x);
        newX = newX.scatter_add(0, expandedCluster, x);

        // 修正均值
        Tensor count = zeros(new long[]{newNumNodes, 1}, x.options());
        count = count.scatter_add(0, clusterTensor.unsqueeze(1), ones(new long[]{numNodes, 1}, x.options()));
        newX = newX.divide(count.clamp_min(new Scalar(1.0)));

        // 4. 重构边索引 (使用 PyG pooling 工具逻辑)
        // 这里简化处理：通常使用 coalesce + remove_self_loops
        return new EdgePoolingOutput(newX, clusterTensor);
    }
}
//public class EdgePooling extends Module {
//    private LinearImpl lin; // 计算 Edge Score
//    private double dropout;
//
//    public EdgePooling(long inChannels, double dropout) {
//        this.dropout = dropout;
//        // 输入: Cat(x_i, x_j) -> [2 * In]
//        this.lin = new LinearImpl(2 * inChannels, 1);
//        register_module("lin", lin);
//    }
//
//    /**
//     * @return {x, edge_index, batch, unpool_info}
//     */
//    public Tensor[] forward(Tensor x, Tensor edge_index, Tensor batch) {
//        long numNodes = x.size(0);
//
//        // 1. 计算 Edge Scores
//        Tensor row = edge_index.select(0, 0);
//        Tensor col = edge_index.select(0, 1);
//
//        Tensor xRow = x.index_select(0, row);
//        Tensor xCol = x.index_select(0, col);
//
//        // [E, 2*C]
//        Tensor catFeat = torch.cat(new TensorVector(xRow, xCol), 1);
//        Tensor scores = lin.forward(catFeat).squeeze(1); // [E]
//        scores = torch.dropout(scores, dropout, is_training());
//        scores = torch.softmax(scores, 0); // 或者 sigmoid
//
//        // 2. 寻找非重叠的边 (Matching)
//        // 这里需要实现一个类似 graclus 的匹配逻辑，但是基于 scores 排序
//        // 我们可以复用 ClusterPooling.graclus 的逻辑，但在 CPU 上根据 score 排序
//
//        // 简化的 CPU Matching 实现：
//        // cluster[i] 存储节点 i 归属的新节点 ID
//        long[] clusterArr = new long[(int)numNodes];
//        java.util.Arrays.fill(clusterArr, -1);
//
//        // 获取排序索引 (Descending)
//        Tensor sortIdx = torch.argsort(scores, 0, true);
//        long[] sortIdxArr = sortIdx.cpu().dataAsLongArray();
//        long[] rowArr = row.cpu().dataAsLongArray();
//        long[] colArr = col.cpu().dataAsLongArray();
//
//        int newIdx = 0;
//
//        // Greedy Matching
//        for (long idx : sortIdxArr) {
//            int u = (int) rowArr[(int)idx];
//            int v = (int) colArr[(int)idx];
//
//            if (clusterArr[u] == -1 && clusterArr[v] == -1) {
//                // Merge u and v into newIdx
//                clusterArr[u] = newIdx;
//                clusterArr[v] = newIdx;
//                newIdx++;
//            }
//        }
//
//        // 处理未合并节点
//        for (int i = 0; i < numNodes; i++) {
//            if (clusterArr[i] == -1) {
//                clusterArr[i] = newIdx++;
//            }
//        }
//
//        Tensor cluster = torch.tensor(clusterArr).to(x.device(), torch.ScalarType.Long);
//
//        // 3. 聚合特征 (Coarsening)
//        // Merged nodes: x_new = max(x_u, x_v) + score * (x_u + x_v) (ASAP style) or just sum
//        // EdgePooling 原文: max pooling
//        Tensor newX = ClusterPooling.max_pool(cluster, x);
//
//        // 4. 重构 Edge Index
//        // 这一步比较繁琐，需要将 old_edge_index 的端点映射到 new cluster id
//        // 然后去重 (remove duplicate edges)
//        Tensor newRow = cluster.index_select(0, row);
//        Tensor newCol = cluster.index_select(0, col);
//
//        // 移除自环 (newRow != newCol)
//        Tensor mask = newRow.ne(newCol);
//        Tensor finalRow = newRow.masked_select(mask);
//        Tensor finalCol = newCol.masked_select(mask);
//
//        // Stack -> Unique (去重)
//        // unique(dim=1)
//        Tensor newedge_indexRaw = torch.stack(new TensorVector(finalRow, finalCol), 0);
//        // LibTorch unique 算子比较 tricky，这里简化返回未去重的（GNN 通常能容忍多重边）
//
//        Tensor newBatch = ClusterPooling.max_pool(cluster, batch.to(torch.ScalarType.Float)).to(torch.ScalarType.Long);
//
//        return new Tensor[]{newX, newedge_indexRaw, newBatch, cluster};
//    }
//}