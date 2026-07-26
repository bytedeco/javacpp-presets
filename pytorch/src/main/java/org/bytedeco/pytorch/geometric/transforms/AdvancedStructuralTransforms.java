package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public class AdvancedStructuralTransforms {

    /**
     * ToDense: 将稀疏邻接表转换为稠密邻接矩阵
     * 适用于池化后的子图处理 [N, N]
     */
     public static class ToDense implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            long N = data.x.size(0);
            Tensor adj = zeros(new long[]{N, N}, data.x.options());
            Tensor ones = ones(new long[]{data.edge_index.size(1)}, data.x.options());
            // 使用 index_put 将边位置填 1
            adj.index_put_(new TensorIndexVector(new TensorIndex(data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(0))))),
                    new TensorIndex(data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1)))))), ones);
            data.adj = adj;
            return data;
        }
    }

    /**
     * TwoHop: 增加二阶边 (i -> j -> k => i -> k)
     * 增加图的连通性，缩短长程通信距离
     */
    public static class TwoHop implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            // 简单实现：通过稀疏矩阵乘法 A * A
            Tensor adj = new ToDense().apply(data).adj;
            Tensor twoHopAdj = adj.matmul(adj).gt(new Scalar(0)).toType(kLong());
            data.edge_index = twoHopAdj.nonzero().t();
            return data;
        }
    }

    /**
     * GCNNorm: 经典的 GCN 归一化 A' = D^-0.5 * (A + I) * D^-0.5
     */
    public static class GCNNorm implements BaseTransform {
//        @Override
//        public GraphData apply(GraphData data) {
//            long N = data.x.size(0);
//            // 1. 添加自环
//            Tensor edge_index = add_remaining_self_loops(data.edge_index, N);
//            // 2. 计算度 D
//            Tensor deg = zeros(new long[]{N}, data.x.options());
//            deg.scatter_add_(0, edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1)))), ones(new long[]{edge_index.size(1)}, data.x.options()));
//            // 3. 计算 D^-0.5
//            Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
//            degInvSqrt.masked_fill_(degInvSqrt.isinf(), new Scalar(0.0));
//
//            data.edge_weight = degInvSqrt.index_select(0, edge_index.index(new TensorIndexVector(new TensorIndex(tensor(0)))))
//                    .mul(degInvSqrt.index_select(0, edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1))))));
//            data.edge_index = edge_index;
//            return data;
//        }

        @Override
        public GraphData apply(GraphData data) {
            long N = data.numNodes();
            Tensor edgeIndex = data.edge_index;

            // 1. 添加自环 (假设你已经实现了这个工具类)
            // edgeIndex 形状应为 [2, E]
            Tensor loopEdgeIndex = add_remaining_self_loops(edgeIndex, N);

            // 2. 计算度 D
            // deg 形状 [N]
            Tensor deg = zeros(new long[]{N}, data.x.options());

            // 获取目标节点索引 (col)，即 edge_index[1]
            // 使用 .select(0, 1) 获取第 0 维的索引 1，返回的是 [E] 的 1D 张量
            Tensor col = loopEdgeIndex.select(0, 1);

            // 构造 src，形状必须也是 [E] 的 1D 张量，且与 col 长度一致
            long numEdges = loopEdgeIndex.size(1);
            Tensor values = ones(new long[]{numEdges}, data.x.options());

            // 执行 scatter_add_
            // 此时 deg [N], col [E], values [E] 都是 1D，符合 LibTorch 要求
            deg.scatter_add_(0, col, values);

            // 3. 计算 D^-0.5
            Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
            // 处理无穷大（度为0的情况）
            degInvSqrt.masked_fill_(degInvSqrt.isinf(),new Scalar(0));

            // 4. 计算对称归一化系数: w = D_i^-0.5 * D_j^-0.5
            Tensor row = loopEdgeIndex.select(0, 0);
            Tensor col_ = loopEdgeIndex.select(0, 1);
            Tensor edgeWeight = degInvSqrt.index_select(0, row).mul(degInvSqrt.index_select(0, col_));

            // 更新数据
            data.edge_index = loopEdgeIndex;
            data.edge_weight = edgeWeight;

            return data;
        }

        private Tensor add_remaining_self_loops(Tensor edge_index, long N) {
            Tensor loop = arange(new Scalar(0),new Scalar(N) , edge_index.options());
            return cat(new TensorVector(edge_index, stack(new TensorVector(loop, loop), 0)), 1);
        }
    }

    /**
     * SIGN: 预计算多阶算子特征 (Inception 风格)
     * 将 X, AX, A^2X, A^3X ... 拼接
     */
    public static class SIGN implements BaseTransform {
        private int k;
        public SIGN(int k) { this.k = k; }

        @Override
        public GraphData apply(GraphData data) {
            // 预先进行 GCN 归一化
            data = new GCNNorm().apply(data);
            Tensor x = data.x;
            java.util.List<Tensor> xs = new java.util.ArrayList<>();
            xs.add(x);

            Tensor currentX = x;
            for (int i = 0; i < k; i++) {
                // 模拟简单的邻域聚合: X_next = D^-0.5 A D^-0.5 * X
                // 这里使用 scatter/gather 实现稀疏聚合
                currentX = aggregate(currentX, data.edge_index, data.edge_weight);
                xs.add(currentX);
            }
            // 拼接特征: [N, D * (K+1)]
            data.x = cat(new TensorVector(xs.toArray(new Tensor[0])), 1);
            return data;
        }

        private Tensor aggregate2(Tensor x, Tensor edge_index, Tensor edgeWeight) {
            Tensor row = edge_index.index(new TensorIndexVector(new TensorIndex(tensor(0))));
            Tensor col = edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1))));
            Tensor msg = x.index_select(0, row).mul(edgeWeight.view(-1, 1));
            Tensor out = zeros_like(x);
            out.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg);
            return out;
        }

        private Tensor aggregate(Tensor x, Tensor edge_index, Tensor edgeWeight) {
            // 使用 select(0, 0) 明确获取第一行，并降维成 [num_edges] 的 1D 向量
            Tensor row = edge_index.select(0, 0);
            Tensor col = edge_index.select(0, 1);

            // 检查 edgeWeight 形状，确保它是 [num_edges, 1] 以便进行 broadcast
            // 如果 edgeWeight 是 [num_edges]，view(-1, 1) 是正确的
            Tensor weight = edgeWeight.view(new long[]{-1, 1});

            // 此时 row 是 1D vector，index_select 不再报错
            Tensor msg = x.index_select(0, row).mul(weight);

            Tensor out = zeros_like(x);

            // scatter_add_ 的 index 必须与 msg 维度一致
            // col.unsqueeze(-1) 将 [E] 变为 [E, 1]
            // expand_as(msg) 将 [E, 1] 变为 [E, num_features]
            Tensor scatterIdx = col.unsqueeze(-1).expand_as(msg);

            out.scatter_add_(0, scatterIdx, msg);

            return out;
        }
    }
}