package org.bytedeco.pytorch.geometric.transforms;


import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public class AdvancedStructureTransforms {

    /**
     * FeaturePropagation: 特征传播
     * 用于补全缺失节点特征 (Missing Features)
     * 原理：通过多次拉普拉斯平滑，将已知特征扩散到缺失节点
     */
    public static class FeaturePropagation implements BaseTransform {
        private int numIterations;

        public FeaturePropagation(int numIterations) {
            this.numIterations = numIterations;
        }

        @Override
        public GraphData apply(GraphData data) {
            // 假设 data.x 中存在 NaN 或 0 代表缺失，使用 mask 标记
            Tensor x = data.x.clone();
            Tensor mask = data.x.norm(new ScalarOptional(new Scalar(0)), new long[]{1}).gt(new Scalar(0.0)); // 有特征的节点 mask

            // 1. 获取归一化邻接矩阵 A' = D^-1 * A
            Tensor adj = new AdvancedStructuralTransforms.ToDense().apply(data).adj;
            Tensor deg = adj.sum(1);
            Tensor dInv = deg.pow(new Scalar(-1.0)).masked_fill_(deg.eq(new Scalar(0.0)), new Scalar(0.0));
            Tensor p = diag(dInv).matmul(adj); // 转移矩阵

            // 2. 迭代传播
            for (int i = 0; i < numIterations; i++) {
                x = p.matmul(x);
                // 关键点：保持已知特征节点的特征不变 (Keep known features)
                x.index_put_(new TensorIndexVector(new TensorIndex(mask)), data.x.index(new TensorIndexVector(new TensorIndex(mask))));
            }
            data.x = x;
            return data;
        }
    }

    /**
     * HalfHop: 图上采样增强
     * 通过在每条边中间插入“虚拟节点”，减缓消息传递速度，缓解过平滑
     */
    public static class HalfHop implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            long numNodes = data.numNodes();

            // 1. 使用 select(0, index) 确保拿到的是 1D 向量 [numEdges]
            Tensor row = data.edge_index.select(0, 0);
            Tensor col = data.edge_index.select(0, 1);
            long numEdges = row.size(0);

            // 2. 扩充特征矩阵：增加虚拟节点
            // vNodeFeats: [numEdges, numFeatures]
            Tensor vNodeFeats = zeros(new long[]{numEdges, data.x.size(1)}, data.x.options());
            data.x = cat(new TensorVector(data.x, vNodeFeats), 0);

            // 3. 构建新边
            // dummyIndices: [numEdges] 的 1D 向量
            Tensor dummyIndices = arange(new Scalar(numNodes), new Scalar(numNodes + numEdges), data.edge_index.options());

            // 现在 row, col, dummyIndices 都是 1D [numEdges]
            // stack 后会变成 2D [2, numEdges]
            Tensor u2d = stack(new TensorVector(row, dummyIndices), 0);
            Tensor d2v = stack(new TensorVector(dummyIndices, col), 0);

            // 4. 拼接新边：[2, numEdges] cat [2, numEdges] -> [2, 2*numEdges]
            data.edge_index = cat(new TensorVector(u2d, d2v), 1);

            return data;
        }
    
//        @Override
        public GraphData call2(GraphData data) {
            long numNodes = data.x.size(0);
            Tensor row = data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(0))));
            Tensor col = data.edge_index.index(new TensorIndexVector(new TensorIndex(tensor(1))));
            long numEdges = row.size(0);

            // 1. 扩充特征矩阵：增加 numEdges 个虚拟节点
            Tensor vNodeFeats = zeros(new long[]{numEdges, data.x.size(1)}, data.x.options());
            data.x = cat(new TensorVector(data.x, vNodeFeats), 0);

            // 2. 构建新边：原边 (u, v) 变为 (u, dummy) 和 (dummy, v)
            Tensor dummyIndices = arange(new Scalar(numNodes), new Scalar(numNodes + numEdges), data.edge_index.options());

            Tensor u2d = stack(new TensorVector(row, dummyIndices), 0);
            Tensor d2v = stack(new TensorVector(dummyIndices, col), 0);

            data.edge_index = cat(new TensorVector(u2d, d2v), 1);
            return data;
        }
    }

    /**
     * AddGPSE: 结合结构与位置的综合编码
     * 结合了 RWPE (局部指纹) 和 LapPE (全局坐标) 的增强算子
     */
    public static class AddGPSE implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            // 组合执行 Laplacian 和 RandomWalk
            data = new SpectralAndStructuralTransforms.AddLaplacianEigenvectorPE(8).apply(data);
            data = new SpectralAndStructuralTransforms.AddRandomWalkPE(16).apply(data);
            return data;
        }
    }
}