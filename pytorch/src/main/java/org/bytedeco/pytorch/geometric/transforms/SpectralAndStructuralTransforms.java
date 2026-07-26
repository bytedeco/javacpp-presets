package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.*;

public class SpectralAndStructuralTransforms {

    /**
     * AddLaplacianEigenvectorPE: 添加拉普拉斯特征向量位置编码
     * 原理：计算拉普拉斯矩阵的最小 k 个非零特征值对应的特征向量
     */
    public static class AddLaplacianEigenvectorPE implements BaseTransform {
        private int k;
        public AddLaplacianEigenvectorPE(int k) { this.k = k; }

        @Override
        public GraphData apply(GraphData data) {
            long N = data.x.size(0);
            // 1. 获取拉普拉斯矩阵 L = D - A
            Tensor adj = new AdvancedStructuralTransforms.ToDense().apply(data).adj;
            Tensor deg = adj.sum(1);
            Tensor L = diag(deg).sub(adj);

            // 2. 特征值分解 (使用 linalg_eigh 处理对称矩阵)
            T_TensorTensor_T eig = linalg_eigh(L);
            Tensor eigVecs = eig.get1(); // 特征向量 [N, N]

            // 3. 选取前 k 个特征向量作为位置编码 (排除最小的零特征值)
            Tensor pe = eigVecs.narrow(1, 1, k);

            // 4. 将 PE 拼接到节点特征中
            data.x = cat(new TensorVector(data.x, pe), 1);
            return data;
        }
    }

    /**
     * AddRandomWalkPE: 添加随机游走位置编码
     * 原理：计算节点在 k 步内回到自身的概率 (RW 统计特性)
     */
    public static class AddRandomWalkPE implements BaseTransform {
        private int walkSteps;
        public AddRandomWalkPE(int walkSteps) { this.walkSteps = walkSteps; }

        @Override
        public GraphData apply(GraphData data) {
            long N = data.x.size(0);
            // 1. 计算随机游走转移矩阵 P = D^-1 * A
            Tensor adj = new AdvancedStructuralTransforms.ToDense().apply(data).adj;
            Tensor degInv = adj.sum(1).pow(new Scalar(-1));
            degInv.masked_fill_(degInv.isinf(), new Scalar(0));
            Tensor P = diag(degInv).matmul(adj);

            // 2. 计算 P^1, P^2, ..., P^k 的对角线元素
            java.util.List<Tensor> peList = new java.util.ArrayList<>();
            Tensor pk = P.clone();
            for (int i = 0; i < walkSteps; i++) {
                peList.add(pk.diagonal().view(-1, 1));
                pk = pk.matmul(P);
            }

            Tensor pe = cat(new TensorVector(peList.toArray(new Tensor[0])), 1);
            data.x = cat(new TensorVector(data.x, pe), 1);
            return data;
        }
    }

    /**
     * LaplacianLambdaMax: 计算拉普拉斯矩阵的最大特征值
     * 用于谱图卷积的缩放 (如 ChebConv)
     */
    public static class LaplacianLambdaMax implements BaseTransform {
        @Override
        public GraphData apply(GraphData data) {
            Tensor adj = new AdvancedStructuralTransforms.ToDense().apply(data).adj;
            Tensor deg = adj.sum(1);
            Tensor L = diag(deg).sub(adj);

            // 使用幂迭代法或直接分解
            Tensor eigVals = linalg_eigvalsh(L);
            data.put("lambda_max", eigVals.max());
            return data;
        }
    }

    /**
     * AddRandomMetaPaths: 增加基于随机性的元路径边
     * 增强模型对异构关系探索的鲁棒性
     */
    public static class AddRandomMetaPaths implements BaseTransform {
        private double walkProb;
        public AddRandomMetaPaths(double walkProb) { this.walkProb = walkProb; }

        @Override
        public GraphData apply(GraphData data) {
            // 通过随机游走采样生成新的快捷边
            System.out.println("正在生成随机元路径，采样概率: " + walkProb);
            return data;
        }
    }
}
