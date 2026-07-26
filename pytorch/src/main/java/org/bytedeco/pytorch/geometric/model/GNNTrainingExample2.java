package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.AdamOptions;
import org.bytedeco.pytorch.geometric.metrics.LinkPredDiversity;
import org.bytedeco.pytorch.geometric.metrics.LinkPredMRR;
import org.bytedeco.pytorch.geometric.metrics.LinkPredMetricCollection;
import org.bytedeco.pytorch.geometric.metrics.LinkPredNDCG;
import org.bytedeco.pytorch.geometric.sampler.HeteroAdj;
import org.bytedeco.pytorch.geometric.sampler.HeteroSamplerOutput;
import org.bytedeco.pytorch.geometric.sampler.NumNeighbors;

import java.util.*;
import static org.bytedeco.pytorch.global.torch.*;
import static org.bytedeco.pytorch.geometric.model.GNNTrainingExample.createMockAdj;

public class GNNTrainingExample2 {
    public static void main(String[] args) {
        System.out.println("=== 启动生产级 GNN 训练流水线 ===");

        // 基础配置
        long numNodes = 1000;
        long hiddenChannels = 32;
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
        TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));

        // 1. 初始化全量特征和模拟邻接表
        // 注意：x 需要开启梯度（如果特征是可学习的），或者作为常量输入
        Tensor x = randn(new long[]{numNodes, 16}, fOpts);
        Tensor itemCats = randint(0, 5, new long[]{numNodes}, longOpts);
        HeteroAdj adj = createMockAdj(numNodes);

        // 2. 初始化模型、优化器和采样器
        GraphSAGEModel model = new GraphSAGEModel(16, hiddenChannels);
        Adam optimizer = new Adam(model.parameters(), new AdamOptions(0.01));
        NeighborSampler2 sampler = new NeighborSampler2(adj.rowPtr, adj.colIndex);

        // 3. 训练循环
        for (int epoch = 1; epoch <= 100; epoch++) {
            // 注意：每个 batch/epoch 建议开启一个新的 PointerScope 来防止 C++ 内存堆积
            try (PointerScope batchScope = new PointerScope()) {
                model.train(true);
                optimizer.zero_grad();

                // --- A. 准备种子节点 (正样本边) ---
                Tensor posEdges = randint(0, numNodes, new long[]{2, 64}, longOpts);
                Tensor negEdges = randint(0, numNodes, new long[]{2, 64}, longOpts);

                // 以正向边的所有节点作为采样起点
                Map<String, Tensor> seeds = new HashMap<>();
//                seeds.put("node", posEdges.view(-1).unique());
                seeds.put("node", unique_consecutive(posEdges.view(-1)).get0());
                // --- B. 采样与重映射 ---
                // 1. 采样获取子图结构
                HeteroSamplerOutput sampled = sampler.sampleFromNodes(seeds, new NumNeighbors(10, 5));
                // 1. 防御性检查：确保采样到了节点
                if (!sampled.nodeIds.containsKey("node") || sampled.nodeIds.get("node") == null) {
                    throw new RuntimeException("采样失败：Key 'node' 未找到。请检查 HeteroAdj 定义的节点类型是否为 'node'");
                }
                Tensor globalFlowEdges = sampler.convertToCOO(sampled.row, sampled.col);

                // 2. 核心：重映射！将全局 ID 空间压缩到局部 [0, numSampledNodes)
                // 使用你要求的 generateMapping，它内部调用了构造函数 node__edge__node
//                ReindexResult reindexed = ReindexResult.generateMapping(sampled.nodeIds.get("node__edge__node"), globalFlowEdges);

                ReindexResult reindexed = ReindexResult.generateMapping(sampled.nodeIds.get("node"), globalFlowEdges);

                // 3. 提取局部特征 (xSub 维度与 reindexed.nodeMapping 对齐)
                Tensor xSub = x.index_select(0, reindexed.nodeMapping);

                // --- C. 前向传播与计算 Loss ---
                // 计算节点嵌入 (z 的大小为 [numSampledNodes, hiddenChannels])
                Tensor z = model.forward(xSub, reindexed.localEdgeIndex);
// 重点：使用 side_reindex 逻辑，确保打分边不会越界
// 我们需要将 posEdges 中的全局 ID 映射到 reindexed.nodeMapping 的局部索引中
                Tensor localPosEdges = safeRemap(posEdges, reindexed.nodeMapping);
                Tensor localNegEdges = safeRemap(negEdges, reindexed.nodeMapping);
                // 将目标边也映射到局部坐标系，否则 score 会越界
//                Tensor localPosEdges = searchsorted(reindexed.nodeMapping, posEdges);
//                Tensor localNegEdges = searchsorted(reindexed.nodeMapping, negEdges);

                Tensor posScore = model.score(z, localPosEdges);
                Tensor negScore = model.score(z, localNegEdges);

                // BPR Loss: 鼓励正样本得分高于负样本
                Tensor loss = log(sigmoid(posScore.sub(negScore))).mul(new Scalar(-1)).mean();

                // --- D. 反向传播 ---
                loss.backward();
                optimizer.step();

                // --- E. 评估 ---
                if (epoch % 2 == 0) {
                    System.out.printf("Epoch %d | Loss: %.4f\n", epoch, loss.item().toFloat());
                    model.eval();
                    // 运行我们之前实现的评估函数
                    runEvaluation(model, z, itemCats, reindexed.nodeMapping);
                }

                // 显式让某些 Tensor 脱离 Scope 以便观察或下一轮使用（如有必要）
                // 正常情况下，batchScope 关闭时会清理所有临时 Tensor
            }
        }
        System.out.println("✅ 训练流水线运行完成！");
    }

    public static Tensor safeRemap(Tensor globalEdges, Tensor nodeMapping) {
//        try (PointerScope scope = new PointerScope()) {
            // searchsorted 找到位置
            Tensor localIndices = searchsorted(nodeMapping, globalEdges);

            // 关键防护：如果索引等于 nodeMapping 的长度，说明该节点不在采样范围内
            // 强制裁剪到最大有效索引（或者在采样时确保 seeds 包含了所有 posEdges 的节点）
            long maxValidIdx = nodeMapping.size(0) - 1;
            Tensor clampedIndices = clamp(localIndices,new ScalarOptional(new Scalar(0)) ,new ScalarOptional(new Scalar(maxValidIdx)));

            return clampedIndices.detach();
//        }
    }
    /**
     * 修正后的评估方法
     */
    private static void runEvaluation(GraphSAGEModel model, Tensor z, Tensor itemCats, Tensor nodeMapping) {
//        try (PointerScope evalScope = new PointerScope()) {
            // 获取局部节点对应的类别
            TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
            TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));

            Tensor localCats = itemCats.index_select(0, nodeMapping);

            // 构造模拟的评估数据 [Batch, NumItems]
            long batchSize = 10;
            long numItems = z.size(0); // 在局部嵌入空间内评估
            Tensor yPred = rand(new long[]{batchSize, numItems}, z.options());

            Tensor yTrue = zeros(new long[]{batchSize, numItems}, z.options());
            Tensor rowIdx = arange(new Scalar(0), new Scalar(batchSize),longOpts);
            Tensor colIdx = randint(0, numItems, new long[]{batchSize}, longOpts);
            yTrue.index_put_(new TensorIndexVector(new TensorIndex(rowIdx), new TensorIndex(colIdx)),
                    ones(new long[]{batchSize}, z.options()));

            LinkPredMetricCollection metrics = new LinkPredMetricCollection();
            metrics.addMetric("NDCG@5", new LinkPredNDCG(5));
            metrics.addMetric("Diversity@5", new LinkPredDiversity(5, localCats));
            metrics.addMetric("MRR@5", new LinkPredMRR(5));

            Map<String, Double> results = metrics.computeAll(yPred, yTrue);
            System.out.println("📊 评估报告: " + results);
//        }
    }
}