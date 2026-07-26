package org.bytedeco.pytorch.geometric.model;
import org.bytedeco.pytorch.data.sampler.*;
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
import org.bytedeco.pytorch.geometric.sampler.NeighborSampler;
import org.bytedeco.pytorch.geometric.sampler.NumNeighbors;

import java.util.*;

import static org.bytedeco.pytorch.global.torch.*;

public class GNNTrainingExample {
    public static void main(String[] args) {
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
        TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        try (PointerScope scope = new PointerScope()) {
            // --- 1. 数据准备 ---
            long numNodes = 1000;
            Tensor x = randn(new long[]{numNodes, 16}, fOpts); // 初始特征
            // 模拟类别用于 Diversity 指标
            Tensor itemCats = randint(0, 5, new long[]{numNodes}, longOpts);

            // --- 2. 采样器配置 (使用我们实现的 Sampler) ---
            HeteroAdj adj = createMockAdj(numNodes);
            NeighborSampler2 sampler = new NeighborSampler2(adj.rowPtr, adj.colIndex);

            // --- 3. 模型与优化器 ---
            GraphSAGEModel model = new GraphSAGEModel(16, 32);
            Adam optimizer = new Adam(model.parameters(), new AdamOptions(0.01));

            // --- 4. 训练循环 ---
            for (int epoch = 1; epoch <= 10; epoch++) {
                model.train(true);

                // 获取一批正/负边 (EdgeSamplerInput 逻辑)
                Tensor posEdges = randint(0, numNodes, new long[]{2, 64}, fOpts);
                Tensor negEdges = randint(0, numNodes, new long[]{2, 64}, fOpts);

                // 2. 采样（获取邻居结构）---------
                Map<String, Tensor> seeds = new HashMap<>();
                seeds.put("node", unique_consecutive(posEdges.view(-1)).get0()); // 以目标边的节点为种子
                HeteroSamplerOutput sampled = sampler.sampleFromNodes(seeds, new NumNeighbors(10, 5));
                Tensor globalFlowEdges = sampler.convertToCOO(sampled.row, sampled.col);
                Tensor allNodes = sampled.nodeIds.get("node");
                // 3. 构建全量局部索引 (包含种子节点和所有邻居)
// 我们需要把所有采样涉及到的节点提取出来做映射
                Tensor allNodesInSubgraph = sampled.nodeIds.get("node");
//                ReindexResult subgraphData = ReindexResult.generateMapping(allNodesInSubgraph,globalFlowEdges);

// 4. 重映射卷积用的边 (用于 forward)
//                Tensor globalFlowEdges = sampler.convertToCOO(sampled.row, sampled.col);
//                Tensor localFlowEdges = ReindexResult.remap(globalFlowEdges, subgraphData.nodeMapping);

                // 5. 重映射打分用的边 (用于 score)
//                Tensor localPosEdges = ReindexResult.remap(posEdges, subgraphData.nodeMapping);

                ReindexResult reindexed = ReindexResult.generateMapping(allNodes, globalFlowEdges);
                Tensor xSub = x.index_select(0, reindexed.nodeMapping);

// 4. 前向传播 (z 的行数将等于 xSub 的行数)
                Tensor z = model.forward(xSub, reindexed.localEdgeIndex);


// 5. 对正样本边进行同样的重映射 (复用 nodeMapping 字典)
                Tensor localPosEdges = searchsorted(reindexed.nodeMapping, posEdges);
                Tensor posScore = model.score(z, localPosEdges);
// 6. 提取局部特征并前向传播
//                Tensor xSub = x.index_select(0, subgraphData.nodeMapping);
//                Tensor z = model.forward(xSub, localFlowEdges); // 这里的 z 不再为空

// 7. 计算得分
//                Tensor posScore = model.score(z, localPosEdges);
                
//                // 采样节点邻居 (NeighborSampler)
//                Map<String, Tensor> seeds = new HashMap<>();
//                seeds.put("node", unique_consecutive(posEdges.view(-1)).get0());
//                HeteroSamplerOutput sampled = sampler.sampleFromNodes(seeds, new NumNeighbors(10, 5));

                // 前向传播
                optimizer.zero_grad();
                
//                Tensor globalEdgeIndex = sampler.convertToCOO(sampled.row, sampled.col);
// 2. 重映射 (核心步骤)
//                ReindexResult reindexed = ReindexResult.reindex(globalEdgeIndex);
//                Tensor posScore = model.score(z, reindexed.localEdgeIndex);
// 3. 提取局部特征 (只取采样到的节点特征)
// 这样即使原图有 1 亿个节点，这里也只处理被采样的几百个节点
//                Tensor xSub = x.index_select(0, reindexed.nodeMapping);

// 4. 前向传播
//                Tensor z = model.forward(xSub, reindexed.localEdgeIndex);
                
                //2 可用
//                Tensor edgeIndex = sampler.convertToCOO(sampled.row, sampled.col);
//                Tensor z = model.forward(x, edgeIndex);
                
                //3 失败了
//                Tensor z = model.forward(x, sampled.col.get("edge")); // 简化版子图计算

//                Tensor posScore = model.score(z, posEdges);
                Tensor negScore = model.score(z, negEdges);

                // BPR Loss: -log(sigmoid(pos - neg))
                Tensor loss = log(sigmoid(posScore.sub(negScore))).mul(new Scalar(-1)).mean();
                loss.backward();
                optimizer.step();

                if (epoch % 2 == 0) {
                    // --- 5. 评估过程 (使用我们实现的 Metrics) ---
                    model.eval();
                    System.out.println("Epoch " + epoch + " | Loss: " + loss.item().toFloat());

                    runEvaluation(model, z, itemCats);
                }
            }
        }
    }

    public static HeteroAdj createMockAdj(long numNodes) {
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
        TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
//        try (PointerScope scope = new PointerScope()) {
            // 1. 模拟每条边：每个节点随机连接 5 个邻居
            long numEdgesPerNode = 5;
            long totalEdges = numNodes * numEdgesPerNode;

            // 构建 rowPtr: [0, 5, 10, 15, ...]
            long[] rowPtrArr = new long[(int) numNodes + 1];
            for (int i = 0; i <= numNodes; i++) {
                rowPtrArr[i] = i * numEdgesPerNode;
            }
            Tensor rowPtr = tensor(rowPtrArr, longOpts).detach();

            // 构建 colIndex: 随机生成目标节点
            Tensor colIndex = randint(0, numNodes, new long[]{totalEdges}, longOpts).detach();

            // 2. 封装进 HeteroAdj (支持异构，这里我们定义一种边类型)
            HeteroAdj adj = new HeteroAdj();
            adj.addEdgeType("node__edge__node", rowPtr, colIndex);
            adj.addEdgeType("node", rowPtr, colIndex);

            System.out.println("✅ MockAdj 创建成功: 节点数=" + numNodes + ", 边数=" + totalEdges);
            return adj;
//        }
    }
    public static HeteroAdj createMockAdj2(long numNodes) {
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
        TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        HeteroAdj adj = new HeteroAdj();
        // 模拟 User 到 Item 的边
        long edgesPerNode = 3;
        long[] ptrArr = new long[(int)numNodes + 1];
        for(int i=0; i<=numNodes; i++) ptrArr[i] = i * edgesPerNode;

        Tensor rowPtr = tensor(ptrArr, longOpts);
        Tensor colIdx = randint(0, numNodes, new long[]{numNodes * edgesPerNode},longOpts);

        adj.addEdgeType("user__to__item", rowPtr, colIdx);
        return adj;
    }

    public void testIntegratedFlow() {
        System.out.println("=== 启动 GNN 全流程集成测试 ===");
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
        TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        // 1. 创建数据
        HeteroAdj adj = createMockAdj(100);

        // 2. 初始化采样器 (使用你要求的 API 格式)
        NeighborSampler2 sampler = new NeighborSampler2(adj.rowPtr, adj.colIndex);

        // 3. 执行采样
        Map<String, Tensor> seeds = new HashMap<>();
        seeds.put("user", tensor(new long[]{1, 5, 10}, longOpts));
        HeteroSamplerOutput result = sampler.sampleFromNodes(seeds, new NumNeighbors(10, 5));

        System.out.println("✅ 采样成功，节点类型 'user' 采样总数: " + result.nodeIds.get("user").size(0));

        // 4. 运行评估 (Metric)
        LinkPredMetricCollection metrics = new LinkPredMetricCollection();
        metrics.addMetric("MRR@5", new LinkPredMRR(5));

        Tensor yPred = rand(new long[]{3, 50}, fOpts); // 3个种子的预测
        Tensor yTrue = zeros_like(yPred); // 简化标签

        Map<String, Double> evalResults = metrics.computeAll(yPred, yTrue);
        System.out.println("✅ 评估成功: " + evalResults);
    }
    private static void runEvaluation(GraphSAGEModel model, Tensor z, Tensor itemCats) {
        // 模拟预测所有节点对的得分 [Batch, NumItems]
        TensorOptions longOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long));
        TensorOptions fOpts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        Tensor yPred = rand(new long[]{10, 100}, fOpts);
        Tensor yTrue = zeros_like(yPred);
        Tensor a1 = arange(new Scalar(0), new Scalar(10), longOpts);
        Tensor b2 = randint(0, 100, new long[]{10}, longOpts);
        Tensor c3 = ones(new long[]{10}, fOpts);
//        Tensor[] tensors = new Tensor[]{arange(new Scalar(0), new Scalar(10), longOpts), randint(0, 100, new long[]{10}, longOpts)}, ones(new long[]{10}, fOpts));
        TensorIndexVector indices = new TensorIndexVector(new TensorIndex(a1), new TensorIndex(b2));

        yTrue.index_put_(indices,c3);

        LinkPredMetricCollection metrics = new LinkPredMetricCollection();
        metrics.addMetric("NDCG@10", new LinkPredNDCG(10));
        metrics.addMetric("Diversity@10", new LinkPredDiversity(10, itemCats));
        metrics.addMetric("MRR@10", new LinkPredMRR(10));

        Map<String, Double> results = metrics.computeAll(yPred, yTrue);
        System.out.println("📊 评估报告: " + results);
    }
}