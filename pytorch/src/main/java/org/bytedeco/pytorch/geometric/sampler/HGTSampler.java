package org.bytedeco.pytorch.geometric.sampler;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;

import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import java.util.HashMap;
import java.util.Map;
import static org.bytedeco.pytorch.global.torch.*;

public class HGTSampler {
    private final HeteroAdj adj;

    public HGTSampler(HeteroAdj adj) {
        this.adj = adj;
    }

    /**
     * @param seedNodes 初始节点 (nodeType -> tensor)
     * @param numLayers 采样层数
     * @param nodeBudget 每一层每种节点类型的最大采样数量
     */
    public HeteroSamplerOutput sample(Map<String, Tensor> seedNodes, int numLayers, int nodeBudget) {
        HeteroSamplerOutput output = new HeteroSamplerOutput();
        Map<String, Tensor> currentLayerSeeds = new HashMap<>(seedNodes);

        // 初始化 output，将种子节点存入
        for (Map.Entry<String, Tensor> entry : seedNodes.entrySet()) {
            output.nodeIds.put(entry.getKey(), entry.getValue().detach());
        }

        for (int i = 0; i < numLayers; i++) {
            Map<String, Tensor> nextLayerNodes = new HashMap<>();

            for (Map.Entry<String, Tensor> entry : currentLayerSeeds.entrySet()) {
                String nodeType = entry.getKey();
                Tensor seeds = entry.getValue();

                // 1. 聚合当前类型种子节点的所有潜在邻居
                Tensor candidates = getAggregatedNeighbors(nodeType, seeds);

                if (candidates.numel() > 0) {
                    Tensor selected;
                    if (candidates.size(0) <= nodeBudget) {
                        selected = candidates;
                    } else {
                        // 2. 预算控制：简单随机采样 (HGT 核心步骤)
                        // 创建全 1 权重进行等概率采样
                        Tensor weights = ones(new long[]{candidates.size(0)},
                                seeds.options().dtype(new ScalarTypeOptional(kFloat())));
                        Tensor indices = multinomial(weights, nodeBudget, false, null);
                        selected = candidates.index_select(0, indices);
                    }

                    // 3. 更新下一层种子并合并到总输出
                    nextLayerNodes.put(nodeType, selected);

                    Tensor existing = output.nodeIds.getOrDefault(nodeType,
                            empty(new long[]{0}, seeds.options(), new MemoryFormatOptional()));
                    Tensor combined = cat(new TensorVector(existing, selected), 0);
                    // 使用你提供的 unique 逻辑进行去重
                    output.nodeIds.put(nodeType, unique_consecutive(combined).get0().detach());
                }
            }
            currentLayerSeeds = nextLayerNodes;
            if (currentLayerSeeds.isEmpty()) break;
        }
        return output;
    }

    private Tensor getAggregatedNeighbors(String nodeType, Tensor seeds) {
        TensorVector results = new TensorVector();

        for (String edgeType : adj.rowPtr.keySet()) {
            // HGT 采样通常考虑指向该 nodeType 的边 (Target-based)
            // 假设格式: "src__rel__dest"，匹配 dest == nodeType
            if (edgeType.endsWith(nodeType)) {
                Tensor ptr = adj.rowPtr.get(edgeType);
                Tensor col = adj.colIndex.get(edgeType);

                Tensor neighbors = rawSliceNeighbors(ptr, col, seeds);
                if (neighbors.numel() > 0) {
                    results.put(neighbors);
                }
            }
        }

        if (results.empty()) {
            return empty(new long[]{0}, seeds.options(), new MemoryFormatOptional());
        }
        return unique_consecutive(cat(results, 0)).get0().detach();
    }

    private Tensor rawSliceNeighbors(Tensor ptr, Tensor col, Tensor seeds) {
        TensorVector collected = new TensorVector();
        for (int i = 0; i < seeds.size(0); i++) {
            // 使用严格的 API: TensorIndexVector + TensorIndex
            long nodeIdx = seeds.index(new TensorIndexVector(new TensorIndex(tensor(i)))).item().toLong();
            long start = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx)))).item().toLong();
            long end = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx + 1)))).item().toLong();

            if (end > start) {
                collected.put(col.slice(0, new LongOptional(start), new LongOptional(end), 1));
            }
        }
        return collected.empty() ?
                empty(new long[]{0}, seeds.options(), new MemoryFormatOptional()) : cat(collected, 0);
    }
}
//public class HGTSampler {
//    private Map<String, Tensor> adjRowPtr, adjColIndex;
//    private Map<String, Tensor> nodeTimes; // HGT 往往依赖时间戳过滤
//
//    public HeteroSamplerOutput sample(Map<String, Tensor> seedNodes, int numLayers, int nodeBudget) {
//        try (PointerScope scope = new PointerScope()) {
//            HeteroSamplerOutput output = new HeteroSamplerOutput();
//            Map<String, Tensor> currentSeeds = seedNodes;
//
//            for (int layer = 0; layer < numLayers; layer++) {
//                Map<String, Tensor> nextLayerSeeds = new HashMap<>();
//
//                for (String nodeType : currentSeeds.keySet()) {
//                    Tensor seeds = currentSeeds.get(nodeType);
//
//                    // 1. 获取所有候选邻居
//                    Tensor candidates = getAllNeighbors(seeds);
//
//                    // 2. 计算采样权重 (Importance)
//                    // HGT 逻辑：Weight = 1 / degree(candidate) 或基于时间戳
//                    Tensor weights = ones_like(candidates, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())),new MemoryFormatOptional());
//
//                    // 3. 预算内采样 (Multinomial)
//                    long numToSample = Math.min(candidates.size(0), (long)nodeBudget);
//                    Tensor selectedIdx = multinomial(weights, numToSample, false,new GeneratorOptional());
//                    Tensor sampledNodes = candidates.index_select(0, selectedIdx);
//
//                    nextLayerSeeds.put(nodeType, sampledNodes);
//                    output.nodeIds.put(nodeType, sampledNodes);
//                }
//                currentSeeds = nextLayerSeeds;
//            }
//            return output;
//        }
//    }
//
//    private Tensor getAllNeighbors(Tensor seeds) {
//        // 核心：基于 CSR 快速检索所有关联节点
//        return arange(new Scalar(0),new Scalar(10) , new TensorOptions().dtype(new ScalarTypeOptional(kLong()))); // 示意
//    }
//}