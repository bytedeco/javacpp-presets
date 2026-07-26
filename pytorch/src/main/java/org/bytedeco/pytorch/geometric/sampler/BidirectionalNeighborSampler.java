package org.bytedeco.pytorch.geometric.sampler;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class BidirectionalNeighborSampler {
    private final HeteroAdj forwardAdj;
    private final HeteroAdj backwardAdj;

    public BidirectionalNeighborSampler(HeteroAdj forward, HeteroAdj backward) {
        this.forwardAdj = forward;
        this.backwardAdj = backward;
    }

    public HeteroSamplerOutput sample(Map<String, Tensor> seedNodes, int[] numNeighbors) {
        // 使用 PointerScope 确保中间产生的临时 Tensor 被回收
//        try (PointerScope scope = new PointerScope()) {
            HeteroSamplerOutput output = new HeteroSamplerOutput();

            for (Map.Entry<String, Tensor> entry : seedNodes.entrySet()) {
                String nodeType = entry.getKey();
                Tensor seeds = entry.getValue();

                // 1. 下游采样 (Forward)
                Tensor fwNeighbors = sampleFromSpecificAdj(forwardAdj, nodeType, seeds, numNeighbors[0]);

                // 2. 上游采样 (Backward)
                Tensor bwNeighbors = sampleFromSpecificAdj(backwardAdj, nodeType, seeds, numNeighbors[1]);

                // 3. 安全合并：必须确保 TensorVector 不为空
                TensorVector allTensors = new TensorVector();
                allTensors.put(seeds); // 种子节点永远不为空

                if (fwNeighbors != null && fwNeighbors.numel() > 0) {
                    allTensors.put(fwNeighbors);
                }
                if (bwNeighbors != null && bwNeighbors.numel() > 0) {
                    allTensors.put(bwNeighbors);
                }

                // 现在调用 cat 是安全的，因为至少有 seeds
                Tensor merged = cat(allTensors, 0);

                // 去重并放入输出 (unique 会返回一个新的 Tensor，需要调用 detach 保持其生命周期)
                output.nodeIds.put(nodeType, unique_consecutive(merged).get0().detach());
            }
            return output;
//        }
    }

    private Tensor sampleFromSpecificAdj(HeteroAdj adj, String nodeType, Tensor seeds, int count) {
        TensorVector results = new TensorVector();

        for (String edgeType : adj.rowPtr.keySet()) {
            // 假设边类型定义格式为: "sourceType__edgeName__targetType"
            if (edgeType.startsWith(nodeType)) {
                Tensor ptr = adj.rowPtr.get(edgeType);
                Tensor col = adj.colIndex.get(edgeType);

                Tensor sampled = rawNeighborSample(ptr, col, seeds, count);
                if (sampled != null && sampled.numel() > 0) {
                    results.put(sampled);
                }
            }
        }

        if (results.empty()) {
            // 返回一个形状为 [0] 的 LongTensor 而不是 null
            return empty(new long[]{0}, seeds.options(),new MemoryFormatOptional());
        }

        return cat(results, 0);
    }

    private Tensor rawNeighborSample(Tensor ptr, Tensor col, Tensor seeds, int count) {
        // 简化的采样逻辑：直接取每个种子的前 N 个邻居
        // 实际生产中应使用 torch.multinomial 或特殊的 C++ 算子
        TensorVector collected = new TensorVector();
        for (int i = 0; i < seeds.size(0); i++) {
            long nodeIdx = seeds.index(new TensorIndexVector(new TensorIndex(tensor(i)))).item().toLong();
            long start = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx)))).item().toLong();
            long end = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx + 1)))).item().toLong();

            long actualCount = Math.min(end - start, (long) count);
            if (actualCount > 0) {
                collected.put(col.slice(0, new LongOptional(start), new LongOptional(start + actualCount),1));
            }
        }
        return collected.empty() ? empty(new long[]{0}, seeds.options(),new MemoryFormatOptional()) : cat(collected, 0);
    }
}
//public class BidirectionalNeighborSampler2 {
//    private final HeteroAdj forwardAdj;
//    private final HeteroAdj backwardAdj;
//
//    public BidirectionalNeighborSampler(HeteroAdj forward, HeteroAdj backward) {
//        this.forwardAdj = forward;
//        this.backwardAdj = backward;
//    }
//
//    /**
//     * @param seedNodes 种子节点 (nodeType -> tensor)
//     * @param numNeighbors 采样数量 [forward_count, backward_count]
//     */
//    public HeteroSamplerOutput sample(Map<String, Tensor> seedNodes, int[] numNeighbors) {
////        try (PointerScope scope = new PointerScope()) {
//            HeteroSamplerOutput output = new HeteroSamplerOutput();
//
//            for (Map.Entry<String, Tensor> entry : seedNodes.entrySet()) {
//                String nodeType = entry.getKey();
//                Tensor seeds = entry.getValue();
//
//                // 1. 下游采样 (Forward)
//                Tensor fwNeighbors = sampleFromSpecificAdj(forwardAdj, nodeType, seeds, numNeighbors[0]);
//
//                // 2. 上游采样 (Backward)
//                Tensor bwNeighbors = sampleFromSpecificAdj(backwardAdj, nodeType, seeds, numNeighbors[1]);
//
//                // 3. 合并并去重
//                Tensor allNodes = cat(new TensorVector(seeds, fwNeighbors, bwNeighbors), 0);
//                output.nodeIds.put(nodeType, unique_consecutive(allNodes).get0());
//            }
//            return output;
////        }
//    }
//
//    private Tensor sampleFromSpecificAdj(HeteroAdj adj, String nodeType, Tensor seeds, int count) {
//        // 查找所有以该 nodeType 为源的边类型
//        TensorVector results = new TensorVector();
//        for (String edgeType : adj.rowPtr.keySet()) {
//            if (edgeType.startsWith(nodeType)) {
//                Tensor ptr = adj.rowPtr.get(edgeType);
//                Tensor col = adj.colIndex.get(edgeType);
//                // 调用基础的邻居切片算子 (见下文)
//                results.push_back(rawNeighborSample(ptr, col, seeds, count));
//                System.out.println("edgeType: " + edgeType + ", neighbors: " + results.get(results.size() - 1));
//            }
//        }
//        return cat(results, 0);
//    }
//
//    // 模拟底层 C++ 采样算子
//    private Tensor rawNeighborSample(Tensor ptr, Tensor col, Tensor seeds, int count) {
//        // 获取每个种子的起始和结束 offset
//        Tensor start = ptr.index_select(0, seeds);
//        Tensor end = ptr.index_select(0, seeds.add(new Scalar(1)));
//        // 简化实现：抽取每个节点的前 count 个邻居 (实际应使用随机采样)
//        return col.index_select(0, arange(new Scalar(0), new Scalar(Math.min(col.size(0), count)), seeds.options()));
//    }
//}
//
//public class BidirectionalNeighborSampler {
//    private Map<String, Tensor> adjRowPtr, adjColIndex;       // 出边 (Downstream)
//    private Map<String, Tensor> revAdjRowPtr, revAdjColIndex; // 入边 (Upstream)
//
//    public BidirectionalNeighborSampler(Map<String, Tensor> forward, Map<String, Tensor> backward) {
//        this.adjRowPtr = forward.get("rowPtr");
//        this.adjColIndex = forward.get("colIndex");
//        this.revAdjRowPtr = backward.get("rowPtr");
//        this.revAdjColIndex = backward.get("colIndex");
//    }
//
//    public HeteroSamplerOutput sample(Map<String, Tensor> seedNodes, int numNeighbors) {
//        try (PointerScope scope = new PointerScope()) {
//            HeteroSamplerOutput output = new HeteroSamplerOutput();
//
//            for (String nodeType : seedNodes.keySet()) {
//                Tensor seeds = seedNodes.get(nodeType);
//
//                // 1. Downstream Sampling (沿出边方向)
//                Tensor downstreamNodes = sampleLayer(adjRowPtr, adjColIndex, seeds, numNeighbors);
//
//                // 2. Upstream Sampling (沿入边方向)
//                Tensor upstreamNodes = sampleLayer(revAdjRowPtr, revAdjColIndex, seeds, numNeighbors);
//
//                // 3. Merge & Unique
//                Tensor combined = cat(new TensorVector(seeds, downstreamNodes, upstreamNodes), 0);
//                output.nodeIds.put(nodeType, combined.unique());
//            }
//            return output;
//        }
//    }
//
//    private Tensor sampleLayer(Tensor ptr, Tensor col, Tensor seeds, int count) {
//        // 实现具体的邻居随机采样逻辑
//        return empty(new long[]{0}); // 占位符
//    }
//}