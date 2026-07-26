package org.bytedeco.pytorch.geometric.model;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.sampler.HeteroSamplerOutput;
import org.bytedeco.pytorch.geometric.sampler.NumNeighbors;

import java.util.HashMap;
import java.util.Map;
import static org.bytedeco.pytorch.global.torch.*;

public class NeighborSampler2 {
    // 存储异构图的 CSR 数据：edgeType -> Tensor
    private final Map<String, Tensor> rowPtrs;
    private final Map<String, Tensor> colIndices;

    // 适配你提到的调用方式：NeighborSampler(adj.rowPtr, adj.colIndex)
    public NeighborSampler2(Map<String, Tensor> rowPtrs, Map<String, Tensor> colIndices) {
        this.rowPtrs = rowPtrs;
        this.colIndices = colIndices;
    }

    // 在 NeighborSampler 或工具类中添加
    public Tensor convertToCOO(Map<String, Tensor> row, Map<String, Tensor> col) {
//        try (PointerScope scope = new PointerScope()) {
            // 假设我们采样的是同质图或特定边类型 "edge"
            Tensor r = row.get("edge");
            Tensor c = col.get("edge");

            if (r == null || c == null) {
                // 如果没采样到边，返回一个空的 [2, 0] Tensor
                return empty(new long[]{2, 0}, new TensorOptions().dtype(new ScalarTypeOptional(kLong())),new MemoryFormatOptional());
            }

            // 将 row 和 col 堆叠成 [2, E] 的 edge_index
            return stack(new TensorVector(r, c), 0).detach();
//        }
    }
    /**
     * 执行采样逻辑
     * @param seedNodes 初始节点集
     * @param numNeighbors 每层采样的数量对象
     */
    public HeteroSamplerOutput sampleFromNodes(Map<String, Tensor> seedNodes, NumNeighbors numNeighbors) {
//        try (PointerScope scope = new PointerScope()) {
            HeteroSamplerOutput output = new HeteroSamplerOutput();

            // 记录已采样的节点，防止重复
            Map<String, Tensor> currentSeeds = seedNodes;

            for (int i = 0; i < numNeighbors.numHops(); i++) {
                int count = (int) numNeighbors.get(i);
                Map<String, Tensor> nextLayerSeeds = new HashMap<>();

                for (Map.Entry<String, Tensor> entry : currentSeeds.entrySet()) {
                    String nodeType = entry.getKey();
                    Tensor seeds = entry.getValue();

                    // 查找所有以该节点类型为源的边
                    Tensor sampledNeighbors = sampleLayer(nodeType, seeds, count);

                    if (sampledNeighbors.numel() > 0) {
                        nextLayerSeeds.put(nodeType, sampledNeighbors);
                        // 更新总输出
                        Tensor existing = output.nodeIds.getOrDefault(nodeType, seeds);
                        Tensor combined = cat(new TensorVector(existing, sampledNeighbors), 0);
                        output.nodeIds.put(nodeType, unique_consecutive(combined).get0().detach());
                    }
                }
                currentSeeds = nextLayerSeeds;
            }
            return output;
//        }
    }

    private Tensor sampleLayer(String nodeType, Tensor seeds, int count) {
        TensorVector results = new TensorVector();
        // 遍历所有边类型
        for (String edgeType : rowPtrs.keySet()) {
            if (edgeType.startsWith(nodeType)) {
                Tensor ptr = rowPtrs.get(edgeType);
                Tensor col = colIndices.get(edgeType);
                // 执行切片采样
                results.put(rawSlice(ptr, col, seeds, count));
            }
        }
        return results.empty() ? empty(new long[]{0}, seeds.options(),new MemoryFormatOptional()) : cat(results, 0);
    }

    private void rawSliceWithEdges(Tensor ptr, Tensor col, Tensor seeds, int count,
                                   TensorVector rowOut, TensorVector colOut) {
        for (int i = 0; i < seeds.size(0); i++) {
            long nodeIdx = seeds.index(new TensorIndexVector(new TensorIndex(tensor(i)))).item().toLong();
            long start = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx)))).item().toLong();
            long end = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx + 1)))).item().toLong();

            long take = Math.min(end - start, (long) count);
            if (take > 0) {
                Tensor targetNodes = col.slice(0, new LongOptional(start), new LongOptional(start + take), 1);
                colOut.put(targetNodes);
                // 构造对应的源节点 Tensor [take]
                rowOut.put(full(new long[]{take}, new Scalar(nodeIdx), new TensorOptions().dtype(new ScalarTypeOptional(kLong()))));
            }
        }
    }
    private Tensor rawSlice(Tensor ptr, Tensor col, Tensor seeds, int count) {
        TensorVector collected = new TensorVector();
        for (int i = 0; i < seeds.size(0); i++) {
            long nodeIdx = seeds.index(new TensorIndexVector(new TensorIndex(tensor(i)))).item().toLong();
            long start = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx)))).item().toLong();
            long end = ptr.index(new TensorIndexVector(new TensorIndex(tensor(nodeIdx + 1)))).item().toLong();

            long take = Math.min(end - start, (long) count);
            if (take > 0) {
                collected.put(col.slice(0, new LongOptional(start), new LongOptional(start + take), 1));
            }
        }
        return collected.empty() ? empty(new long[]{0}, seeds.options(),new MemoryFormatOptional()) : cat(collected, 0);
    }
}