package org.bytedeco.pytorch.geometric.sampler;
import org.bytedeco.pytorch.autograd.*;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.pooling.ClusterPooling;
import org.bytedeco.pytorch.geometric.nn.pooling.EdgePooling;
import org.bytedeco.pytorch.geometric.nn.pooling.EdgePoolingOutput;
import org.bytedeco.pytorch.geometric.nn.pooling.SAGPooling;

import java.util.Arrays;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.randn;
import static org.bytedeco.pytorch.global.torch.zeros;

public class NeighborSampler extends BaseSampler {
    private final int[] numNeighbors; // 每层采样的数量，例如 [10, 5] 表示采样两层

    // 存储异构图的 CSR 数据：edgeType -> Tensor
//    private final Map<String, Tensor> rowPtrs;
//    private final Map<String, Tensor> colIndices;

    // 适配你提到的调用方式：NeighborSampler(adj.rowPtr, adj.colIndex)
//    public NeighborSampler(Map<String, Tensor> rowPtrs, Map<String, Tensor> colIndices) {
//        this.rowPtrs = rowPtrs;
//        this.colIndices = colIndices;
//    }
    public NeighborSampler(Tensor edge_index, long numNodes, int[] numNeighbors) {
        super(edge_index, numNodes);
        this.numNeighbors = numNeighbors;
    }

    @Override
    public SamplerModels.SamplerOutput sampleFromNodes(SamplerModels.NodeSamplerInput input) {
        Tensor seedNodes = input.node_indices;

        // 1. 初始化采样结果容器
        // adjs 用于存储每层的局部边
        java.util.List<Tensor[]> adjs = new java.util.ArrayList<>();
        Tensor currentNodes = seedNodes;

        // 2. 逐层采样
        for (int k : numNeighbors) {
            // 在 edge_index 中查找以 currentNodes 为 target (col) 的边
            // 注意：PyG 的采样通常是反向进行的 (message passing: source -> target)

            // 简化逻辑：找到所有连接到 currentNodes 的源节点
            // 这里通常需要一个 CSR (Compressed Sparse Row) 格式来加速查找
            // 为了保持纯 Tensor 实现，我们使用 mask 过滤
            Tensor mask = isin(edge_index.select(0, 1), currentNodes);
            Tensor subsetEdgeIndex = edge_index.index_select(1, mask.nonzero().reshape(-1));

            // 随机采样 K 个邻居 (此处为简化版，实际需要按节点分组采样)
            long totalEdges = subsetEdgeIndex.size(1);
            if (totalEdges > k * currentNodes.size(0)) {
                Tensor randIdx = randperm(totalEdges, edge_index.options()).slice(0, new LongOptional(0) , new LongOptional(k * currentNodes.size(0)),1);
                subsetEdgeIndex = subsetEdgeIndex.index_select(1, randIdx);
            }

            // 更新 currentNodes 为采样到的新节点
            Tensor cats = cat(new TensorVector(currentNodes, subsetEdgeIndex.select(0, 0)), 0);
            Tensor newNodes = unique_consecutive(cats).get0();

            // 记录本层结果 (此处需将全局索引映射为局部索引)
            // ... 局部映射逻辑 ...

            currentNodes = newNodes;
        }

        // 3. 封装 SamplerOutput (示例返回)
        return new SamplerModels.SamplerOutput(currentNodes, null, null, null);
    }

    /**
     * 对应 EdgeSampler 的入口
     */
//    Override
    public HeteroSamplerOutput sampleFromEdges(Map<String, Tensor> edgeLabelIndex,
                                               NegativeSampling negSampling) {
        // 1. 获取正样本节点
        // 2. 如果 negSampling 不为空，生成负样本边
        // 3. 将所有涉及节点合并，调用 sampleFromNodes
        return new HeteroSamplerOutput();
    }
    @Override
    public SamplerModels.SamplerOutput sampleFromEdges(EdgeSamplerInput input) {
        throw new UnsupportedOperationException("Edge sampling not implemented yet.");
    }
}
