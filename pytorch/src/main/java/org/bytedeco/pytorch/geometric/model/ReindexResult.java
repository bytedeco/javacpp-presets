package org.bytedeco.pytorch.geometric.model;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.MemoryFormatOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

import static org.bytedeco.pytorch.global.torch.*;

public  class ReindexResult {
    public Tensor localEdgeIndex;
    public Tensor nodeMapping; // 局部索引到原始 ID 的映射
    public ReindexResult(Tensor localEdgeIndex, Tensor nodeMapping) {
        this.localEdgeIndex = localEdgeIndex;
        this.nodeMapping = nodeMapping;
    }

//    public static ReindexResult generateMapping(Tensor allSampledNodes) {
//        ReindexResult res = new ReindexResult();
//        // 保证映射是唯一的且有序的
//        res.nodeMapping = allSampledNodes.unique().sort().get0().detach();
//        return res;
//    }

    /**
     * 符合要求的 generateMapping 实现
     * @param allSampledNodes 采样涉及的所有全局节点 ID
     * @param globalEdges 需要重映射的全局边 [2, E]
     */

    public static ReindexResult generateMapping(Tensor allSampledNodes, Tensor globalEdges) {
        // 1. 健壮性检查：防止传进来的 Tensor 本身就是 NULL
        if (allSampledNodes == null || allSampledNodes.isNull()) {
            throw new RuntimeException("generateMapping 失败: 输入的 allSampledNodes Tensor 为空。请检查采样器返回的 Key 是否正确。");
        }
        if (globalEdges == null || globalEdges.isNull()) {
            // 如果没有边，创建一个空的局部边 Tensor [2, 0]
            globalEdges = empty(new long[]{2, 0}, allSampledNodes.options(),new MemoryFormatOptional());
        }

//        try (PointerScope scope = new PointerScope()) {
            // 2. 建立映射表
            // 使用 unique().get0() 获取去重后的节点，再 sort().get0() 排序
//            Tensor nodeMapping = allSampledNodes.unique().get0().sort().get0().detach();

            Tensor nodeMapping = unique_consecutive(allSampledNodes).get0().sort().get0().detach();

            // 3. 翻译坐标
            Tensor localEdgeIndex = searchsorted(nodeMapping, globalEdges).detach();

            return new ReindexResult(localEdgeIndex, nodeMapping);
//        }
    }
    public static ReindexResult generateMapping2(Tensor allSampledNodes, Tensor globalEdges) {
//        try (PointerScope scope = new PointerScope()) {
            // 1. 建立映射表 (字典): 局部索引 -> 全局 ID
            // 使用 unique 和 sort 确保局部索引 0, 1, 2... 对应有序的全局 ID
            Tensor nodeMapping = unique_consecutive(allSampledNodes).get0().sort().get0().detach();

            // 2. 查字典 (Reindex): 将全局边转换为局部索引
            // searchsorted(sorted_sequence, values) 
            // 返回 globalEdges 中的每个值在 nodeMapping 中的插入位置索引
            Tensor localEdgeIndex = searchsorted(nodeMapping, globalEdges).detach();

            // 3. 调用你的构造函数返回
            return new ReindexResult(localEdgeIndex, nodeMapping);
//        }
    }

    public static Tensor remap(Tensor globalEdges, Tensor nodeMapping) {
//        try (PointerScope scope = new PointerScope()) {
            // 使用 searchsorted 将全局 ID 映射为局部索引 [0, 1, 2...]
            return searchsorted(nodeMapping, globalEdges).detach();
//        }
    }
    public static Tensor getLocalEdgeIndex(Tensor globalEdgeIndex, Tensor nodeMapping) {
//        try (PointerScope scope = new PointerScope()) {
            // 使用 searchsorted 将全局 ID 映射为 nodeMapping 中的索引位置
            // 注意：nodeMapping 必须是升序的 unique 节点集合
            return searchsorted(nodeMapping, globalEdgeIndex).detach();
//        }
    }
    public static ReindexResult reindex(Tensor edgeIndex) {
//    try (PointerScope scope = new PointerScope()) {
        // 1. 获取所有涉及到的唯一节点
        Tensor nodes =  unique_consecutive(edgeIndex.view(-1)).get0();

        // 2. 构建映射表 (原始 ID -> 局部 ID)
        // 在 LibTorch 中，我们利用 torch.searchsorted 实现高效映射
        Tensor sortedNodes = nodes.sort().get0();

        // 3. 将 edgeIndex 中的全局 ID 替换为在 sortedNodes 中的位置
        Tensor localRow = searchsorted(sortedNodes, edgeIndex.select(0, 0));
        Tensor localCol = searchsorted(sortedNodes, edgeIndex.select(0, 1));

        Tensor localEdgeIndex = stack(new TensorVector(localRow, localCol), 0);

        return new ReindexResult(localEdgeIndex.detach(), sortedNodes.detach());
//    }
    }
    
}


