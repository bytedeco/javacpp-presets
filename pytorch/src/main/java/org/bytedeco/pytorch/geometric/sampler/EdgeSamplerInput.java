package org.bytedeco.pytorch.geometric.sampler;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class EdgeSamplerInput {
    private Tensor row; // 源节点
    private Tensor col; // 目标节点
    private Tensor edgeIndex; // Shape: [2, num_edges]
    private Tensor edgeLabels; // Shape: [num_edges] (Optional)

    public EdgeSamplerInput(Tensor edgeIndex, Tensor edgeLabels) {
        this.edgeIndex = edgeIndex;
        this.edgeLabels = edgeLabels;
        // 假设 edgeIndex 形状为 [2, E]
        this.row = edgeIndex.select(0, 0);
        this.col = edgeIndex.select(0, 1);
    }

    public long size() {
        return edgeIndex.size(1);
    }

    /**
     * 获取指定索引的边数据
     * @param indices 采样批次的索引
     * @return 包含起始点和终点的 Tensor
     */
    public Tensor getEdges(Tensor indices) {
        // 使用 index_select 提取特定边
        return edgeIndex.index_select(1, indices);
    }

    public Tensor getBatch(Tensor indices) {
        Tensor batchRow = row.index_select(0, indices);
        Tensor batchCol = col.index_select(0, indices);
        // 返回 [2, batch_size] 的结构
        return stack(new TensorVector(batchRow, batchCol), 0);
    }
    public Tensor getLabels(Tensor indices) {
        return (edgeLabels != null) ? edgeLabels.index_select(0, indices) : null;
    }
}