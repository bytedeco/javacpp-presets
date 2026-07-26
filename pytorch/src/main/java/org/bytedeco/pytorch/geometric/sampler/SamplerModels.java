package org.bytedeco.pytorch.geometric.sampler;
import org.bytedeco.pytorch.*;

public class SamplerModels {

    // NodeSamplerInput: 记录采样起始节点及其配置
    public static class NodeSamplerInput {
        public Tensor node_indices; // 目标采样点 [N]
        public String input_type;   // 对于异构图，指明节点类型
        // 可以扩展 time, weight 等字段
        public NodeSamplerInput(Tensor node_indices) { this.node_indices = node_indices; }
    }

    // SamplerOutput: 采样后的局部图结构
    public static class SamplerOutput {
        public Tensor node;         // 采样涉及的所有原始节点索引 (去重后的全局 ID)
        public Tensor row;          // 局部边索引的源节点 (Local ID)
        public Tensor col;          // 局部边索引的目标节点 (Local ID)
        public Tensor edge;         // 采样涉及的原始边索引 (Global ID)
        public Tensor batch;        // 节点所属的 seed node 编号 (可选)

        public SamplerOutput(Tensor node, Tensor row, Tensor col, Tensor edge) {
            this.node = node;
            this.row = row;
            this.col = col;
            this.edge = edge;
        }
    }
}
