package org.bytedeco.pytorch.geometric.sampler;

import org.bytedeco.pytorch.*;
import java.util.Map;
import java.util.HashMap;

public class HeteroSamplerOutput {
    // 节点类型 -> 节点ID Tensor
    public Map<String, Tensor> nodeIds = new HashMap<>();

    // 边类型 -> 边ID Tensor
    public Map<String, Tensor> edgeIds = new HashMap<>();

    // 采样后的局部邻接关系 (row, col)
    public Map<String, Tensor> row = new HashMap<>();
    public Map<String, Tensor> col = new HashMap<>();

    // 用于表示 batch 信息的 map
    public Map<String, Tensor> batch = new HashMap<>();

    @Override
    public String toString() {
        return "HeteroSamplerOutput(nodes=" + nodeIds.keySet() + ", edges=" + edgeIds.keySet() + ")";
    }
}
