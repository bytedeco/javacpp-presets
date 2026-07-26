package org.bytedeco.pytorch.geometric.sampler;
import org.bytedeco.pytorch.*;

public abstract class BaseSampler {
    // 原始图数据
    protected Tensor edge_index;
    protected long numNodes;

    public BaseSampler(Tensor edge_index, long numNodes) {
        this.edge_index = edge_index;
        this.numNodes = numNodes;
    }

    // 从节点出发采样
    public abstract SamplerModels.SamplerOutput sampleFromNodes(SamplerModels.NodeSamplerInput input);

    // 从边出发采样 (通常用于链接预测)
    public abstract SamplerModels.SamplerOutput sampleFromEdges(EdgeSamplerInput input);
}