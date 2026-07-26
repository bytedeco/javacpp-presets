package org.bytedeco.pytorch.geometric.nn.pooling;

import org.bytedeco.pytorch.Tensor;

public class VoxelOutput {
    public Tensor pos;      // 下采样后的坐标 [M, 3]
    public Tensor batch;    // 新的 batch 索引 [M]
    public Tensor cluster;  // 原始点到新点的映射关系 [N]

    public VoxelOutput(Tensor pos, Tensor batch, Tensor cluster) {
        this.pos = pos;
        this.batch = batch;
        this.cluster = cluster;
    }
}