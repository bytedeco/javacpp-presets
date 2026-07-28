package org.bytedeco.pytorch.geometric.nn.pooling;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Parameter;

import static org.bytedeco.pytorch.global.torch.mm;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * MemPooling: 基于记忆矩阵的池化
 * 核心：利用学习到的聚类分配矩阵 S 将 N 个节点映射到 M 个簇节点
 */
public class MemPooling extends Module {
    private long numClusters;
    private LinearImpl assignLayer;

    public MemPooling(long inDim, long numClusters) {
        this.numClusters = numClusters;
        // 用于计算聚类分配矩阵 S
        this.assignLayer = register_module("assignLayer", new LinearImpl(inDim, numClusters));
    }

    public Tensor forward(Tensor x) {
        // 1. 计算分配矩阵 S [N, M] 并通过 Softmax 归一化
        Tensor s = softmax(assignLayer.forward(x), 1);

        // 2. 矩阵乘法实现粗化: x_pooled = S^T * X
        // [M, N] * [N, D] -> [M, D]
        Tensor x_pooled = mm(s.transpose(0, 1), x);

        return x_pooled;
    }
}