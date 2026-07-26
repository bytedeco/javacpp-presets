package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * Exact L2 KNN
 * 使用 torch.cdist 计算欧氏距离
 */
public class L2KNNIndex extends KNNIndex {
    public L2KNNIndex(long k) { super(k); }

    @Override
    public Tensor[] search(Tensor x, Tensor y, Tensor batchX, Tensor batchY) {
        Tensor target = (y == null) ? x : y;

        // 简单实现：全对全距离 (Batch 较大时建议使用 mask 或 loop 处理)
        // [N, M]
        Tensor dist = torch.cdist(x, target);

        // 如果提供了 Batch，需要屏蔽掉不同 Batch 的节点 (设为无穷大)
        if (batchX != null && batchY != null) {
            // mask[i, j] = (batchX[i] != batchY[j])
            Tensor mask = batchX.unsqueeze(1).ne(batchY.unsqueeze(0));
            dist.masked_fill_(mask, new Scalar(Float.POSITIVE_INFINITY));
        }

        // TopK (smallest distances)
        // topk returns (values, indices)
        T_TensorTensor_T ret = torch.topk(dist, k, 1, false, true); // largest=false (min), sorted=true
        return new Tensor[]{ret.get0(), ret.get1()};
    }
}

