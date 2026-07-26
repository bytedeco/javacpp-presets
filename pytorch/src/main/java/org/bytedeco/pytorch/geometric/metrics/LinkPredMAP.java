package org.bytedeco.pytorch.geometric.metrics;


import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

// === MAP @ K (Mean Average Precision) ===
public class LinkPredMAP extends LinkPredMetric {
    public LinkPredMAP(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            T_TensorTensor_T topk = yPred.topk(k, 1, true, true);
            Tensor relevantAtK = yTrue.gather(1, topk.get1());

            // 计算累积精度：cumsum(rel) / arange(1, k+1)
            Tensor ranks = arange(new Scalar(1), new Scalar(k + 1), yPred.options()).view(1, -1);
            Tensor precisionAtI = relevantAtK.cumsum(1).div(ranks);

            // 只保留相关位置的精度，然后求平均
            Tensor ap = (precisionAtI.mul(relevantAtK)).sum(1).div(
                    where(relevantAtK.sum(1).gt(new Scalar(0)), relevantAtK.sum(1), ones_like(relevantAtK.sum(1)))
            );
            return ap;
//        }
    }
}