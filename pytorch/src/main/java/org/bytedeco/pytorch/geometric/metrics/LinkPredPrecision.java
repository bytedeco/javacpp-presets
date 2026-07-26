package org.bytedeco.pytorch.geometric.metrics;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.kFloat;

// === Precision @ K ===
public class LinkPredPrecision extends LinkPredMetric {
    public LinkPredPrecision(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            // 获取 Top-K 的索引
            T_TensorTensor_T topk= yPred.topk(k, 1, true, true);
            Tensor indices = topk.get1();

            // 提取对应位置的真实标签
            Tensor relevantAtK = yTrue.gather(1, indices);

            // 计算每个 Batch 的 Precision: (相关数 / k)
            return relevantAtK.sum(1).to(kFloat()).div(new Scalar(k));
//        }
    }
}


