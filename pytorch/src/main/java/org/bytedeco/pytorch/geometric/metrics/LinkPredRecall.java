package org.bytedeco.pytorch.geometric.metrics;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

// === Recall @ K ===
public class LinkPredRecall extends LinkPredMetric {
    public LinkPredRecall(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
             T_TensorTensor_T topk= yPred.topk(k, 1, true, true);
            Tensor relevantAtK = yTrue.gather(1, topk.get1());

            Tensor totalRelevant = yTrue.sum(1).to(kFloat());
            // 避免除以 0
            totalRelevant = where(totalRelevant.gt(new Scalar(0)), totalRelevant, ones_like(totalRelevant));

            return relevantAtK.sum(1).to(kFloat()).div(totalRelevant);
//        }
    }
}