package org.bytedeco.pytorch.geometric.metrics;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.metrics.*;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.kFloat;
import static org.bytedeco.pytorch.global.torch.tensor;

public class LinkPredHitRatio extends LinkPredMetric {
    public LinkPredHitRatio(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            T_TensorTensor_T topk= yPred.topk(k, 1, true, true);
            Tensor rel = yTrue.gather(1, topk.get1());

            // 只要 sum > 0 就算 Hit (1), 否则 (0)
            return rel.sum(1).gt(new Scalar(0)).to(kFloat()).detach();
//        }
    }
}
