package org.bytedeco.pytorch.geometric.metrics;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.metrics.*;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

public class LinkPredMRR extends LinkPredMetric {
    public LinkPredMRR(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            T_TensorTensor_T topk= yPred.topk(k, 1, true, true);
            Tensor rel = yTrue.gather(1, topk.get1());

            // 找到第一个非零相关项的索引
            // 构造 [1, 1/2, 1/3, ..., 1/k]
            Tensor reciprocalRanks = ones(new long[]{1, k}, yPred.options().dtype(new ScalarTypeOptional(kFloat())))
                    .div(arange(new Scalar(1), new Scalar(k + 1), yPred.options()).view(1, -1).to(kFloat()));

            // 逻辑：取相关项中最优的排名倒数
            Tensor maskedRanks = rel.to(kFloat()).mul(reciprocalRanks);
            Tensor mrr, maxIdx;
            // 获取每行的最大倒数排名 (即第一个出现的 1)
            return maskedRanks.max(1).get0().detach();
//        }
    }
}