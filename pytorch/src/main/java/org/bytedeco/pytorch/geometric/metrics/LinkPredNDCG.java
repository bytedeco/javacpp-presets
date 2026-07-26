package org.bytedeco.pytorch.geometric.metrics;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.metrics.*;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

public class LinkPredNDCG extends LinkPredMetric {
    public LinkPredNDCG(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            T_TensorTensor_T topk= yPred.topk(k, 1, true, true);
            Tensor rel = yTrue.gather(1, topk.get1()); // 获取 Top-K 处的真实标签

            // 1. 计算 DCG: rel / log2(rank + 1)
            Tensor ranks = arange(new Scalar(1), new Scalar(k + 1), yPred.options()).view(1, -1).to(kFloat());
            Tensor logRanks = log2(ranks.add(new Scalar(1.0)));
            Tensor dcg = rel.to(kFloat()).div(logRanks).sum(1);

            // 2. 计算 IDCG (理想情况下的 DCG)
            Tensor idealRel = yTrue.sort(1, true).get0().slice(1, new LongOptional(0), new LongOptional(k), 1);
            Tensor idcg = idealRel.to(kFloat()).div(logRanks).sum(1);

            // 3. NDCG = DCG / IDCG
            return where(idcg.gt(new Scalar(0)), dcg.div(idcg), zeros_like(dcg)).detach();
//        }
    }
}
