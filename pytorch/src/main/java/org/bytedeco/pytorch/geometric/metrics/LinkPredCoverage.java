package org.bytedeco.pytorch.geometric.metrics;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.metrics.*;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

public class LinkPredCoverage extends LinkPredMetric {
    public LinkPredCoverage(int k) { super(k); }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            T_TensorTensor_T topk= yPred.topk(k, 1, true, true);
            Tensor indices = topk.get1(); // [Batch, K]

            // 获取所有推荐过的唯一索引
            Tensor uniqueRecommended = unique_consecutive(indices.view(-1)).get0();
            long numUnique = uniqueRecommended.size(0);
            long totalItems = yPred.size(1);

            // Coverage 是一个全局指标，这里返回一个标量 Tensor
            return tensor((double) numUnique / totalItems).to(kFloat()).detach();
//        }
    }
}
