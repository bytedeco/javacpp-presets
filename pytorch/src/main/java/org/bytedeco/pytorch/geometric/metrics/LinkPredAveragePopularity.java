package org.bytedeco.pytorch.geometric.metrics;

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.Tensor;

public class LinkPredAveragePopularity extends LinkPredMetric {
    private Tensor itemPopularity; // [num_items] 预先计算好的 Item 受欢迎程度

    public LinkPredAveragePopularity(int k, Tensor itemPopularity) {
        super(k);
        this.itemPopularity = itemPopularity;
    }

    @Override
    public Tensor compute(Tensor yPred, Tensor yTrue) {
//        try (PointerScope scope = new PointerScope()) {
            Tensor indices = yPred.topk(k, 1, true, true).get1();

            // 映射到流行度分值
            Tensor popScores = itemPopularity.index_select(0, indices.view(-1)).view(indices.sizes());

            // 计算每个用户的平均推荐流行度
            return popScores.mean(1).detach();
//        }
    }
}
