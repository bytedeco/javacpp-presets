package org.bytedeco.pytorch.geometric.metrics;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public abstract class LinkPredMetric {
    protected int k;

    public LinkPredMetric(int k) {
        this.k = k;
    }

    /**
     * @param yPred: [batch_size, num_items] 各个项的预测概率/得分
     * @param yTrue: [batch_size, num_items] 真实标签 (1 为相关, 0 为不相关)
     */
    public abstract Tensor compute(Tensor yPred, Tensor yTrue);
}