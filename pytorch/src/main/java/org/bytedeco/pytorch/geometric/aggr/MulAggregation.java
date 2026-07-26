package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * 2. org.bytedeco.pytorch.geometric.aggr.MulAggregation (Product)
 * 元素乘积聚合
 */
public class MulAggregation extends Aggregation {
    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        return AggrUtils.scatter(x, index, dimSize, "prod");
    }
}