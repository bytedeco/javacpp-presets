package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

// 1. org.bytedeco.pytorch.geometric.aggr.SumAggregation
public class SumAggregation extends Aggregation {
    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        return AggrUtils.scatter(x, index, dimSize, "sum");
    }
}