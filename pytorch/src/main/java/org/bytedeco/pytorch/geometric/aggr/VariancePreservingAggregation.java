package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

public class VariancePreservingAggregation extends Aggregation {

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. Sum org.bytedeco.pytorch.geometric.aggr.Aggregation
        Tensor agg = AggrUtils.scatter(x, index, dimSize, "sum");

        // 2. Compute Degree
        Tensor deg = AggrUtils.compute_degree(index, dimSize);
        deg = deg.clamp_min(new Scalar(1.0));

        // 3. Scale by 1 / sqrt(deg)
        // [Batch] -> [Batch, 1]
        Tensor scale = deg.sqrt().reciprocal().unsqueeze(1);

        return agg.mul(scale);
    }
}