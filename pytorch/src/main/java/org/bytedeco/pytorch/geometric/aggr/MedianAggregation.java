package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * Median org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 计算邻居特征的中位数
 */
public class MedianAggregation extends Aggregation {

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. 转为 Dense, 填充 NaN
        Tensor[] denseData = AggrUtils.to_dense_batch(x, index, dimSize, Float.NaN);
        Tensor denseX = denseData[0]; // [N, MaxDeg, F]

        // 2. NanMedian on dim 1
        // values, indices = nanmedian(...)
        Tensor out = torch.nanmedian(denseX, 1).get0();

        return torch.nan_to_num(out);
    }
}