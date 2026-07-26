package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * Quantile org.bytedeco.pytorch.geometric.aggr.Aggregation
 * 计算邻居特征的分位数 (例如 q=0.5 即中位数)
 */
public class QuantileAggregation extends Aggregation {
    private double q;

    public QuantileAggregation(double q) {
        this.q = q;
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. 转为 Dense: [N, MaxDeg, F], 填充 NaN 以便忽略
        // 注意：我们填充 Float.NaN
        Tensor[] denseData = AggrUtils.to_dense_batch(x, index, dimSize, Float.NaN);
        Tensor denseX = denseData[0];

        // 2. 计算 Quantile (dim=1 是邻居维度)
        // nanquantile 会自动忽略 NaN 值
        // 输出: [N, F] linear, lower, higher, midpoint or nearest,
        Tensor out = torch.nanquantile(denseX, q, new LongOptional(1), false, "nearest");

        // 处理全 NaN 的行 (即没有邻居的节点)，nanquantile 可能返回 NaN
        // 通常将其置为 0
        return torch.nan_to_num(out);
    }
}