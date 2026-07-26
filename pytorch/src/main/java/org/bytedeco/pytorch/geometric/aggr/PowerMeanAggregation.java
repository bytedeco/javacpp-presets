package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * Power Mean org.bytedeco.pytorch.geometric.aggr.Aggregation
 * Generalized mean: p=1 -> Mean, p=inf -> Max, p=-inf -> Min
 */
public class PowerMeanAggregation extends Aggregation {
    private Tensor p;
    private boolean learnP;

    public PowerMeanAggregation(long channels, boolean learnP) {
        this.learnP = learnP;
        // 初始化 p=1.0 (等价于 Mean)
        Tensor initP = torch.ones(new long[]{1, channels}, new TensorOptions());

        if (learnP) {
            this.p = new Tensor(initP);
            register_parameter("p", p);
        } else {
            this.p = new Tensor(initP);
            register_buffer("p", initP);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // y = (Mean( |x|^p )) ^ (1/p)

        // 1. Clamp x to be non-negative for power operation (or use abs)
        Tensor xAbs = x.abs().clamp_min(new Scalar(1e-7)); // 防止 0^p 梯度问题

        // 2. Calculate x^p
        Tensor xPow = xAbs.pow(p);

        // 3. Mean org.bytedeco.pytorch.geometric.aggr.Aggregation of x^p
        Tensor agg = AggrUtils.scatter(xPow, index, dimSize, "mean");

        // 4. Root (1/p)
        // agg = agg.clamp_min(1e-7); // 再次防止负数或0
        return agg.pow(p.reciprocal());
    }
}
