package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * 3. org.bytedeco.pytorch.geometric.aggr.VarAggregation (Variance)
 * Var(X) = E[X^2] - (E[X])^2
 */
public class VarAggregation extends Aggregation {
    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // Mean(X)
        Tensor mean = AggrUtils.scatter(x, index, dimSize, "mean");
        // Mean(X^2)
        Tensor x2 = x.mul(x);
        Tensor mean2 = AggrUtils.scatter(x2, index, dimSize, "mean");

        // Var = Mean2 - Mean^2
        Tensor var = mean2.sub(mean.pow(new Scalar(2)));

        // 数值稳定性: ReLU 防止负数 (浮点误差)
        return torch.relu(var);
    }
}
