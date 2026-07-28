package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;

/**
 * Variance aggregation: {@code Var(X) = E[X²] − (E[X])²} (non-negative via relu).
 */
public class VarAggregation extends Aggregation {

    private final double eps;

    public VarAggregation() {
        this(0.0);
    }

    public VarAggregation(double eps) {
        super();
        this.eps = eps;
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        Tensor mean = AggrUtils.scatter(x, index, dimSize, "mean");
        Tensor mean2 = AggrUtils.scatter(x.mul(x), index, dimSize, "mean");
        Tensor var = mean2.sub(mean.pow(new Scalar(2)));
        // Numerical floor: floating-point can yield tiny negatives
        var = torch.relu(var);
        if (eps > 0) {
            var = var.add(new Scalar(eps));
        }
        return var;
    }
}
