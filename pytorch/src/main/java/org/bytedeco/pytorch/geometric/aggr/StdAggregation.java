package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

/**
 * Standard-deviation aggregation: {@code Std(X) = sqrt(Var(X) + ε)}.
 */
public class StdAggregation extends Aggregation {

    private final VarAggregation varAggr;
    private final double eps;

    public StdAggregation() {
        this(1e-6);
    }

    public StdAggregation(double eps) {
        super();
        this.eps = eps;
        this.varAggr = new VarAggregation(0.0);
        // Not registered as submodule — stateless utility; keeps param count clean
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        Tensor var = varAggr.forward(x, index, dimSize);
        return var.add(new Scalar(eps)).sqrt();
    }
}
