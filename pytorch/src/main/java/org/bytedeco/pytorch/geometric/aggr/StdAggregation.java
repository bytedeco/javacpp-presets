package org.bytedeco.pytorch.geometric.aggr;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

/**
 * 4. org.bytedeco.pytorch.geometric.aggr.StdAggregation (Standard Deviation)
 * Std(X) = sqrt(Var(X) + eps)
 */
public class StdAggregation extends Aggregation {
    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        Tensor var = new VarAggregation().forward(x, index, dimSize);
        // 加 eps 防止 sqrt(0) 梯度爆炸
        return var.add(new Scalar(1e-6)).sqrt();
    }
}
