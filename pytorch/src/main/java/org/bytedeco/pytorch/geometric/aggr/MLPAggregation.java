package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * MLP org.bytedeco.pytorch.geometric.aggr.Aggregation
 * x' = Reduce(MLP(x))
 */
public class MLPAggregation extends Aggregation {
    private SequentialImpl mlp;
    private String reduce; // "sum", "mean", "max"

    public MLPAggregation(long inChannels, long outChannels, String reduce) {
        this.reduce = reduce;
        this.mlp = new SequentialImpl();
        this.mlp.push_back(new LinearImpl(inChannels, outChannels));
        this.mlp.push_back(new ReLUImpl());
        register_module("mlp", mlp);
    }

    // 支持传入自定义 MLP
    public MLPAggregation(SequentialImpl mlp, String reduce) {
        this.reduce = reduce;
        this.mlp = mlp;
        register_module("mlp", mlp);
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. Transform
        Tensor xTrans = mlp.forward(x);

        // 2. Aggregate
        return AggrUtils.scatter(xTrans, index, dimSize, reduce);
    }
}