package org.bytedeco.pytorch.geometric.aggr;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
//import org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils;

/**
 * Softmax org.bytedeco.pytorch.geometric.aggr.Aggregation
 * Learnable temperature t.
 * Out = Sum( Softmax(x/t) * x )
 */
public class SoftmaxAggregation extends Aggregation {
    private Tensor t; // Temperature inverse (beta)
    private boolean learnT;

    public SoftmaxAggregation(long channels, boolean learnT) {
        this.learnT = learnT;
        // 初始化 t=1.0
        Tensor initT = torch.ones(new long[]{1, channels}, new TensorOptions());

        if (learnT) {
            this.t = new Tensor(initT);
            register_parameter("t", t);
        } else {
            // 如果不学习，注册为 buffer (不会被 optimizer 更新，但会被 state_dict 保存)
            this.t = new Tensor(initT); // JavaCPP 简化处理，逻辑上应当是 buffer
            register_buffer("t", initT);
        }
    }

    @Override
    public Tensor forward(Tensor x, Tensor index, long dimSize) {
        // 1. Apply Temperature: x * t (or x / (1/t))
        // 注意 t 的形状是 [1, C]，广播到 [N, C]
        Tensor score = x.mul(t);

        // 2. Calculate Spatial Softmax based on index
        Tensor alpha = AggrUtils.scatter_softmax(score, index, dimSize);

        // 3. Weighted Sum
        Tensor weighted = x.mul(alpha);
        return AggrUtils.scatter(weighted, index, dimSize, "sum");
    }
}