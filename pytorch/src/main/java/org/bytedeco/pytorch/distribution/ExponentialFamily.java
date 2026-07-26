package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.Tensor;

public abstract class ExponentialFamily extends Distribution {
    // 指数族分布的对数配分函数 (Log Partition Function)
    public abstract Tensor log_normalizer(Tensor... params);

    // 均值参数到自然参数的映射 (可选实现)
    public abstract Tensor mean_carrier_measure();
}
