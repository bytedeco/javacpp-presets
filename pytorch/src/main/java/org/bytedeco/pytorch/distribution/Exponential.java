package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Exponential extends Distribution implements AutoCloseable {
    private final Tensor rate;  // 率参数λ（必须>0）

    // 预定义标量（复用避免重复创建）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值

    // 构造函数：校验参数合法性 + 深拷贝
    public Exponential(Tensor rate) {
        // 校验率参数λ>0（指数分布核心约束）
        Tensor rateLe0 = torch.le(rate, torch.tensor(0.0f, rate.options()));
        if (torch.any(rateLe0).item().toBool()) {
            rateLe0.close();
            throw new IllegalArgumentException("指数分布rate(λ)必须严格大于0！");
        }
        rateLe0.close();

        // 深拷贝避免外部修改内部状态
        this.rate = rate.clone();
    }

    @Override
    public String name() {
        return "Exponential";
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：复用父类方法扩展采样形状
        long[] extendedShape = getExtendedShape(rate, sampleShape);
        Tensor expandedRate = rate.expand(extendedShape); // 扩展rate到批量形状

        // 步骤2：生成Uniform(ε,1-ε)随机数（避免log(0)/log(1)）
        Tensor u = torch.rand(extendedShape, rate.options())
                .clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));

        // 步骤3：指数分布采样公式：x = -log(1-u)/λ（Scalar转Tensor运算）
        Tensor oneTensor = torch.tensor(1.0f, u.options());
        Tensor oneMinusU = torch.sub(oneTensor, u); // 1-u
        Tensor logOneMinusU = torch.log(oneMinusU); // log(1-u)
        Tensor sample = torch.div(torch.neg(logOneMinusU), expandedRate); // -log(1-u)/λ

        // 释放所有临时张量
        expandedRate.close();
        u.close();
        oneTensor.close();
        oneMinusU.close();
        logOneMinusU.close();

        return sample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：校验输入v≥0
        Tensor vLt0 = torch.lt(v, torch.tensor(0.0f, v.options()));
        if (torch.any(vLt0).item().toBool()) {
            vLt0.close();
            throw new IllegalArgumentException("log_prob输入v必须大于等于0！");
        }
        vLt0.close();

        // 步骤2：数值稳定性处理（避免v过大导致数值溢出）
        Tensor safeV = v.clamp(new ScalarOptional(new Scalar(0.0f)), new ScalarOptional(v.max().item()));
        Tensor expandedRate = rate.expand(safeV.sizes()); // 扩展rate到v的形状

        // 步骤3：计算对数概率公式：log(λ) - λx
        Tensor logRate = torch.log(expandedRate); // log(λ)
        Tensor rateMulV = torch.mul(expandedRate, safeV); // λx
        Tensor logProb = torch.sub(logRate, rateMulV);

        // 释放临时张量
        safeV.close();
        expandedRate.close();
        logRate.close();
        rateMulV.close();

        return logProb;
    }

    @Override
    public Tensor entropy() {
        // 熵公式：H = 1 - log(λ)（Scalar转Tensor运算）
        Tensor oneTensor = torch.tensor(1.0f, rate.options());
        Tensor logRate = torch.log(rate); // log(λ)
        Tensor entropy = torch.sub(oneTensor, logRate); // 1 - log(λ)

        // 释放临时张量
        oneTensor.close();
        logRate.close();

        return entropy;
    }

    @Override
    public Tensor mean() {
        // 均值公式：1/λ（返回拷贝避免外部修改）
        Tensor mean = rate.reciprocal().clone();
        return mean;
    }

    // 资源释放：实现AutoCloseable
    @Override
    public void close() {
        rate.close();
        // 释放预定义Scalar

    }
}
