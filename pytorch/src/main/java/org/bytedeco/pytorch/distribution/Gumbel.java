package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public class Gumbel extends Distribution implements AutoCloseable {
    private final Tensor loc;  // μ（位置参数）
    private final Tensor scale; // β（尺度参数，必须>0）
    // 欧拉-马歇罗尼常数（γ≈0.57721），复用避免重复创建
    private static final Scalar EULER_GAMMA = new Scalar(0.57721);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar EPS = new Scalar(1e-8); // 数值稳定性极小值

    // 构造函数：校验核心参数合法性
    public Gumbel(Tensor loc, Tensor scale) {
        // 校验scale>0（耿贝尔分布的尺度参数必须>0）
        if (torch.any(scale.le(new Scalar(0.0))).item().toBool()) {
            throw new IllegalArgumentException("Gumbel分布的scale(β)必须大于0！");
        }
        // 深拷贝避免外部修改内部状态
        this.loc = loc.clone();
        this.scale = scale.clone();
    }

    @Override // 补充@Override注解
    public String name() {
        return "Gumbel";
    }

    @Override // 修复：支持批量采样形状 + 修正采样公式 + 数值稳定性
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展采样形状（与其他分布逻辑对齐）
        long[] extendedShape = getExtendedShape(loc, sampleShape);
        // 步骤2：生成Uniform(0,1)的随机数（添加极小值避免u=0/1）
        Tensor u = torch.rand(extendedShape, loc.options())
                .clamp(new ScalarOptional(new Scalar(1e-8)),new ScalarOptional(new Scalar(1.0 - 1e-8))); // 限制u∈(1e-8, 1-1e-8)
        // 步骤3：修正采样公式：μ - β * log(-log(u))
        Tensor logU = torch.log(u);
        Tensor negLogU = logU.neg(); // -log(u)
        Tensor logNegLogU = torch.log(negLogU); // log(-log(u))
        Tensor sample = loc.expand(extendedShape)
                .sub(scale.expand(extendedShape).mul(logNegLogU));

        // 释放所有临时张量
        logU.close();
        negLogU.close();
        logNegLogU.close();
        u.close();

        return sample;
    }

    @Override // 修复：数值稳定性 + 清晰的公式表达
    public Tensor log_prob(Tensor v) {
        // 步骤1：计算z = (v - μ)/β
        Tensor z = v.sub(loc).div(scale);
        // 步骤2：数值稳定性处理：z过大时，exp(-z)≈0，避免溢出
        Tensor zClamped = z.clamp(new ScalarOptional(new Scalar(-100.0f)) , new ScalarOptional(new Scalar(100.0f))); // 限制z范围
        Tensor expNegZ = torch.exp(zClamped.neg());

        // 步骤3：完整对数概率公式：-log(β) - z - exp(-z)
        Tensor logScale = torch.log(scale);
        Tensor logProb = logScale.neg()  // -log(β)
                .sub(z)                  // -z
                .sub(expNegZ);           // -exp(-z)

        // 释放临时张量
        z.close();
        zClamped.close();
        expNegZ.close();
        logScale.close();

        return logProb;
    }

    @Override // 补充@Override注解 + 复用标量
    public Tensor entropy() {
        // 熵公式：log(β) + γ + 1
        return torch.log(scale)
                .add(new Scalar(0.57721))
                .add(new Scalar(1.0f));
    }

    @Override // 补充@Override注解 + 复用标量
    public Tensor mean() {
        // 均值公式：μ + β * γ
        return loc.add(scale.mul(new Scalar(0.57721)));
    }

    // 资源释放：实现AutoCloseable
    @Override
    public void close() {
        loc.close();
        scale.close();

    }
}
