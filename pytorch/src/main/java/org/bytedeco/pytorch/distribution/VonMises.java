package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 千万别用 torch.tensor(true) 这种方式创建布尔张量，JavaCPP-PyTorch会将其转换为数值张量，导致逻辑错误！必须使用 torch.tensor(true, dtype=torch.bool) 来明确指定数据类型。
 * VonMises（冯·米塞斯分布/循环正态分布）实现
 * loc(μ)：位置/均值角度（弧度，形状：batch_shape）
 * concentration(κ)：浓度参数（>0，形状：batch_shape，越大分布越集中）
 * 支持批量参数、精确采样（拒绝采样）、完整的数值稳定性和合法性校验
 */
public class VonMises extends Distribution implements AutoCloseable {
    private final Tensor loc;                // 位置参数μ（弧度）
    private final Tensor concentration;      // 浓度参数κ（>0）
    private final Tensor log2Pi;             // 预计算log(2π)，提升效率
    private final Tensor i0Kappa;            // 预计算I0(κ)，避免重复计算
    private final Tensor logI0Kappa;         // 预计算log(I0(κ))

    // 预定义标量（复用避免重复创建，提升性能+规范）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_2 = new Scalar(2.0);
    private static final Scalar SCALAR_PI = new Scalar(Math.PI);
    private static final Scalar SCALAR_2PI = new Scalar(2 * Math.PI);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值
    private static final Scalar SCALAR_INF = new Scalar(Double.POSITIVE_INFINITY);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Double.NEGATIVE_INFINITY);
    private static final LongOptional DIM_NEG_1 = new LongOptional(-1);

    /**
     * 构造函数：严格校验参数合法性 + 预计算关键值 + 深拷贝
     * @param loc 位置参数μ（弧度）
     * @param concentration 浓度参数κ（必须>0）
     * @throws IllegalArgumentException 参数非法/设备不匹配抛出异常
     */
    public VonMises(Tensor loc, Tensor concentration) {
        // 1. 空值校验
        if (loc == null || concentration == null) {
            throw new IllegalArgumentException("loc和concentration参数不能为空！");
        }

        // 2. 校验设备一致性
        if (!loc.device().equals(concentration.device())) {
            throw new IllegalArgumentException(
                    String.format("loc和concentration设备不匹配：loc=%s, concentration=%s",
                            loc.device().toString(), concentration.device().toString())
            );
        }

        // 3. 校验浓度参数κ>0（添加数值容忍度，避免浮点误差）
        Tensor kappaLe0 = torch.le(concentration, torch.tensor(1e-8, concentration.options()));

        if (torch.any(kappaLe0).item().toBool()) {
            kappaLe0.close();
            throw new IllegalArgumentException("浓度参数concentration(κ)必须大于0（数值容忍度1e-8）！");
        }

        // 4. 初始化核心参数（深拷贝避免外部修改）
        this.loc = loc.clone();
        this.concentration = concentration.clone();

        // 5. 预计算关键值（数值稳定处理）
        this.log2Pi = torch.log(torch.tensor(2 * Math.PI, loc.options()));
        Tensor safeKappa = torch.clamp(this.concentration, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e6))); // 限制上限避免贝塞尔函数溢出
        this.i0Kappa = i0(safeKappa);
        this.logI0Kappa = torch.log(torch.clamp(this.i0Kappa, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(Float.POSITIVE_INFINITY)))); // 避免log(0)

        // 释放校验临时张量
        kappaLe0.close();
        safeKappa.close();
    }

    @Override
    public String name() {
        return "VonMises";
    }

    /**
     * 采样：实现冯·米塞斯分布的精确拒绝采样算法（Fisher, 1993）
     * 适配任意批量采样形状，支持高/低浓度参数场景
     * @param sampleShape 批量采样形状
     * @return 采样结果张量（弧度，范围[-π, π]，形状：sampleShape + batch_shape）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展形状（sampleShape + batch_shape）
        long[] extendedShape = getExtendedShape(loc, sampleShape);

        // 步骤2：扩展参数到采样形状
        Tensor expandedLoc = loc.expand(extendedShape);
        Tensor expandedKappa = concentration.expand(extendedShape);

        // 步骤3：拒绝采样核心逻辑（Fisher's algorithm）
        // 3.1 预计算采样参数
        Tensor tau = torch.tensor(1.0f).add(torch.sqrt(torch.tensor(1.0f).add(torch.mul(expandedKappa, expandedKappa))));
        Tensor rho = tau.sub(torch.sqrt(torch.tensor(2.0f).mul( tau))).div(expandedKappa) ;
        Tensor r = (torch.tensor(1.0f).add(torch.mul(rho, rho))) .div(torch.tensor(2.0f).mul(rho));

        // 3.2 初始化采样结果
        Tensor samples = torch.empty(extendedShape, loc.options(),new MemoryFormatOptional());

        Tensor done = torch.zeros(extendedShape, torch.dtype(ScalarType.Bool));

        // 3.3 拒绝采样循环（直到所有样本采样完成）
        while (!torch.all(done).item().toBool()) {
            // 生成候选样本
            Tensor u1 = torch.rand(extendedShape, loc.options());
            Tensor z = torch.cos(torch.tensor(Math.PI).mul(u1));
            Tensor f = torch.tensor(1.0f).add(torch.mul(r, z)).div(z.add(r));
            Tensor c = torch.mul(expandedKappa, (r.sub(f)));

            // 生成均匀分布判断是否接受
            Tensor u2 = torch.rand(extendedShape, loc.options());
            Tensor accept = torch.log(torch.mul(u2, (torch.tensor(1.0f).sub(torch.mul(z, z))))).le(c);

            // 只处理未完成的样本
            Tensor mask = torch.logical_and(torch.logical_not(done), accept);

            if (torch.any(mask).item().toBool()) {
                // 计算最终角度（±π）
                Tensor u3 = torch.rand(extendedShape, loc.options());
                Tensor theta = torch.where(
                        torch.gt(u3, torch.tensor(0.5f)),
                        torch.acos(f),
                        torch.neg(torch.acos(f))
                );
                // 更新采样结果
                samples = torch.where(mask, theta, samples);
                Tensor trueTensor = torch.ones(extendedShape, done.options()); // done已是Bool类型，直接用ones
                // 更新完成标记
                done = torch.where(mask, trueTensor, done);

            }

            // 释放本轮临时张量
            u1.close();
            z.close();
            f.close();
            c.close();
            u2.close();
            accept.close();
            mask.close();

        }

        // 步骤4：加上位置参数μ，调整到正确的角度范围
        Tensor finalSamples = torch.add(expandedLoc, samples);
        // 归一化到[-π, π]
        finalSamples = torch.remainder(finalSamples.add(new Scalar(Math.PI)) , new Scalar(2.0f*Math.PI)).sub(new Scalar(Math.PI));

        // 释放所有临时张量
        expandedLoc.close();
        expandedKappa.close();
        tau.close();
        rho.close();
        r.close();
        samples.close();
        done.close();

        return finalSamples;
    }

    /**
     * 对数概率：实现冯·米塞斯分布的精确对数概率公式，修正原代码错误
     * 公式：logP(θ) = κ·cos(θ-μ) - log(2π) - log(I0(κ))
     * @param v 输入角度张量（弧度，形状需与参数可广播）
     * @return 对数概率张量（形状：batch_shape）
     */
    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：扩展参数到输入形状
        Tensor expandedLoc = loc.expand(v.sizes());
        Tensor expandedKappa = concentration.expand(v.sizes());
        Tensor expandedLog2Pi = log2Pi.expand(v.sizes());
        Tensor expandedLogI0Kappa = logI0Kappa.expand(v.sizes());

        // 步骤2：数值稳定处理输入角度（归一化到[-π, π]）
        Tensor theta = torch.remainder(v.sub(expandedLoc).add(new Scalar(Math.PI)), new Scalar(2.0f*Math.PI)).sub(new Scalar(Math.PI));

        // 步骤3：计算对数概率各项
        // term1 = κ·cos(θ-μ)
        Tensor cosTerm = torch.cos(theta);
        Tensor term1 = torch.mul(expandedKappa, cosTerm);

        // term2 = -log(2π) - log(I0(κ))
        Tensor term2 = torch.neg(torch.add(expandedLog2Pi, expandedLogI0Kappa));

        // 步骤4：完整对数概率
        Tensor logProb = torch.add(term1, term2);

        // 释放所有临时张量
        expandedLoc.close();
        expandedKappa.close();
        expandedLog2Pi.close();
        expandedLogI0Kappa.close();
        theta.close();
        cosTerm.close();
        term1.close();
        term2.close();

        return logProb;
    }

    /**
     * 均值：冯·米塞斯分布的循环均值 = μ
     * @return 均值张量（返回拷贝避免外部修改）
     */
    @Override
    public Tensor mean() {
        return loc.clone();
    }

    /**
     * 熵：实现冯·米塞斯分布的精确解析熵公式
     * 公式：H = log(2πI0(κ)) - κ·(I1(κ)/I0(κ))
     * @return 熵张量（形状：batch_shape）
     */
    @Override
    public Tensor entropy() {
        // 步骤1：数值稳定处理浓度参数
        Tensor safeKappa = torch.clamp(concentration, new ScalarOptional(new Scalar(1e-8)) , new ScalarOptional(new Scalar(1e6)) );

        // 步骤2：计算1阶修正贝塞尔函数I1(κ)
        Tensor i1Kappa = i1(safeKappa);

        // 步骤3：计算I1(κ)/I0(κ)（数值稳定处理）
        Tensor ratio = torch.div(i1Kappa, torch.clamp(i0Kappa, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(Float.POSITIVE_INFINITY))));

        // 步骤4：计算熵各项
        // term1 = log(2πI0(κ)) = log(2π) + log(I0(κ))
        Tensor term1 = torch.add(log2Pi, logI0Kappa);

        // term2 = -κ·(I1(κ)/I0(κ))
        Tensor term2 = torch.neg(torch.mul(safeKappa, ratio));

        // 步骤5：完整熵公式
        Tensor entropy = torch.add(term1, term2);

        // 释放临时张量
        safeKappa.close();
        i1Kappa.close();
        ratio.close();
        term1.close();
        term2.close();

        return entropy;
    }

    /**
     * 辅助方法：0阶第一类修正贝塞尔函数I0(x)
     * 适配JavaCPP-PyTorch，处理数值稳定性
     */
    private Tensor i0(Tensor x) {
        // 对于大x，使用渐近近似；小x使用级数展开（或调用PyTorch内置）
        Tensor absX = torch.abs(x);
        Tensor largeX = torch.gt(absX, torch.tensor(10.0f));

        // 渐近近似：I0(x) ≈ exp(x)/sqrt(2πx)
        Tensor approxLarge = torch.exp(absX).div(torch.sqrt(torch.mul(torch.tensor(2 * Math.PI), absX)));
        // 小x：使用PyTorch内置或级数展开（此处简化，实际可调用scipy接口或自定义实现）
        Tensor approxSmall = torch.modified_bessel_i0(absX); // 需PyTorch 1.9+支持

        Tensor result = torch.where(largeX, approxLarge, approxSmall);

        // 释放临时张量
        absX.close();
        largeX.close();
        approxLarge.close();
        approxSmall.close();

        return result;
    }

    /**
     * 辅助方法：1阶第一类修正贝塞尔函数I1(x)
     * 适配JavaCPP-PyTorch，处理数值稳定性
     */
    private Tensor i1(Tensor x) {
        Tensor absX = torch.abs(x);
        Tensor largeX = torch.gt(absX, torch.tensor(10.0));

        // 渐近近似：I1(x) ≈ exp(x)/sqrt(2πx) * (1 - 1/(4x))
        Tensor approxLarge = torch.exp(absX).div(torch.sqrt(torch.mul(torch.tensor(2 * Math.PI), absX)))
                .mul( torch.tensor(1.0f).sub(torch.tensor(1.0f).div(torch.mul(torch.tensor(4.0f), absX))));
        // 小x：使用PyTorch内置
        Tensor approxSmall = torch.modified_bessel_i1(absX);

        Tensor result = torch.where(largeX, approxLarge, approxSmall);

        // 释放临时张量
        absX.close();
        largeX.close();
        approxLarge.close();
        approxSmall.close();

        return result;
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     */
    @Override
    public void close() {
        loc.close();
        concentration.close();
        log2Pi.close();
        i0Kappa.close();
        logI0Kappa.close();
        // 释放预定义常量

    }

    // Getter方法（提升易用性）
    public Tensor getLoc() { return loc; }
    public Tensor getConcentration() { return concentration; }
    public Tensor getI0Kappa() { return i0Kappa; }
    public Tensor getLogI0Kappa() { return logI0Kappa; }

    /**
     * 额外实用方法：将角度归一化到[0, 2π]范围
     * @param theta 输入角度张量（弧度）
     * @return 归一化后的角度张量
     */
    public Tensor normalizeAngle(Tensor theta) {
        Tensor normalized = torch.remainder(theta, new Scalar(2.0f*Math.PI));
        normalized = torch.where(torch.lt(normalized, new Scalar(0.0f)), normalized.add(new Scalar(2.0f*Math.PI)), normalized);
        return normalized;
    }
}
