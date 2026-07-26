package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * LogNormal（对数正态）分布实现
 * 对数均值m(μ)：log(X)的均值
 * 对数标准差s(σ)：log(X)的标准差（必须>0）
 * 修复点：
 * 1. 修复s=0的校验逻辑（包含≤0的所有情况）
 * 2. 优化资源管理（静态Scalar不随实例释放）
 * 3. 增强v≤0时log_prob的返回逻辑（严格返回-∞）
 * 4. 提升数值稳定性，避免极端值溢出
 */
public class LogNormal extends Distribution implements AutoCloseable {
    private final Tensor m;   // 对数均值μ
    private final Tensor s;   // 对数标准差σ（必须>0）
    private boolean isClosed = false; // 防止重复释放

    // 预定义静态标量（仅初始化一次，不随实例释放）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_2 = new Scalar(2.0f);
    private static final Scalar SCALAR_SQRT_2PI = new Scalar((float) Math.sqrt(2 * Math.PI));
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Float.NEGATIVE_INFINITY);
    private static final Scalar SCALAR_MAX_EXP = new Scalar(20.0f);
    private static final Scalar SCALAR_MIN_EXP = new Scalar(-80.0f); // 限制exp输入下限

    /**
     * 构造函数：校验参数合法性 + 深拷贝 + 数值保护
     * @param m 对数均值μ
     * @param s 对数标准差σ（必须>0）
     * @throws IllegalArgumentException 标准差≤0时抛出异常
     */
    public LogNormal(Tensor m, Tensor s) {
        // 校验输入张量非空
        if (m == null || s == null) {
            throw new IllegalArgumentException("对数正态分布m(μ)和s(σ)参数不能为空！");
        }

        // 统一转换为Float32+CPU，避免类型/设备不匹配
        Tensor mCpu = m.to(new Device(DeviceType.CPU),kFloat()).clone().detach();
        Tensor sCpu = s.to(new Device(DeviceType.CPU),kFloat()).clone().detach();

        // 核心修复：校验s ≤ 0（包含0和负数）
        Tensor sLe0 = torch.le(sCpu, torch.tensor(0.0f, sCpu.options()));
        try {
            if (torch.any(sLe0).item().toBool()) {
                throw new IllegalArgumentException("对数正态分布s(σ)必须大于0！");
            }
        } finally {
            sLe0.close(); // 确保临时张量释放
        }

        // 数值保护：避免s过小导致后续计算溢出
        Tensor safeS = torch.clamp(sCpu, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 深拷贝避免外部修改内部状态
        this.m = mCpu.clone().detach();
        this.s = safeS.clone().detach();

        // 释放临时张量
        mCpu.close();
        sCpu.close();
        safeS.close();
    }

    @Override
    public String name() {
        return "LogNormal";
    }

    /**
     * 采样：实现对数正态分布标准采样公式，支持批量采样
     * 公式：X = exp(μ + σ * N(0,1))
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();
        // 扩展采样形状
        long[] extendedShape = getExtendedShape(m, sampleShape);
        Tensor expandedM = m.expand(extendedShape).clone().detach();
        Tensor expandedS = s.expand(extendedShape).clone().detach();

        // 采样标准正态分布
        Tensor normalSample = randn(extendedShape, m.options());
        // 计算μ + σ*N(0,1)，限制数值范围避免exp溢出
        Tensor logX = torch.add(expandedM, torch.mul(expandedS, normalSample))
                .clamp(new ScalarOptional(SCALAR_MIN_EXP), new ScalarOptional(SCALAR_MAX_EXP));

        // 最终采样结果：exp(logX)，确保采样值>0
        Tensor logNormalSample = exp(logX).clamp(new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 释放临时张量
        expandedM.close();
        expandedS.close();
        normalSample.close();
        logX.close();

        return logNormalSample.clone().detach();
    }

    /**
     * 对数概率：实现对数正态分布完整对数概率公式，增强数值稳定性
     * 公式：log(f(x)) = -log(xσ√(2π)) - (logx - μ)²/(2σ²)
     * v≤0时严格返回-∞
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 统一类型和设备
        Tensor vCpu = v.to(m.options(),false, true, new MemoryFormatOptional()).clone().detach();
        // 标记v≤0的位置
        Tensor vLe0Mask = torch.le(vCpu, SCALAR_0);

        // 初始化对数概率张量为-∞
        Tensor logProb = torch.full_like(vCpu, SCALAR_NEG_INF, m.options(), new MemoryFormatOptional());
        // 筛选v>0的位置进行计算
        Tensor vGt0Mask = torch.logical_not(vLe0Mask);
        Tensor vGt0 = torch.masked_select(vCpu, vGt0Mask);

        if (vGt0.numel() > 0) {
            // 扩展m/s到v>0部分的形状
            long[] vGt0Shape = vGt0.sizes().vec().get();
            Tensor expandedM = m.expand(vGt0Shape).clone().detach();
            Tensor expandedS = s.expand(vGt0Shape).clone().detach();

            // 数值稳定性处理
            Tensor safeV = torch.clamp(vGt0, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));
            Tensor safeS = torch.clamp(expandedS, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

            // 计算对数概率各项
            Tensor logV = torch.log(safeV);
            Tensor logS = torch.log(safeS);
            Tensor logSqrt2Pi = torch.log(torch.tensor((float) Math.sqrt(2 * Math.PI)));

            // term1 = -[logx + logs + log√(2π)]
            Tensor term1 = torch.neg(torch.add(torch.add(logV, logS), logSqrt2Pi));

            // term2 = - (logx - μ)² / (2σ²)
            Tensor logVMinusM = torch.sub(logV, expandedM);
            Tensor logVMinusMSq = torch.pow(logVMinusM, SCALAR_2);
            Tensor twoSSq = torch.mul(torch.pow(safeS, SCALAR_2), SCALAR_2);
            Tensor term2 = torch.neg(torch.div(logVMinusMSq, twoSSq));

            // 合并两项
            Tensor validLogProb = torch.add(term1, term2);

            // 将有效结果回填到logProb中
            logProb = torch.masked_scatter(logProb, vGt0Mask, validLogProb);

            // 释放临时张量
            expandedM.close();
            expandedS.close();
            safeV.close();
            safeS.close();
            logV.close();
            logS.close();
            logSqrt2Pi.close();
            term1.close();
            logVMinusM.close();
            logVMinusMSq.close();
            twoSSq.close();
            term2.close();
            validLogProb.close();
        }

        // 释放临时张量
        vCpu.close();
        vLe0Mask.close();
        vGt0Mask.close();
        vGt0.close();

        return logProb.clone().detach();
    }

    /**
     * 熵：实现对数正态分布完整熵公式
     * 公式：H = μ + 0.5 + log(σ√(2π))
     */
    @Override
    public Tensor entropy() {
        checkClosed();
        // 数值稳定性处理
        Tensor safeS = torch.clamp(s, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 计算各项
        Tensor logS = torch.log(safeS);
        Tensor logSqrt2Pi = torch.log(torch.tensor((float) Math.sqrt(2 * Math.PI)));

        // 完整熵公式
        Tensor entropy = torch.add(
                torch.add(m, SCALAR_0_5),
                torch.add(logS, logSqrt2Pi)
        );

        // 释放临时张量
        safeS.close();
        logS.close();
        logSqrt2Pi.close();

        return entropy.clone().detach();
    }

    /**
     * 均值：对数正态分布均值公式 E[X] = exp(μ + σ²/2)
     */
    @Override
    public Tensor mean() {
        checkClosed();
        // 计算σ²/2
        Tensor sSq = torch.pow(s, SCALAR_2);
        Tensor sSqHalf = torch.mul(sSq, SCALAR_0_5);

        // 计算μ + σ²/2，限制数值范围避免exp溢出
        Tensor expArg = torch.add(m, sSqHalf)
                .clamp(new ScalarOptional(SCALAR_MIN_EXP), new ScalarOptional(SCALAR_MAX_EXP));

        // 最终均值，确保>0
        Tensor mean = exp(expArg).clamp(new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 释放临时张量
        sSq.close();
        sSqHalf.close();
        expArg.close();

        return mean.clone().detach();
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 检查实例是否已释放，避免重复使用
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("LogNormal实例已释放，无法继续使用！");
        }
    }

    /**
     * 扩展采样形状
     */

    /**
     * 资源释放：仅释放实例相关张量，静态Scalar不释放
     */
    @Override
    public void close() {
        if (!isClosed) {
            m.close();
            s.close();
            isClosed = true;
        }
    }

    // Getter方法（返回拷贝避免外部修改）
    public Tensor getM() {
        checkClosed();
        return m.clone().detach();
    }

    public Tensor getS() {
        checkClosed();
        return s.clone().detach();
    }
}
