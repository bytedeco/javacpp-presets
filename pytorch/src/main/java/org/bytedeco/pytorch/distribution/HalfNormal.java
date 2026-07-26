package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 半正态分布（HalfNormal）实现类
 * 修复点：
 * 1. 检测scale≤0（包含0）
 * 2. 优化熵和log_prob的数值稳定性
 * 3. 修复资源释放逻辑（静态常量不释放）
 * 4. 统一张量类型/设备，避免精度偏差
 */
public class HalfNormal extends Distribution implements AutoCloseable {
    private final Tensor scale;  // 尺度参数σ（必须>0）
    private final TensorOptions baseOptions; // 统一的设备/类型配置
    private boolean isClosed = false; // 防止重复释放

    // 数值稳定性常量（静态，仅定义不释放）
    private static final float EPS = 1e-8f;
    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI); // ≈0.79788456
    private static final double SQRT_PI_E_OVER_2 = Math.sqrt(Math.PI * Math.E / 2.0); // ≈2.22144147
    private static final double LOG_SQRT_2_OVER_PI = Math.log(SQRT_2_OVER_PI); // ≈-0.22579135

    // 构造函数：严格校验scale>0 + 统一类型/设备 + 深拷贝
    public HalfNormal(Tensor scale) {
        // 1. 统一转换为Float32+CPU，避免类型/设备不匹配导致的精度问题
        Tensor scaleCpu = scale.to(kFloat())
                .to(torch.device(new Device(DeviceType.CPU)), false, true, new MemoryFormatOptional())
                .clone().detach();
        this.baseOptions = scaleCpu.options();

        // 2. 严格校验scale>0（包含0的情况）
        Scalar scalar0 = new Scalar(0.0f);
        Tensor scaleLe0 = torch.le(scaleCpu, torch.tensor(0.0f, baseOptions));
        try {
            if (torch.any(scaleLe0).item().toBool()) {
                throw new IllegalArgumentException("半正态分布scale(σ)必须大于0！");
            }
        } finally {
            scaleLe0.close();
            scalar0.close();
        }

        // 3. 数值保护：避免scale过小导致计算溢出
        Scalar scalarEPS = new Scalar(EPS);
        Tensor safeScale = torch.clamp(scaleCpu, new ScalarOptional(scalarEPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 4. 深拷贝保存最终参数
        this.scale = safeScale.clone().detach();

        // 释放临时张量
        scaleCpu.close();
        scalarEPS.close();
        safeScale.close();
    }

    @Override
    public String name() {
        return "HalfNormal";
    }

    /**
     * 采样：半正态分布 = σ * |N(0,1)|
     * 保证采样值全≥0，且分布特性符合理论值
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();

        // 步骤1：扩展采样形状
        long[] extendedShape = getExtendedShape(scale, sampleShape);
        Tensor expandedScale = scale.expand(extendedShape).clone().detach();

        // 步骤2：采样标准正态分布并取绝对值
        Tensor normalSample = torch.randn(extendedShape, baseOptions);
        Tensor absNormalSample = torch.abs(normalSample);
        Tensor halfNormalSample = torch.mul(absNormalSample, expandedScale);

        // 释放临时张量
        expandedScale.close();
        normalSample.close();
        absNormalSample.close();

        return halfNormalSample.clone().detach();
    }

    /**
     * 对数概率：严格对齐数学公式 + 数值稳定性
     * log_prob(x) = log(√(2/π)) - log(σ) - x²/(2σ²) （x≥0）
     * log_prob(x) = -∞ （x<0）
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 1. 统一转换为Float32+CPU，确保类型/设备对齐
        Tensor vCpu = v.to(baseOptions, false, true, new MemoryFormatOptional()).clone().detach();

        // 2. 扩展scale到v的形状
        Tensor expandedScale = scale.expand_as(vCpu).clone().detach();
        Tensor safeScale = torch.clamp(expandedScale, new ScalarOptional(new Scalar(EPS)), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 3. 计算核心log_prob（预计算常量避免重复计算）
        // term1 = log(√(2/π)) ≈-0.22579135
        Tensor term1 = torch.full_like(safeScale, new Scalar((float) LOG_SQRT_2_OVER_PI), baseOptions, new MemoryFormatOptional());
        // term2 = -log(σ)
        Tensor logScale = torch.log(safeScale);
        Tensor term2 = torch.neg(logScale);
        // term3 = -x²/(2σ²)
        Tensor vSquared = torch.pow(vCpu, new Scalar(2.0f));
        Tensor scaleSquared = torch.pow(safeScale, new Scalar(2.0f));
        Tensor denominator = torch.mul(scaleSquared, new Scalar(2.0f));
        // 数值保护：避免分母为0
        denominator = torch.clamp(denominator, new ScalarOptional(new Scalar(EPS)), new ScalarOptional(new Scalar(Float.MAX_VALUE)));
        Tensor term3 = torch.neg(torch.div(vSquared, denominator));

        // 基础对数概率
        Tensor lpBase = torch.add(torch.add(term1, term2), term3);

        // 4. 处理v<0的情况（返回-∞）
        Scalar scalar0 = new Scalar(0.0f);
        Tensor vGe0 = torch.ge(vCpu, torch.tensor(0.0f, baseOptions));
        Tensor negInfTensor = torch.full_like(lpBase, new Scalar(Float.NEGATIVE_INFINITY), baseOptions, new MemoryFormatOptional());
        Tensor logProb = torch.where(vGe0, lpBase, negInfTensor);

        // 释放所有临时张量
        vCpu.close();
        expandedScale.close();
        safeScale.close();
        term1.close();
        logScale.close();
        term2.close();
        vSquared.close();
        scaleSquared.close();
        denominator.close();
        term3.close();
        lpBase.close();
        scalar0.close();
        vGe0.close();
        negInfTensor.close();

        return logProb.clone().detach();
    }

    /**
     * 熵：H = log(σ * √(πe/2))
     * 优化数值稳定性，确保σ=2时输出≈1.4921
     */
    @Override
    public Tensor entropy() {
        checkClosed();

        // 预计算常量：√(πe/2) ≈2.22144147
        Tensor sqrtPiEOver2Tensor = torch.full_like(scale, new Scalar((float) SQRT_PI_E_OVER_2), baseOptions, new MemoryFormatOptional());
        Tensor scaleMulConst = torch.mul(scale, sqrtPiEOver2Tensor);

        // 数值保护：避免log(0)
        Tensor safeArg = torch.clamp(scaleMulConst, new ScalarOptional(new Scalar(EPS)), new ScalarOptional(new Scalar(Float.MAX_VALUE)));
        Tensor entropy = torch.log(safeArg);

        // 释放临时张量
        sqrtPiEOver2Tensor.close();
        scaleMulConst.close();
        safeArg.close();

        return entropy.clone().detach();
    }

    /**
     * 均值：μ = σ * √(2/π)
     * 确保σ=2时输出≈1.5958
     */
    @Override
    public Tensor mean() {
        checkClosed();

        // 预计算常量：√(2/π) ≈0.79788456
        Tensor sqrt2OverPiTensor = torch.full_like(scale, new Scalar((float) SQRT_2_OVER_PI), baseOptions, new MemoryFormatOptional());
        Tensor mean = torch.mul(scale, sqrt2OverPiTensor).clone().detach();

        // 释放临时张量
        sqrt2OverPiTensor.close();

        return mean;
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 检查实例是否已释放，避免重复使用
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("HalfNormal实例已释放，无法继续使用！");
        }
    }

    /**
     * 扩展采样形状（兼容批量参数）
     */

    /**
     * 安全释放资源（避免重复释放）
     */
    @Override
    public void close() {
        if (!isClosed) {
            scale.close();
            isClosed = true;
        }
    }

    // 获取scale参数（返回拷贝，避免外部修改）
    public Tensor getScale() {
        checkClosed();
        return scale.clone().detach();
    }
}
