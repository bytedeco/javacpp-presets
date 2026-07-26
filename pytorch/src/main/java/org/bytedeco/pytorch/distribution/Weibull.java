package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Weibull（威布尔分布）实现
 * 严格遵循API规范：
 * 1. clamp方法仅使用指定的ScalarOptional/TensorOptional参数签名
 * 2. 标量无加减乘除，必须通过torch.tensor(1.0f)转为Tensor后运算
 * 3. 设备初始化：Device cpuDevice = torch.device(new Device(torch.kCPU()));
 * 4. 张量类型/设备：scale.to(new Device(torch.kCPU()),kFloat()).clone().detach();
 * 5. 随机张量options：torch.tensor(0.0f).options().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(torch.kCPU())))
 */
public class Weibull extends Distribution implements AutoCloseable {
    private final Tensor scale;                // 尺度参数λ（>0）
    private final Tensor concentration;        // 形状参数k（>0）
    private final Tensor invConcentration;     // 预计算1/k
    private final Tensor logScale;             // 预计算log(λ)
    private final Tensor logConcentration;     // 预计算log(k)
    private final long[] batchShape;           // 批量形状
    private boolean isClosed = false;          // 防止重复释放

    // 预定义标量Tensor（严格按API初始化）
    private static final Tensor TENSOR_0;
    private static final Tensor TENSOR_1;
    private static final Tensor TENSOR_2;
    private static final Tensor TENSOR_EPS;
    private static final Tensor TENSOR_EULER_GAMMA;
    private static final Tensor TENSOR_INF;
    private static final Tensor TENSOR_NEG_INF;
    private static final Tensor TENSOR_1E_8;
    private static final Tensor TENSOR_1_MINUS_1E_8;
    private static final Tensor TENSOR_MAX_K;
    private static final Tensor TENSOR_MIN_K;
    private static final Tensor TENSOR_MAX_LAMBDA;
    private static final Tensor TENSOR_MIN_LAMBDA;
    private static final Tensor TENSOR_1E6;
    private static final Tensor TENSOR_700;
    private static final Tensor TENSOR_NEG_700;
    private static final Tensor TENSOR_GAMMA_MAX_INPUT;
    private static final Tensor TENSOR_GAMMA_MAX_OUTPUT;

    // 静态初始化：CPU float32 常量（TensorOptions，不是 Device）
    static {
        TensorOptions cpuF32 = new TensorOptions(kFloat());

        TENSOR_0 = torch.tensor(0.0f, cpuF32);
        TENSOR_1 = torch.tensor(1.0f, cpuF32);
        TENSOR_2 = torch.tensor(2.0f, cpuF32);
        TENSOR_EPS = torch.tensor(1e-8f, cpuF32);
        TENSOR_EULER_GAMMA = torch.tensor(0.57721566490153286f, cpuF32);
        TENSOR_INF = torch.tensor(Float.POSITIVE_INFINITY, cpuF32);
        TENSOR_NEG_INF = torch.tensor(Float.NEGATIVE_INFINITY, cpuF32);
        TENSOR_1E_8 = torch.tensor(1e-8f, cpuF32);
        TENSOR_1_MINUS_1E_8 = torch.tensor(1.0f - 1e-8f, cpuF32);
        TENSOR_MAX_K = torch.tensor(1e6f, cpuF32);
        TENSOR_MIN_K = torch.tensor(1e-6f, cpuF32);
        TENSOR_MAX_LAMBDA = torch.tensor(1e6f, cpuF32);
        TENSOR_MIN_LAMBDA = torch.tensor(1e-8f, cpuF32);
        TENSOR_1E6 = torch.tensor(1e6f, cpuF32);
        TENSOR_700 = torch.tensor(700.0f, cpuF32);
        TENSOR_NEG_700 = torch.tensor(-700.0f, cpuF32);
        TENSOR_GAMMA_MAX_INPUT = torch.tensor(500.0f, cpuF32);
        TENSOR_GAMMA_MAX_OUTPUT = torch.tensor(1e38f, cpuF32); // float32 max-ish
    }

    /**
     * 构造函数：修复所有检测逻辑 + 严格API规范
     */
    public Weibull(Tensor scale, Tensor concentration) {
        // 1. 空值校验（测试3通过）
        if (scale == null || concentration == null) {
            throw new IllegalArgumentException("scale和concentration参数不能为空！");
        }

        // 2. 严格按API：scale.to(new Device(torch.kCPU()),kFloat()).clone().detach();
        Tensor scaleFloat32 = scale.to(new Device(torch.kCPU()), kFloat()).clone().detach();
        Tensor concentrationFloat32 = concentration.to(new Device(torch.kCPU()), kFloat()).clone().detach();

        // 3. 修复：形状广播检测（从后往前校验，测试5通过）
        long[] scaleShape = getTensorShape(scaleFloat32);
        long[] concShape = getTensorShape(concentrationFloat32);
        boolean canBroadcast = checkBroadcastCompatibility(scaleShape, concShape);

        if (!canBroadcast) {
            scaleFloat32.close();
            concentrationFloat32.close();
            throw new IllegalArgumentException(
                    String.format("形状无法广播：scale=%s, concentration=%s",
                            shapeToString(scaleShape), shapeToString(concShape))
            );
        }
        this.batchShape = broadcastShapes(scaleShape, concShape);

        // 4. 修复：scale≤0检测（测试1通过）
        Tensor scaleLeZero = torch.le(scaleFloat32, TENSOR_0);
        Tensor scaleLeEps = torch.le(scaleFloat32, TENSOR_1E_8);
        if (torch.any(scaleLeZero).item().toBool() || torch.any(scaleLeEps).item().toBool()) {
            scaleFloat32.close();
            concentrationFloat32.close();
            scaleLeZero.close();
            scaleLeEps.close();
            throw new IllegalArgumentException("尺度参数scale(λ)必须大于0（数值容忍度1e-8）！");
        }

        // 5. 修复：concentration≤0检测（测试2通过）
        Tensor concLeZero = torch.le(concentrationFloat32, TENSOR_0);
        Tensor concLeEps = torch.le(concentrationFloat32, TENSOR_1E_8);
        if (torch.any(concLeZero).item().toBool() || torch.any(concLeEps).item().toBool()) {
            scaleFloat32.close();
            concentrationFloat32.close();
            scaleLeZero.close();
            scaleLeEps.close();
            concLeZero.close();
            concLeEps.close();
            throw new IllegalArgumentException("形状参数concentration(k)必须大于0（数值容忍度1e-8）！");
        }

        // 6. 数值稳定化：严格使用clamp的ScalarOptional参数签名
        ScalarOptional minLambda = new ScalarOptional(TENSOR_MIN_LAMBDA.item());
        ScalarOptional maxLambda = new ScalarOptional(TENSOR_MAX_LAMBDA.item());
        Tensor scaleStabilized = torch.clamp(scaleFloat32, minLambda, maxLambda);

        ScalarOptional minK = new ScalarOptional(TENSOR_MIN_K.item());
        ScalarOptional maxK = new ScalarOptional(TENSOR_MAX_K.item());
        Tensor concStabilized = torch.clamp(concentrationFloat32, minK, maxK);

        // 7. 初始化核心参数
        this.scale = scaleStabilized.clone().detach();
        this.concentration = concStabilized.clone().detach();

        // 8. 预计算关键值（标量运算严格用Tensor）
        this.invConcentration = torch.reciprocal(concStabilized);
        this.logScale = torch.log(this.scale);
        this.logConcentration = torch.log(this.concentration);

        // 释放临时对象
        scaleFloat32.close();
        concentrationFloat32.close();
        scaleLeZero.close();
        scaleLeEps.close();
        concLeZero.close();
        concLeEps.close();
        minLambda.close();
        maxLambda.close();
        minK.close();
        maxK.close();
        scaleStabilized.close();
        concStabilized.close();
    }

    /**
     * 严格的广播兼容性检测（PyTorch官方规则：从后往前）
     */
    private boolean checkBroadcastCompatibility(long[] aShape, long[] bShape) {
        int aDim = aShape.length;
        int bDim = bShape.length;
        int maxDim = Math.max(aDim, bDim);

        for (int i = maxDim - 1; i >= 0; i--) {
            int aIdx = i - (maxDim - aDim);
            int bIdx = i - (maxDim - bDim);

            long aVal = (aIdx >= 0) ? aShape[aIdx] : 1;
            long bVal = (bIdx >= 0) ? bShape[bIdx] : 1;

            if (aVal != 1 && bVal != 1 && aVal != bVal) {
                return false;
            }
        }
        return true;
    }

    @Override
    public String name() {
        return "Weibull";
    }

    /**
     * 采样：严格API初始化随机张量
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();
        long[] extendedShape = concatShapes(sampleShape, batchShape);

        // 扩展参数
        Tensor expandedScale = scale.expand(extendedShape);
        Tensor expandedInvConcentration = invConcentration.expand(extendedShape);

        // 严格按API：torch.tensor(0.0f).options().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(torch.kCPU())))
        TensorOptions randOptions = torch.tensor(0.0f).options()
                .dtype(new ScalarTypeOptional(kFloat()))
                .device(new DeviceOptional(new Device(torch.kCPU())));
        Tensor u = torch.rand(extendedShape, randOptions);

        // clamp调用：严格使用ScalarOptional参数签名
        ScalarOptional uMin = new ScalarOptional(TENSOR_1E_8.item());
        ScalarOptional uMax = new ScalarOptional(TENSOR_1_MINUS_1E_8.item());
        Tensor uSafe = torch.clamp(u, uMin, uMax);

        // 标量运算：严格使用Tensor（无Scalar直接运算）
        Tensor oneMinusUSafe = TENSOR_1.sub(uSafe); // 正确：TENSOR_1是torch.tensor(1.0f)
        Tensor logOneMinusU = torch.log(oneMinusUSafe);
        Tensor negLogOneMinusU = torch.neg(logOneMinusU);

        // clamp调用：严格API签名
        ScalarOptional negLogMin = new ScalarOptional(TENSOR_0.item());
        ScalarOptional negLogMax = new ScalarOptional(TENSOR_1E6.item());
        Tensor negLogOneMinusUSafe = torch.clamp(negLogOneMinusU, negLogMin, negLogMax);

        Tensor negLogOneMinusUPow = torch.pow(negLogOneMinusUSafe, expandedInvConcentration);
        Tensor sample = expandedScale.mul(negLogOneMinusUPow);

        // 释放临时对象
        expandedScale.close();
        expandedInvConcentration.close();
        randOptions.close();
        u.close();
        uMin.close();
        uMax.close();
        uSafe.close();
        oneMinusUSafe.close();
        logOneMinusU.close();
        negLogOneMinusU.close();
        negLogMin.close();
        negLogMax.close();
        negLogOneMinusUSafe.close();
        negLogOneMinusUPow.close();

        return sample;
    }

    /**
     * log_prob：修复精度 + 严格API规范
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 严格按API：to(new Device(torch.kCPU()),kFloat())
        Tensor vFloat32 = v.to(new Device(torch.kCPU()), kFloat()).clone().detach();
        long[] vShape = getTensorShape(vFloat32);
        long[] finalShape = broadcastShapes(vShape, batchShape);
        Tensor vExpanded = vFloat32.expand(finalShape);

        // 扩展参数
        Tensor expandedScale = scale.expand(finalShape);
        Tensor expandedConcentration = concentration.expand(finalShape);
        Tensor expandedLogScale = logScale.expand(finalShape);
        Tensor expandedLogConcentration = logConcentration.expand(finalShape);

        // 校验输入值域
        Tensor xGe0 = torch.ge(vExpanded, TENSOR_0);

        // 仅处理x=0，不修改正常输入
        Tensor xSafe = torch.where(xGe0, vExpanded, TENSOR_1E_8);

        // 严格匹配理论公式（无过度限制）
        Tensor logX = torch.log(xSafe);
        // term1 = logk - logλ（Tensor运算）
        Tensor term1 = expandedLogConcentration.sub(expandedLogScale);
        // term2 = (k-1)(logx - logλ)（k-1用TENSOR_1.sub）
        Tensor logXMinusLogScale = logX.sub(expandedLogScale);
        Tensor kMinus1 = expandedConcentration.sub(TENSOR_1); // 正确：TENSOR_1是torch.tensor(1.0f)
        Tensor term2 = kMinus1.mul(logXMinusLogScale);
        // term3 = -(x/λ)^k（精准计算）
        Tensor xOverScale = xSafe.div(expandedScale);
        Tensor xOverScalePowK = torch.pow(xOverScale, expandedConcentration);
        Tensor term3 = xOverScalePowK.neg();

        // 合法输入的对数概率
        Tensor logProbValid = term1.add(term2).add(term3);
        // 非法输入返回-∞
        Tensor logProb = torch.where(
                xGe0,
                logProbValid,
                torch.full_like(logProbValid, TENSOR_NEG_INF.item(),
                        logProbValid.options(), new MemoryFormatOptional())
        );

        // 释放临时对象
        vFloat32.close();
        vExpanded.close();
        expandedScale.close();
        expandedConcentration.close();
        expandedLogScale.close();
        expandedLogConcentration.close();
        xGe0.close();
        xSafe.close();
        logX.close();
        term1.close();
        logXMinusLogScale.close();
        kMinus1.close();
        term2.close();
        xOverScale.close();
        xOverScalePowK.close();
        term3.close();
        logProbValid.close();

        return logProb;
    }

    /**
     * 均值：修复Infinity + 严格API
     */
    @Override
    public Tensor mean() {
        checkClosed();
        // 1 + 1/k（严格Tensor运算）
        Tensor onePlusInvK = TENSOR_1.add(invConcentration); // 正确：TENSOR_1是torch.tensor(1.0f)

        // 动态限制伽马函数输入
        Tensor onePlusInvKSafe = torch.where(
                onePlusInvK.gt(TENSOR_GAMMA_MAX_INPUT),
                TENSOR_GAMMA_MAX_INPUT,
                onePlusInvK
        );

        // 计算伽马函数
        Tensor lgammaVal = lgamma(onePlusInvKSafe);
        // clamp调用：严格API签名
        ScalarOptional lgammaMin = new ScalarOptional(TENSOR_NEG_700.item());
        ScalarOptional lgammaMax = new ScalarOptional(TENSOR_700.item());
        lgammaVal = torch.clamp(lgammaVal, lgammaMin, lgammaMax);

        Tensor gammaVal = torch.exp(lgammaVal);
        // clamp调用：严格API签名
        ScalarOptional gammaMin = new ScalarOptional(TENSOR_0.item());
        ScalarOptional gammaMax = new ScalarOptional(TENSOR_GAMMA_MAX_OUTPUT.item());
        gammaVal = torch.clamp(gammaVal, gammaMin, gammaMax);

        // 最终均值
        Tensor mean = scale.mul(gammaVal);

        // 释放临时对象
        onePlusInvK.close();
        onePlusInvKSafe.close();
        lgammaVal.close();
        lgammaMin.close();
        lgammaMax.close();
        gammaVal.close();
        gammaMin.close();
        gammaMax.close();

        return mean;
    }

    /**
     * 熵：严格API规范
     */
    @Override
    public Tensor entropy() {
        checkClosed();
        // 1 - 1/k（严格Tensor运算）
        Tensor oneMinusInvK = TENSOR_1.sub(invConcentration);
        Tensor term1 = TENSOR_EULER_GAMMA.mul(oneMinusInvK);

        // logλ - logk
        Tensor term2 = logScale.sub(logConcentration);
        // 常数1必须用TENSOR_1（torch.tensor(1.0f)）
        Tensor term3 = torch.full_like(scale, TENSOR_1.item(),
                scale.options(), new MemoryFormatOptional());

        Tensor entropy = term1.add(term2).add(term3);

        // 释放临时对象
        oneMinusInvK.close();
        term1.close();
        term2.close();
        term3.close();

        return entropy;
    }

    /**
     * 方差：严格API规范
     */
    public Tensor variance() {
        checkClosed();
        // 严格Tensor运算（无Scalar直接操作）
        Tensor onePlusInvK = TENSOR_1.add(invConcentration);
        Tensor twoInvK = TENSOR_2.mul(invConcentration); // TENSOR_2是torch.tensor(2.0f)
        Tensor onePlusTwoInvK = TENSOR_1.add(twoInvK);

        // 限制伽马函数输入
        onePlusInvK = torch.where(
                onePlusInvK.gt(TENSOR_GAMMA_MAX_INPUT),
                TENSOR_GAMMA_MAX_INPUT,
                onePlusInvK
        );
        onePlusTwoInvK = torch.where(
                onePlusTwoInvK.gt(TENSOR_GAMMA_MAX_INPUT),
                TENSOR_GAMMA_MAX_INPUT,
                onePlusTwoInvK
        );

        // 计算伽马函数
        Tensor lgamma1 = lgamma(onePlusInvK);
        Tensor lgamma2 = lgamma(onePlusTwoInvK);
        // clamp调用：严格API
        ScalarOptional lgammaMin = new ScalarOptional(TENSOR_NEG_700.item());
        ScalarOptional lgammaMax = new ScalarOptional(TENSOR_700.item());
        lgamma1 = torch.clamp(lgamma1, lgammaMin, lgammaMax);
        lgamma2 = torch.clamp(lgamma2, lgammaMin, lgammaMax);

        Tensor gamma1 = torch.exp(lgamma1);
        Tensor gamma2 = torch.exp(lgamma2);

        // clamp调用：严格API
        ScalarOptional gammaMin = new ScalarOptional(TENSOR_0.item());
        ScalarOptional gammaMax = new ScalarOptional(TENSOR_GAMMA_MAX_OUTPUT.item());
        gamma1 = torch.clamp(gamma1, gammaMin, gammaMax);
        gamma2 = torch.clamp(gamma2, gammaMin, gammaMax);

        // 计算方差
        Tensor gammaSq = gamma1.pow(TENSOR_2); // TENSOR_2是torch.tensor(2.0f)
        Tensor varCore = gamma2.sub(gammaSq);
        // clamp调用：严格API
        ScalarOptional varMin = new ScalarOptional(TENSOR_0.item());
        ScalarOptional varMax = new ScalarOptional(TENSOR_INF.item());
        varCore = torch.clamp(varCore, varMin, varMax);

        Tensor scaleSq = scale.pow(TENSOR_2);
        Tensor variance = scaleSq.mul(varCore);

        // 释放临时对象
        onePlusInvK.close();
        twoInvK.close();
        onePlusTwoInvK.close();
        lgamma1.close();
        lgamma2.close();
        lgammaMin.close();
        lgammaMax.close();
        gamma1.close();
        gamma2.close();
        gammaMin.close();
        gammaMax.close();
        gammaSq.close();
        varCore.close();
        varMin.close();
        varMax.close();
        scaleSq.close();

        return variance;
    }

    // ------------------------------ 辅助方法 ------------------------------
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("Weibull实例已释放，无法继续使用！");
        }
    }

    private long[] getTensorShape(Tensor tensor) {
        long[] shape = new long[(int) tensor.dim()];
        for (int i = 0; i < tensor.dim(); i++) {
            shape[i] = tensor.size(i);
        }
        return shape;
    }

    private long[] broadcastShapes(long[] a, long[] b) {
        int lenA = a.length;
        int lenB = b.length;
        int maxLen = Math.max(lenA, lenB);
        long[] result = new long[maxLen];

        for (int i = maxLen - 1; i >= 0; i--) {
            int aIdx = i - (maxLen - lenA);
            int bIdx = i - (maxLen - lenB);

            long sA = (aIdx >= 0) ? a[aIdx] : 1;
            long sB = (bIdx >= 0) ? b[bIdx] : 1;

            if (sA != 1 && sB != 1 && sA != sB) {
                throw new IllegalArgumentException(
                        String.format("形状不兼容：%s 和 %s", shapeToString(a), shapeToString(b))
                );
            }
            result[i] = Math.max(sA, sB);
        }
        return result;
    }

    private long[] concatShapes(long[] a, long[] b) {
        long[] result = new long[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }

    private String shapeToString(long[] shape) {
        if (shape.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    // ------------------------------ 资源管理 ------------------------------
    @Override
    public void close() {
        if (!isClosed) {
            scale.close();
            concentration.close();
            invConcentration.close();
            logScale.close();
            logConcentration.close();
            isClosed = true;
        }
    }

    // Getter方法
    public Tensor getScale() {
        checkClosed();
        return scale.clone().detach();
    }

    public Tensor getConcentration() {
        checkClosed();
        return concentration.clone().detach();
    }

    public Tensor getInvConcentration() {
        checkClosed();
        return invConcentration.clone().detach();
    }

    public Tensor getLogScale() {
        checkClosed();
        return logScale.clone().detach();
    }

    public Tensor getLogConcentration() {
        checkClosed();
        return logConcentration.clone().detach();
    }

    public long[] getBatchShape() {
        return batchShape.clone();
    }

    // 静态资源释放
    public static void releaseStaticTensors() {
        TENSOR_0.close();
        TENSOR_1.close();
        TENSOR_2.close();
        TENSOR_EPS.close();
        TENSOR_EULER_GAMMA.close();
        TENSOR_INF.close();
        TENSOR_NEG_INF.close();
        TENSOR_1E_8.close();
        TENSOR_1_MINUS_1E_8.close();
        TENSOR_MAX_K.close();
        TENSOR_MIN_K.close();
        TENSOR_MAX_LAMBDA.close();
        TENSOR_MIN_LAMBDA.close();
        TENSOR_1E6.close();
        TENSOR_700.close();
        TENSOR_NEG_700.close();
        TENSOR_GAMMA_MAX_INPUT.close();
        TENSOR_GAMMA_MAX_OUTPUT.close();
    }
}
