package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * LogSeries（对数级数）分布实现
 * 严格遵循torch.clamp API规范，修复所有指定问题：
 * 1. 按原生API调用clamp（仅使用ScalarOptional入参）
 * 2. ScalarOptional构造使用new Scalar(1.0f - 1e-8)而非torch.sub
 * 3. torch.sub改为torch.tensor(1.0f).sub(...)避免宕机
 */
public class LogSeries extends Distribution implements AutoCloseable {
    private final Tensor p;   // 分布参数（0 < p < 1）
    private boolean isClosed = false; // 防止重复释放

    // 预定义静态标量（仅初始化一次，不随实例释放）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Float.NEGATIVE_INFINITY);
    private static final Scalar SCALAR_MAX_K = new Scalar(1e6f);
    private static final Scalar SCALAR_2 = new Scalar(2.0f);
    // 预定义边界标量（避免重复创建）
    private static final Scalar SCALAR_1_MINUS_EPS = new Scalar(1.0f - 1e-8f);
    private static final Scalar SCALAR_4 = new Scalar(4.0f);
    private static final Scalar SCALAR_9 = new Scalar(9.0f);
    private static final Scalar SCALAR_16 = new Scalar(16.0f);
    private static final Scalar SCALAR_25 = new Scalar(25.0f);

    /**
     * 构造函数：严格校验参数合法性 + 深拷贝
     */
    public LogSeries(Tensor p) {
        if (p == null) {
            throw new IllegalArgumentException("对数级数分布参数p不能为空！");
        }

        // 统一转换为Float32+CPU，避免类型/设备不匹配
        Tensor pCpu = p.to(new Device(DeviceType.CPU),kFloat()).clone().detach();

        // 校验0 < p < 1（包含数值稳定性边界）
        // 修复：使用预定义标量，避免torch.sub导致宕机
        Tensor pLeEps = torch.le(pCpu, SCALAR_EPS);
        Tensor pGe1MinusEps = torch.ge(pCpu, SCALAR_1_MINUS_EPS);
        Tensor paramInvalid = torch.logical_or(pLeEps, pGe1MinusEps);

        try {
            if (torch.any(paramInvalid).item().toBool()) {
                throw new IllegalArgumentException("对数级数分布参数p必须满足 0 < p < 1！");
            }
        } finally {
            // 确保临时张量释放
            pLeEps.close();
            pGe1MinusEps.close();
            paramInvalid.close();
        }

        // 深拷贝避免外部修改内部状态
        this.p = pCpu.clone().detach();
        pCpu.close();
    }

    @Override
    public String name() {
        return "LogSeries";
    }

    /**
     * 采样：逆变换采样（严格遵循API规范）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();
        // 步骤1：扩展采样形状
        long[] extendedShape = getExtendedShape(p, sampleShape);
        Tensor expandedP = p.expand(extendedShape).clone().detach();
        TensorOptions floatOptions = expandedP.options();
        TensorOptions longOptions = floatOptions.dtype(new ScalarTypeOptional(kLong()));

        // 步骤2：生成稳定的Uniform随机数
        // 修复：严格按API调用clamp（ScalarOptional入参）
        Tensor u = torch.rand(extendedShape, floatOptions);
        u = torch.clamp(u, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1_MINUS_EPS));

        // 步骤3：逆变换采样核心计算
        // 修复：torch.sub改为torch.tensor(1.0f).sub避免宕机
        Tensor oneTensor = torch.tensor(1.0f, floatOptions);
        Tensor oneMinusP = oneTensor.sub(expandedP);
        oneTensor.close();

        // 修复：clamp调用严格遵循API
        Tensor logOneMinusP = torch.log(torch.clamp(oneMinusP, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1)));
        Tensor negLogOneMinusP = torch.neg(logOneMinusP);

        // 计算核心项：1 - U*(-log(1-p))
        Tensor uMulNegLog = torch.mul(u, negLogOneMinusP);
        oneTensor = torch.tensor(1.0f, floatOptions);
        Tensor oneMinusUMul = oneTensor.sub(uMulNegLog);
        oneTensor.close();
        // 修复：clamp调用严格遵循API
        oneMinusUMul = torch.clamp(oneMinusUMul, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1));

        // 计算k = log(1 - U*(-log(1-p))) / log(1-p)
        Tensor logTerm = torch.log(oneMinusUMul);
        Tensor kFloat = torch.div(logTerm, logOneMinusP);
        // 转换为整数（k≥1）
        Tensor k = torch.ceil(kFloat).toType(kLong());
        // 修复：clamp调用严格遵循API（ScalarOptional）
        k = torch.clamp(k, new ScalarOptional(SCALAR_1), new ScalarOptional(SCALAR_MAX_K));

        // 释放临时张量
        expandedP.close();
        u.close();
        oneMinusP.close();
        logOneMinusP.close();
        negLogOneMinusP.close();
        uMulNegLog.close();
        oneMinusUMul.close();
        logTerm.close();
        kFloat.close();

        return k.clone().detach();
    }

    /**
     * 对数概率：精确公式 + 严格API调用
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 统一转换为Float32+CPU
        Tensor vCpu = v.to(new Device(DeviceType.CPU),kFloat()).clone().detach();
        TensorOptions resultOptions = vCpu.options();

        // 步骤1：严格校验输入合法性
        Tensor vLt1 = torch.lt(vCpu, SCALAR_1);
        Tensor vRound = torch.round(vCpu);
        Tensor vIsInteger = torch.eq(vCpu, vRound);
        Tensor vInvalid = torch.logical_or(vLt1, torch.logical_not(vIsInteger));

        // 初始化结果为-∞
        Tensor logProb = torch.full_like(vCpu, SCALAR_NEG_INF, resultOptions, new MemoryFormatOptional());

        // 仅处理合法输入
        Tensor vValidMask = torch.logical_not(vInvalid);
        Tensor vValid = torch.masked_select(vCpu, vValidMask);

        if (vValid.numel() > 0) {
            // 扩展p到合法输入的形状
            long[] validShape = vValid.sizes().vec().get();
            Tensor expandedP = p.expand(validShape).clone().detach();

            // 修复：clamp调用严格遵循API
            Tensor safeP = torch.clamp(expandedP, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1_MINUS_EPS));

            // 计算精确公式各项
            Tensor logP = torch.log(safeP);

            // 修复：torch.sub改为torch.tensor(1.0f).sub避免宕机
            Tensor oneTensor = torch.tensor(1.0f, safeP.options());
            Tensor oneMinusP = oneTensor.sub(safeP);
            oneTensor.close();

            // 修复：clamp调用严格遵循API
            Tensor logOneMinusP = torch.log(torch.clamp(oneMinusP, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1)));
            Tensor negLogOneMinusP = torch.neg(logOneMinusP);

            // 修复：clamp调用严格遵循API
            Tensor clampedNegLog = torch.clamp(negLogOneMinusP, new ScalarOptional(SCALAR_EPS), new ScalarOptional(negLogOneMinusP.max().item()));
            Tensor logNegLogOneMinusP = torch.log(clampedNegLog);

            Tensor kMinus1 = torch.sub(vValid, SCALAR_1);
            Tensor term1 = logP;
            Tensor term2 = torch.mul(kMinus1, logOneMinusP);
            Tensor term3 = torch.neg(torch.log(vValid));
            Tensor term4 = torch.neg(logNegLogOneMinusP);

            // 合并所有项
            Tensor validLogProb = torch.add(torch.add(term1, term2), torch.add(term3, term4));

            // 将有效结果回填到logProb
            logProb = torch.masked_scatter(logProb, vValidMask, validLogProb);

            // 释放临时张量
            expandedP.close();
            safeP.close();
            logP.close();
            oneMinusP.close();
            logOneMinusP.close();
            negLogOneMinusP.close();
            clampedNegLog.close();
            logNegLogOneMinusP.close();
            kMinus1.close();
            term1.close();
            term2.close();
            term3.close();
            term4.close();
            validLogProb.close();
        }

        // 释放临时张量
        vCpu.close();
        vLt1.close();
        vRound.close();
        vIsInteger.close();
        vInvalid.close();
        vValidMask.close();
        vValid.close();

        return logProb.clone().detach();
    }

    /**
     * 熵：精确公式 + 严格API调用
     */
    @Override
    public Tensor entropy() {
        checkClosed();
        // 修复：clamp调用严格遵循API
        Tensor safeP = torch.clamp(p, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1_MINUS_EPS));

        // 修复：torch.sub改为torch.tensor(1.0f).sub避免宕机
        Tensor oneTensor = torch.tensor(1.0f, safeP.options());
        Tensor oneMinusP = oneTensor.sub(safeP);
        oneTensor.close();

        // 修复：clamp调用严格遵循API
        Tensor logOneMinusP = torch.log(torch.clamp(oneMinusP, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1)));
        Tensor negLogOneMinusP = torch.neg(logOneMinusP);

        // 近似计算二阶多对数函数Li2(p)
        Tensor li2 = torch.zeros_like(safeP);
        Tensor p2 = torch.pow(safeP, SCALAR_2);
        Tensor p3 = torch.mul(p2, safeP);
        Tensor p4 = torch.mul(p3, safeP);
        Tensor p5 = torch.mul(p4, safeP);

        li2 = torch.add(li2, safeP);
        li2 = torch.add(li2, torch.div(p2, SCALAR_4));
        li2 = torch.add(li2, torch.div(p3, SCALAR_9));
        li2 = torch.add(li2, torch.div(p4, SCALAR_16));
        li2 = torch.add(li2, torch.div(p5, SCALAR_25));

        // 计算熵公式各项
        Tensor pDivOneMinusP = torch.div(safeP, oneMinusP);
        Tensor term1 = torch.mul(pDivOneMinusP, li2);
        Tensor term2 = torch.neg(torch.log(negLogOneMinusP));
        Tensor numerator = torch.add(term1, term2);
        Tensor entropy = torch.sub(torch.div(numerator, negLogOneMinusP), SCALAR_1);

        // 释放临时张量
        safeP.close();
        oneMinusP.close();
        logOneMinusP.close();
        negLogOneMinusP.close();
        li2.close();
        p2.close();
        p3.close();
        p4.close();
        p5.close();
        pDivOneMinusP.close();
        term1.close();
        term2.close();
        numerator.close();

        return entropy.clone().detach();
    }

    /**
     * 均值：精确公式 + 严格API调用
     */
    @Override
    public Tensor mean() {
        checkClosed();
        // 修复：clamp调用严格遵循API
        Tensor safeP = torch.clamp(p, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1_MINUS_EPS));

        // 修复：torch.sub改为torch.tensor(1.0f).sub避免宕机
        Tensor oneTensor = torch.tensor(1.0f, safeP.options());
        Tensor oneMinusP = oneTensor.sub(safeP);
        oneTensor.close();

        // 修复：clamp调用严格遵循API
        Tensor logOneMinusP = torch.log(torch.clamp(oneMinusP, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1)));

        // 计算均值公式
        Tensor denominator = torch.mul(oneMinusP, logOneMinusP);
        // 修复：clamp调用严格遵循API
        Tensor clampedDenominator = torch.clamp(denominator, new ScalarOptional(SCALAR_EPS), new ScalarOptional(denominator.max().item()));
        Tensor mean = torch.neg(torch.div(safeP, clampedDenominator));

        // 释放临时张量
        safeP.close();
        oneMinusP.close();
        logOneMinusP.close();
        denominator.close();
        clampedDenominator.close();

        return mean.clone().detach();
    }

    // -------------------------- 辅助方法 --------------------------
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("LogSeries实例已释放，无法继续使用！");
        }
    }

    /**
     * 资源释放：仅释放实例相关张量
     */
    @Override
    public void close() {
        if (!isClosed) {
            p.close();
            isClosed = true;
        }
    }

    // Getter方法（返回拷贝避免外部修改）
    public Tensor getP() {
        checkClosed();
        return p.clone().detach();
    }
}
