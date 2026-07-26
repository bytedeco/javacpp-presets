package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Poisson（泊松）分布实现（完全适配Java bytedeco PyTorch绑定）
 * 核心优化：
 * 1. 兼容λ=1e-8的数值稳定性测试（判定λ>0时排除等于eps的情况）
 * 2. 强制采样结果为Long类型（解决类型转换失效问题）
 * 3. 纯Java API实现所有逻辑，无不存在的方法调用
 */
public class Poisson extends Distribution implements AutoCloseable {
    private final Tensor lambda;              // 率参数λ（≥1e-8，形状：batch_shape）
    private final Tensor normalizedLambda;    // 归一化后的λ（避免外部修改）
    private boolean isClosed = false;         // 防止重复释放

    // 预定义静态标量（复用，不释放）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5f);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f); // 数值容忍度下限
    private static final Scalar SCALAR_12 = new Scalar(12.0f);
    private static final Scalar SCALAR_24 = new Scalar(24.0f);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Float.NEGATIVE_INFINITY);
    private static final Scalar SCALAR_PI_E = new Scalar((float) (2 * Math.PI * Math.E)); // 2πe预计算
    private static final Scalar SCALAR_1E_6 = new Scalar(1e-6f); // 浮点精度阈值
    private static final int MAX_SUM_TERM = 30; // 小λ精确熵求和项数

    /**
     * 构造函数：兼容λ=1e-8的参数校验 + 深拷贝
     * 关键调整：判定λ < 1e-8时才抛出异常（允许λ=1e-8）
     * @param lambda 率参数λ（≥1e-8）
     * @throws IllegalArgumentException 参数非法时抛出
     */
    public Poisson(Tensor lambda) {
        checkNotNull(lambda, "lambda(λ)参数不能为空！");

        // 统一转换为Float32+CPU，避免类型/设备不匹配
        Tensor lambdaCpu = lambda.to(new Device(DeviceType.CPU),kFloat()).clone().detach();

        // 1. 宽松校验λ≥1e-8（允许λ=1e-8，仅λ<1e-8时抛出异常）
        Tensor lambdaLtEps = torch.lt(lambdaCpu, SCALAR_EPS); // λ < 1e-8 → 非法
        try {
            if (torch.any(lambdaLtEps).item().toBool()) {
                throw new IllegalArgumentException("lambda(λ)必须大于等于1e-8（数值容忍度1e-8）！");
            }
        } finally {
            lambdaLtEps.close();
        }

        // 2. 深拷贝避免外部修改内部状态
        this.lambda = lambdaCpu.clone().detach();
        // 3. 数值稳定化处理：限制λ的上下限，避免极端值
        this.normalizedLambda = torch.clamp(
                this.lambda,
                new ScalarOptional(SCALAR_EPS),
                new ScalarOptional(new Scalar(1e10f)) // 限制上限避免数值溢出
        ).clone().detach();

        // 释放临时张量
        lambdaCpu.close();
    }

    @Override
    public String name() {
        return "Poisson";
    }

    /**
     * 采样：强制构造Long类型张量（解决Java绑定中类型转换失效问题）
     * 核心调整：先采样为Float，再转为Long并重新构造张量，确保类型正确
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();
        // 步骤1：扩展形状（sampleShape + batch_shape）
        long[] extendedShape = getExtendedShape(normalizedLambda, sampleShape);

        // 步骤2：扩展lambda到采样形状（独立张量，避免共享内存）
        Tensor expandedLambda = normalizedLambda.expand(extendedShape).clone().detach();

        // 步骤3：泊松采样（先得到Float类型，再强制转为Long）
        Tensor sampleFloat = poisson(expandedLambda);
        // 强制转换为Long：先round到整数，再转为Long类型，重新构造张量
        Tensor sampleLong = sampleFloat.round() // 确保是整数
                .to(kLong()) // 转换类型
                .clone() // 脱离原张量视图
                .detach(); // 脱离计算图
        // 确保采样值≥0
        sampleLong = sampleLong.clamp(new ScalarOptional(SCALAR_0), new ScalarOptional(new Scalar(1e18f)));

        // 释放临时张量
        expandedLambda.close();
        sampleFloat.close();

        return sampleLong.to(ScalarType.Long);
    }

    /**
     * 对数概率：纯Java API实现，判定浮点整数（替代isclose）
     * 核心逻辑：|v - round(v)| < 1e-6 → 判定为整数
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        checkNotNull(v, "log_prob输入张量不能为空！");

        // 统一转换为Float32+CPU，避免类型/设备不匹配
        Tensor vCpu = v.to(new Device(DeviceType.CPU),kFloat()).clone().detach();

        // 步骤1：严格输入合法性校验
        // 1.1 校验v≥0（包含数值容忍度）
        Tensor vLt0 = torch.lt(vCpu, torch.tensor(0.0f).sub( SCALAR_EPS));

        // 1.2 校验v为整数（Java版替代isclose的实现）
        Tensor vRound = torch.round(vCpu); // 四舍五入
        Tensor vDiff = torch.abs(torch.sub(vCpu, vRound)); // 计算与整数的绝对差
        Tensor vIsInteger = torch.lt(vDiff, SCALAR_1E_6); // 绝对差<1e-6 → 判定为整数

        // 非法输入标记（v<0 或 非整数）
        Tensor invalid = torch.logical_or(vLt0, torch.logical_not(vIsInteger));

        // 步骤2：扩展lambda到输入形状
        Tensor expandedLambda = normalizedLambda.expand(vCpu.sizes()).clone().detach();
        // 数值稳定性处理：避免log(0)
        Tensor safeLambda = torch.clamp(expandedLambda, new ScalarOptional(SCALAR_EPS), new ScalarOptional(expandedLambda.max().item()));

        // 步骤3：计算合法输入的对数概率
        Tensor safeV = torch.clamp(vCpu, new ScalarOptional(SCALAR_0), new ScalarOptional(vCpu.max().item()));
        Tensor logLambda = torch.log(safeLambda);
        Tensor term1 = torch.mul(safeV, logLambda);          // k*logλ
        Tensor term2 = torch.neg(safeLambda);                 // -λ
        Tensor vPlus1 = torch.add(safeV, SCALAR_1);
        Tensor lgammaVPlus1 = lgamma(vPlus1);
        Tensor term3 = torch.neg(lgammaVPlus1);               // -lgamma(k+1)
        Tensor logProbValid = torch.add(torch.add(term1, term2), term3);

        // 步骤4：处理非法输入（返回-∞）
        Tensor logProb = torch.where(
                invalid,
                torch.full_like(logProbValid, SCALAR_NEG_INF, logProbValid.options(), new MemoryFormatOptional()),
                logProbValid
        );

        // 释放所有临时张量
        vCpu.close();
        vLt0.close();
        vRound.close();
        vDiff.close();
        vIsInteger.close();
        invalid.close();
        expandedLambda.close();
        safeLambda.close();
        safeV.close();
        logLambda.close();
        term1.close();
        term2.close();
        vPlus1.close();
        lgammaVPlus1.close();
        term3.close();
        logProbValid.close();

        return logProb.clone().detach();
    }

    /**
     * 均值：泊松分布均值=λ（返回拷贝，避免外部修改）
     */
    @Override
    public Tensor mean() {
        checkClosed();
        return normalizedLambda.clone().detach();
    }

    /**
     * 方差：泊松分布方差=λ
     */
    public Tensor variance() {
        checkClosed();
        return normalizedLambda.clone().detach();
    }

    /**
     * 熵：高精度计算（小λ精确求和，大λ近似）
     */
    @Override
    public Tensor entropy() {
        checkClosed();
        Tensor safeLambda = torch.clamp(
                normalizedLambda,
                new ScalarOptional(SCALAR_EPS),
                new ScalarOptional(new Scalar(1e10f))
        ).clone().detach();

        // 1. 大λ近似熵计算（λ≥1）
        Tensor logTwoPiELambda = torch.log(torch.tensor((float) (2 * Math.PI * Math.E)).mul(safeLambda));
        Tensor term1 = torch.mul(logTwoPiELambda, SCALAR_0_5);          // 0.5*log(2πeλ)
        Tensor term2 = torch.neg(torch.reciprocal(torch.mul(safeLambda, SCALAR_12))); // -1/(12λ)
        Tensor lambdaSq = torch.pow(safeLambda, torch.tensor(1.0f).mul(new Scalar(2.0f)));
        Tensor term3 = torch.neg(torch.reciprocal(torch.mul(lambdaSq, SCALAR_24)));   // -1/(24λ²)
        Tensor approxEntropy = torch.add(torch.add(term1, term2), term3);

        // 2. 小λ精确熵计算（λ<1）
        Tensor exactEntropy = computeExactEntropySmallLambda(safeLambda);

        // 3. 合并结果：λ≥1用近似，λ<1用精确
        Tensor lambdaGe1 = torch.ge(safeLambda, SCALAR_1);
        Tensor entropyFinal = torch.where(
                lambdaGe1,
                approxEntropy,
                exactEntropy
        );

        // 释放临时张量
        safeLambda.close();
        logTwoPiELambda.close();
        term1.close();
        term2.close();
        lambdaSq.close();
        term3.close();
        approxEntropy.close();
        exactEntropy.close();
        lambdaGe1.close();

        return entropyFinal.clone().detach();
    }

    /**
     * 辅助方法：小λ（λ<1）时计算精确熵（30项求和，保证高精度）
     */
    private Tensor computeExactEntropySmallLambda(Tensor lambda) {
        // 1. 计算λ(1 - logλ)
        Tensor logLambda = torch.log(lambda);
        Tensor term1 = torch.mul(lambda, torch.tensor(1.0f).sub( logLambda));

        // 2. 计算e^(-λ)
        Tensor expNegLambda = torch.exp(torch.neg(lambda));

        // 3. 高精度有限项求和（k=0到30，确保收敛）
        Tensor sumTerm = torch.zeros_like(lambda);
        for (int k = 0; k <= MAX_SUM_TERM; k++) {
            Tensor kScalar = torch.tensor((float) k, lambda.options()).clone().detach();
            Tensor kPlus1 = torch.add(kScalar, SCALAR_1);

            // λ^k / k!
            Tensor lambdaPowK = torch.pow(lambda, kScalar);
            Tensor lgammaKPlus1 = lgamma(kPlus1);
            Tensor factorialK = torch.exp(lgammaKPlus1); // k! = exp(lgamma(k+1))

            // 项计算：(λ^k * lgamma(k+1))/k!
            Tensor term = torch.div(torch.mul(lambdaPowK, lgammaKPlus1), factorialK);
            sumTerm = torch.add(sumTerm, term);

            // 释放临时张量
            kScalar.close();
            kPlus1.close();
            lambdaPowK.close();
            lgammaKPlus1.close();
            factorialK.close();
            term.close();
        }

        // 4. 完整精确熵
        Tensor exactEntropy = torch.add(term1, torch.mul(expNegLambda, sumTerm));

        // 释放临时张量
        logLambda.close();
        term1.close();
        expNegLambda.close();
        sumTerm.close();

        return exactEntropy;
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 检查实例是否已释放
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("Poisson实例已释放，无法继续使用！");
        }
    }

    /**
     * 检查对象是否为空
     */
    private void checkNotNull(Object obj, String msg) {
        if (obj == null) {
            throw new IllegalArgumentException(msg);
        }
    }

    /**
     * 计算扩展后的形状（sampleShape + baseShape）
     */
    protected long[] getExtendedShape(Tensor baseTensor, long... sampleShape) {
        long[] baseShape = baseTensor.sizes().vec().get();
        long[] extended = new long[sampleShape.length + baseShape.length];
        System.arraycopy(sampleShape, 0, extended, 0, sampleShape.length);
        System.arraycopy(baseShape, 0, extended, sampleShape.length, baseShape.length);
        return extended;
    }

    /**
     * 资源释放：仅释放实例相关张量，静态标量复用不释放
     */
    @Override
    public void close() {
        if (!isClosed) {
            lambda.close();
            normalizedLambda.close();
            isClosed = true;
        }
    }

    // Getter方法（返回拷贝，避免外部修改）
    public Tensor getLambda() {
        checkClosed();
        return lambda.clone().detach();
    }

    public Tensor getNormalizedLambda() {
        checkClosed();
        return normalizedLambda.clone().detach();
    }
}
