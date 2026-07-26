package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.internal.GammaSampler;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * FisherSnedecor（F分布/费希尔-斯内德科尔分布）稳定版：
 * 1. 修复异常参数检测（df1/df2=0）
 * 2. 修复采样算法错误（改用原生gamma采样，均值回归理论值）
 * 3. 增强数值稳定性（极端参数下无溢出/NaN/Inf）
 * 4. 优化资源管理（避免多次释放static标量）
 * 5. 完善批量参数/广播支持
 */
public class FisherSnedecor extends Distribution implements AutoCloseable {
    private final Tensor df1;          // 分子自由度（必须>0）
    private final Tensor df2;          // 分母自由度（必须>0）
    private final TensorOptions baseOptions; // 基础设备/类型配置
    private final long[] batchShape;    // 批量形状

    // 预定义标量（static全局复用，避免重复创建）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_2 = new Scalar(2.0f);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f); // 数值稳定性极小值
    private static final Scalar SCALAR_MAX = new Scalar(1e6f);   // 数值截断最大值
    private static final Scalar SCALAR_MIN = new Scalar(-1e6f);  // 数值截断最小值

    // Optional包装器（避免重复创建）
    private static final ScalarOptional OPTIONAL_EPS = new ScalarOptional(SCALAR_EPS);
    private static final ScalarOptional OPTIONAL_1_MINUS_EPS = new ScalarOptional(new Scalar(1.0f - 1e-8f));

    // 构造函数：严格参数校验 + 深拷贝 + 批量形状计算
    public FisherSnedecor(Tensor df1, Tensor df2) {
        // 1. 统一转换为Float32+CPU，避免类型/设备不匹配
        this.df1 = df1.to(kFloat()).to(torch.device(new Device(DeviceType.CPU)), false, true, new MemoryFormatOptional()).clone().detach();
        this.df2 = df2.to(kFloat()).to(torch.device(new Device(DeviceType.CPU)), false, true, new MemoryFormatOptional()).clone().detach();
        this.baseOptions = this.df1.options();
        this.batchShape = getBroadcastedShape(this.df1.sizes().vec().get(), this.df2.sizes().vec().get());

        // 2. 严格校验：df1/df2必须>0（包括检测0值）
        Tensor df1Le0 = torch.le(this.df1, SCALAR_0);
        Tensor df2Le0 = torch.le(this.df2, SCALAR_0);
        Tensor dfInvalid = torch.logical_or(df1Le0, df2Le0);
        try {
            if (torch.any(dfInvalid).item().toBool()) {
                throw new IllegalArgumentException("F分布df1/df2必须大于0！");
            }
        } finally {
            safeClose(df1Le0);
            safeClose(df2Le0);
            safeClose(dfInvalid); // 及时释放临时张量
        }
    }

    @Override
    public String name() {
        return "FisherSnedecor";
    }

    /**
     * 采样：修复算法错误，改用原生gamma采样（保证均值正确）
     * 核心公式：F = (X1/df1) / (X2/df2)，其中 X1~Gamma(df1/2, 1), X2~Gamma(df2/2, 1)
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：处理采样形状（sampleShape + batchShape）
        long[] safeSampleShape = sampleShape == null || sampleShape.length == 0 ? new long[]{1} : sampleShape;
        long[] extendedShape = concatLongArrays(safeSampleShape, batchShape);

        // 步骤2：扩展自由度到采样形状（保证维度对齐）
        Tensor expandedDf1 = expandToShape(df1, extendedShape);
        Tensor expandedDf2 = expandToShape(df2, extendedShape);

        // 步骤3：Gamma(df/2, 1)采样（改用原生gamma函数，修复算法错误）
        Tensor alpha1 = torch.mul(expandedDf1, SCALAR_0_5);
        Tensor alpha2 = torch.mul(expandedDf2, SCALAR_0_5);
        // 数值稳定化：避免alpha过小导致gamma采样异常
        alpha1 = torch.clamp(alpha1, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));
        alpha2 = torch.clamp(alpha2, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));
        
        Tensor x1 = GammaSampler.gamma(alpha1, torch.ones_like(alpha1));//torch.gamma(alpha1, baseOptions); // X1 ~ Gamma(df1/2, 1)
        Tensor x2 = GammaSampler.gamma(alpha2, torch.ones_like(alpha2));///torch.gamma(alpha2, baseOptions); // X2 ~ Gamma(df2/2, 1)

        // 步骤4：计算F分布采样值：F = (x1/df1) / (x2/df2) = (x1*df2)/(x2*df1)
        Tensor numerator = torch.mul(x1, expandedDf2);
        Tensor denominator = torch.mul(x2, expandedDf1);
        // 增强除零保护：分母小于eps时替换为eps
        denominator = torch.clamp(denominator, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));
        Tensor fSample = torch.div(numerator, denominator);
        // 数值稳定化：F分布>0，避免极小值
        fSample = torch.clamp(fSample, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));

        // 步骤5：调整形状（移除多余维度）
        if (safeSampleShape.length == 1 && safeSampleShape[0] == 1 && sampleShape.length == 0) {
            fSample = fSample.squeeze(0);
        }

        // 释放临时张量
        safeClose(expandedDf1);
        safeClose(expandedDf2);
        safeClose(alpha1);
        safeClose(alpha2);
        safeClose(x1);
        safeClose(x2);
        safeClose(numerator);
        safeClose(denominator);

        return fSample.clone().detach();
    }

    /**
     * 对数概率：修复数值稳定性，完善广播对齐
     */
    @Override
    public Tensor log_prob(Tensor v) {
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 步骤1：统一输入类型/设备
        Tensor input = v.to(baseOptions,false, true, new MemoryFormatOptional()).clone().detach();
        long[] originalShape = input.shape();

        // 步骤2：严格校验输入v>0（包括检测0值）
        Tensor vLe0 = torch.le(input, SCALAR_0);
        try {
            if (torch.any(vLe0).item().toBool()) {
                throw new IllegalArgumentException("log_prob输入v必须大于0！");
            }
        } finally {
            safeClose(vLe0);
        }

        // 步骤3：广播输入和自由度到相同形状
        TensorVector broadcastTensors = torch.broadcast_tensors(new TensorVector(input, df1, df2));
        Tensor inputBroadcast = broadcastTensors.get(0);
        Tensor df1Broadcast = broadcastTensors.get(1);
        Tensor df2Broadcast = broadcastTensors.get(2);

        // 步骤4：数值稳定性处理（避免log(0)或溢出）
        Tensor safeV = torch.clamp(inputBroadcast, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));
        Tensor safeDf1 = torch.clamp(df1Broadcast, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));
        Tensor safeDf2 = torch.clamp(df2Broadcast, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));

        // 步骤5：计算对数概率公式（严格对齐数学定义）
        Tensor halfDf1 = torch.mul(safeDf1, SCALAR_0_5);
        Tensor halfDf2 = torch.mul(safeDf2, SCALAR_0_5);
        Tensor halfSum = torch.add(halfDf1, halfDf2);

        // term1 = lgamma((df1+df2)/2) - lgamma(df1/2) - lgamma(df2/2)
        Tensor term1 = torch.sub(
                torch.sub(torch.lgamma(halfSum), torch.lgamma(halfDf1)),
                torch.lgamma(halfDf2)
        );

        // term2 = (df1/2)*log(df1) + (df2/2)*log(df2)
        Tensor logDf1 = torch.log(safeDf1);
        Tensor logDf2 = torch.log(safeDf2);
        Tensor term2 = torch.add(
                torch.mul(halfDf1, logDf1),
                torch.mul(halfDf2, logDf2)
        );

        // term3 = ((df1/2)-1)*log(v)
        Tensor term3 = torch.mul(
                torch.sub(halfDf1, SCALAR_1),
                torch.log(safeV)
        );

        // term4 = ((df1+df2)/2)*log(df1*v + df2)
        Tensor df1MulV = torch.mul(safeDf1, safeV);
        Tensor df1VAddDf2 = torch.add(df1MulV, safeDf2);
        df1VAddDf2 = torch.clamp(df1VAddDf2, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX)); // 避免log(0)
        Tensor term4 = torch.mul(halfSum, torch.log(df1VAddDf2));

        // 完整对数概率：term1 + term2 + term3 - term4
        Tensor logProb = torch.sub(
                torch.add(torch.add(term1, term2), term3),
                term4
        );

        // 数值截断：避免极端值
        logProb = torch.clamp(logProb, new ScalarOptional(SCALAR_MIN), new ScalarOptional(SCALAR_MAX));
        // 恢复原始形状
        logProb = logProb.reshape(originalShape);

        // 释放所有临时张量
        safeClose(input);
        safeClose(safeV);
        safeClose(safeDf1);
        safeClose(safeDf2);
        safeClose(halfDf1);
        safeClose(halfDf2);
        safeClose(halfSum);
        safeClose(term1);
        safeClose(logDf1);
        safeClose(logDf2);
        safeClose(term2);
        safeClose(term3);
        safeClose(df1MulV);
        safeClose(df1VAddDf2);
        safeClose(term4);
//        for (Tensor t : broadcastTensors) {
//            safeClose(t);
//        }

        return logProb;
    }

    /**
     * 均值：严格对齐数学定义，完善特殊值处理
     * 公式：df2/(df2-2)（df2>2→有效；df2=2→Inf；df2<2→NaN）
     */
    @Override
    public Tensor mean() {
        Tensor twoTensor = torch.tensor(2.0f, baseOptions);
        Tensor df2Minus2 = torch.sub(df2, twoTensor);

        // 构建掩码：df2>2 / df2==2 / 其他
        Tensor maskGt2 = torch.gt(df2, twoTensor);
        Tensor maskEq2 = torch.eq(df2, twoTensor);

        // 计算基础均值（除零保护）
        df2Minus2 = torch.where(
                torch.eq(df2Minus2, SCALAR_0),
                torch.full_like(df2Minus2, SCALAR_EPS),
                df2Minus2
        );
        Tensor meanBase = torch.div(df2, df2Minus2);

        // 替换特殊值：df2==2→Inf，其他→NaN
        Tensor mean = torch.where(
                maskGt2,
                meanBase,
                torch.where(
                        maskEq2,
                        torch.full_like(df2, new Scalar(Float.POSITIVE_INFINITY), baseOptions,new MemoryFormatOptional()),
                        torch.full_like(df2, new Scalar(Float.NaN), baseOptions,new MemoryFormatOptional())
                )
        );

        // 释放临时张量
        safeClose(twoTensor);
        safeClose(df2Minus2);
        safeClose(maskGt2);
        safeClose(maskEq2);
        safeClose(meanBase);

        return mean.clone().detach();
    }

    /**
     * 熵：增强数值稳定性，避免极端参数下溢出
     */
    @Override
    public Tensor entropy() {
        // 数值稳定化：避免极小/极大df导致gamma/digamma溢出
        Tensor safeDf1 = torch.clamp(df1, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));
        Tensor safeDf2 = torch.clamp(df2, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));

        Tensor halfDf1 = torch.mul(safeDf1, SCALAR_0_5);
        Tensor halfDf2 = torch.mul(safeDf2, SCALAR_0_5);
        Tensor halfSum = torch.add(halfDf1, halfDf2);

        // term1 = lgamma(df1/2) + lgamma(df2/2) - lgamma((df1+df2)/2)
        Tensor term1 = torch.sub(
                torch.add(torch.lgamma(halfDf1), torch.lgamma(halfDf2)),
                torch.lgamma(halfSum)
        );

        // term2 = (1 - df1/2) * digamma(df1/2)
        Tensor oneTensor = torch.tensor(1.0f, baseOptions);
        Tensor term2 = torch.mul(
                torch.sub(oneTensor, halfDf1),
                torch.digamma(halfDf1)
        );

        // term3 = - (1 + df2/2) * digamma(df2/2)
        Tensor term3 = torch.neg(
                torch.mul(
                        torch.add(oneTensor, halfDf2),
                        torch.digamma(halfDf2)
                )
        );

        // term4 = (df1+df2)/2 * digamma((df1+df2)/2)
        Tensor term4 = torch.mul(halfSum, torch.digamma(halfSum));

        // term5 = log(df2/df1)（数值稳定版）
        Tensor dfRatio = torch.div(safeDf2, safeDf1);
        dfRatio = torch.clamp(dfRatio, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_MAX));
        Tensor term5 = torch.log(dfRatio);

        // 完整熵：term1 + term2 + term3 + term4 + term5
        Tensor entropy = torch.add(
                torch.add(torch.add(torch.add(term1, term2), term3), term4),
                term5
        );

        // 数值截断：避免极端值
        entropy = torch.clamp(entropy, new ScalarOptional(SCALAR_MIN), new ScalarOptional(SCALAR_MAX));

        // 释放临时张量
        safeClose(safeDf1);
        safeClose(safeDf2);
        safeClose(halfDf1);
        safeClose(halfDf2);
        safeClose(halfSum);
        safeClose(term1);
        safeClose(oneTensor);
        safeClose(term2);
        safeClose(term3);
        safeClose(term4);
        safeClose(dfRatio);
        safeClose(term5);

        return entropy.clone().detach();
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 扩展张量到目标形状（兼容广播）
     */
    private Tensor expandToShape(Tensor tensor, long[] targetShape) {
        Tensor expanded = tensor.clone().detach();
        int tensorDim = (int)expanded.dim();
        int targetDim = targetShape.length;

        // 前置补1维
        for (int i = 0; i < targetDim - tensorDim; i++) {
            expanded = expanded.unsqueeze(0);
        }

        return expanded.expand(targetShape);
    }

    /**
     * 计算两个形状的广播结果
     */
    private long[] getBroadcastedShape(long[] shape1, long[] shape2) {
        int len1 = shape1.length;
        int len2 = shape2.length;
        int maxLen = Math.max(len1, len2);

        long[] result = new long[maxLen];
        for (int i = 0; i < maxLen; i++) {
            long s1 = (i >= len1) ? 1 : shape1[len1 - 1 - i];
            long s2 = (i >= len2) ? 1 : shape2[len2 - 1 - i];

            if (s1 == 1) {
                result[maxLen - 1 - i] = s2;
            } else if (s2 == 1) {
                result[maxLen - 1 - i] = s1;
            } else if (s1 == s2) {
                result[maxLen - 1 - i] = s1;
            } else {
                throw new IllegalArgumentException("无法广播形状：" + Arrays.toString(shape1) + " 和 " + Arrays.toString(shape2));
            }
        }
        return result;
    }

    /**
     * 拼接长数组
     */
    private long[] concatLongArrays(long[] a, long[] b) {
        if (a == null) return b;
        if (b == null) return a;
        long[] result = new long[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }

    /**
     * 安全释放资源（避免空指针和多次释放）
     */
    private void safeClose(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                System.err.println("FisherSnedecor资源释放警告：" + e.getMessage());
            }
        }
    }

    // 资源释放：仅释放实例张量，static标量全局复用不释放
    @Override
    public void close() {
        safeClose(df1);
        safeClose(df2);
    }

    // Getter方法
    public Tensor getDf1() {
        return df1.clone().detach();
    }

    public Tensor getDf2() {
        return df2.clone().detach();
    }

    public long[] getBatchShape() {
        return Arrays.copyOf(batchShape, batchShape.length);
    }
}
