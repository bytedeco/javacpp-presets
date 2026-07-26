package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.internal.GammaSampler;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Gamma（伽马分布）稳定版：
 * 严格适配 bytedeco/pytorch 的 clamp API 规范
 * 1. 修复异常参数检测（α=0/β=0/v≤0）
 * 2. 修复log_prob/熵的数值计算错误
 * 3. 增强数值稳定性（基于原生 clamp API）
 * 4. 完善资源管理，无内存泄漏
 */
public class Gamma extends Distribution implements AutoCloseable {
    private final Tensor concentration; // 形状参数α（必须>0）
    private final Tensor rate;          // 速率参数β（必须>0）
    private final TensorOptions baseOptions; // 基础设备/类型配置
    private final long[] batchShape;    // 批量形状

    // 数值稳定性常量（仅浮点值）
    private static final float EPS = 1e-8f;
    private static final float MAX_VAL = 1e6f;
    private static final float MIN_VAL = -1e6f;

    // 构造函数：严格参数校验 + 深拷贝 + 批量形状计算
    public Gamma(Tensor concentration, Tensor rate) {
        // 1. 统一转换为Float32+CPU，避免类型/设备不匹配
//        System.out.println("Gamma concentration: " + concentration);
        this.concentration = concentration.to(kFloat())
                .to(torch.device(new Device(DeviceType.CPU)), false, true, new MemoryFormatOptional())
                .clone().detach();
//        System.out.println("Gamma concentration:1 " + concentration);
        this.rate = rate.to(kFloat())
                .to(torch.device(new Device(DeviceType.CPU)), false, true, new MemoryFormatOptional())
                .clone().detach();
        this.baseOptions = this.concentration.options();
//        System.out.println("Gamma concentration:2 " + concentration);
        this.batchShape = getBroadcastedShape(
                this.concentration.sizes().vec().get(),
                this.rate.sizes().vec().get()
        );
//        System.out.println("Gamma concentration3: " + concentration);
        // 2. 严格校验：α/β必须>0（检测≤0，包括0值）
        // 适配 clamp API：ScalarOptional 封装标量
        Scalar scalar0 = new Scalar(0.0f);
//        System.out.println("Gamma concentration:3.1 " + concentration);
        Tensor alphaLe0 = torch.le(this.concentration, torch.tensor(0.0f, baseOptions));
//        System.out.println("Gamma concentration:3.2 " + concentration);
        Tensor betaLe0 = torch.le(this.rate, torch.tensor(0.0f, baseOptions));
//        System.out.println("Gamma concentration:3.5 " + concentration);
        Tensor paramInvalid = torch.logical_or(alphaLe0, betaLe0);
//        System.out.println("Gamma concentration4 " + concentration);
        try {
            if (torch.any(paramInvalid).item().toBool()) {
                throw new IllegalArgumentException("伽马分布concentration(α)和rate(β)必须大于0！");
            }
        } finally {
            // 及时释放临时张量/Scalar
            safeClose(alphaLe0);
            safeClose(betaLe0);
            safeClose(paramInvalid);
            safeClose(scalar0);
        }
    }

    @Override
    public String name() {
        return "Gamma";
    }

    /**
     * 采样：使用LibTorch原生gamma函数，适配 clamp API
     * Gamma(α, β) = Gamma(α, 1) / β
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：处理采样形状（sampleShape + batchShape）
        long[] safeSampleShape = sampleShape == null || sampleShape.length == 0 ? new long[]{1} : sampleShape;
        long[] extendedShape = concatLongArrays(safeSampleShape, batchShape);

        // 步骤2：扩展参数到采样形状（兼容广播）
        Tensor expandedAlpha = expandToShape(concentration, extendedShape);
        Tensor expandedBeta = expandToShape(rate, extendedShape);

        // 步骤3：数值稳定化（适配原生 clamp API）
        // ScalarOptional 封装上下限，第二个参数为null表示无上限
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalarMAX = new Scalar(MAX_VAL);
        expandedAlpha = torch.clamp(expandedAlpha, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));
        expandedBeta = torch.clamp(expandedBeta, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));

        // 步骤4：采样标准伽马分布 Gamma(α, 1)（原生函数）
        Tensor gamma1Sample = GammaSampler.gamma(expandedAlpha, torch.ones_like(expandedAlpha));

        // 步骤5：转换为目标伽马分布 Gamma(α, β) = Gamma(α,1)/β
        Tensor gammaSample = torch.div(gamma1Sample, expandedBeta);
        // 数值稳定化：Gamma分布>0，适配 clamp API
        gammaSample = torch.clamp(gammaSample, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));

        // 步骤6：调整形状（移除多余维度）
        if (safeSampleShape.length == 1 && safeSampleShape[0] == 1 && sampleShape.length == 0) {
            gammaSample = gammaSample.squeeze(0);
        }

        // 释放临时张量/Scalar
        safeClose(expandedAlpha);
        safeClose(expandedBeta);
        safeClose(gamma1Sample);
        safeClose(scalarEPS);
        safeClose(scalarMAX);

        return gammaSample.clone().detach();
    }

    /**
     * 对数概率：严格对齐数学定义，适配 clamp API
     * 公式：log_prob(v) = α*log(β) + (α-1)*log(v) - β*v - lgamma(α)
     */
    @Override
    public Tensor log_prob(Tensor v) {
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 步骤1：统一输入类型/设备
        Tensor input = v.to(baseOptions, false, true,new MemoryFormatOptional()).clone().detach();
        long[] originalShape = input.shape();

        // 步骤2：严格校验输入v>0（检测≤0，包括0值）
        Scalar scalar0 = new Scalar(0.0f);
        Tensor vLe0 = torch.le(input, torch.tensor(0.0f, baseOptions));
        try {
            if (torch.any(vLe0).item().toBool()) {
                throw new IllegalArgumentException("log_prob输入v必须大于0！");
            }
        } finally {
            safeClose(vLe0);
            safeClose(scalar0);
        }

        // 步骤3：广播输入和参数到相同形状
        TensorVector broadcastTensors = torch.broadcast_tensors(new TensorVector(input, concentration, rate));
        Tensor inputBroadcast = broadcastTensors.get(0);
        Tensor alphaBroadcast = broadcastTensors.get(1);
        Tensor betaBroadcast = broadcastTensors.get(2);

        // 步骤4：数值稳定性处理（适配原生 clamp API）
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalarMAX = new Scalar(MAX_VAL);
        Tensor safeV = torch.clamp(inputBroadcast, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));
        Tensor safeAlpha = torch.clamp(alphaBroadcast, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));
        Tensor safeBeta = torch.clamp(betaBroadcast, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));

        // 步骤5：计算对数概率（严格拆分）
        // term1 = α * log(β)
        Tensor logBeta = torch.log(safeBeta);
        Tensor term1 = torch.mul(safeAlpha, logBeta);

        // term2 = (α-1) * log(v)
        Scalar scalar1 = new Scalar(1.0f);
        Tensor alphaMinus1 = torch.sub(safeAlpha, torch.tensor(1.0f, baseOptions));
        Tensor logV = torch.log(safeV);
        Tensor term2 = torch.mul(alphaMinus1, logV);

        // term3 = β * v
        Tensor term3 = torch.mul(safeBeta, safeV);

        // term4 = lgamma(α)
        Tensor term4 = torch.lgamma(safeAlpha);

        // 完整公式：term1 + term2 - term3 - term4
        Tensor logProb = torch.sub(
                torch.add(term1, term2),
                torch.add(term3, term4)
        );

        // 数值截断：适配 clamp API（限制上下限）
        Scalar scalarMIN = new Scalar(MIN_VAL);
        logProb = torch.clamp(logProb, new ScalarOptional(scalarMIN), new ScalarOptional(scalarMAX));
        // 恢复原始形状
        logProb = logProb.reshape(originalShape);

        // 释放所有临时张量/Scalar
        safeClose(input);
        safeClose(safeV);
        safeClose(safeAlpha);
        safeClose(safeBeta);
        safeClose(logBeta);
        safeClose(term1);
        safeClose(scalar1);
        safeClose(alphaMinus1);
        safeClose(logV);
        safeClose(term2);
        safeClose(term3);
        safeClose(term4);
        safeClose(scalarEPS);
        safeClose(scalarMAX);
        safeClose(scalarMIN);
//        for (Tensor t : broadcastTensors) {
//            safeClose(t);
//        }

        return logProb;
    }

    /**
     * 熵：严格对齐数学定义，适配 clamp API
     * 公式：H = α - log(β) + lgamma(α) + (1-α)*digamma(α)
     */
    @Override
    public Tensor entropy() {
        // 数值稳定化：适配原生 clamp API
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalarMAX = new Scalar(MAX_VAL);
        Tensor safeAlpha = torch.clamp(concentration, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));
        Tensor safeBeta = torch.clamp(rate, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));

        // 计算各项（严格拆分）
        Tensor term1 = safeAlpha.clone(); // α
        Tensor term2 = torch.log(safeBeta); // log(β)
        Tensor term3 = torch.lgamma(safeAlpha); // lgamma(α)

        Scalar scalar1 = new Scalar(1.0f);
        Tensor term4 = torch.mul(
                torch.sub(torch.tensor(1.0f, baseOptions), safeAlpha), // (1-α)
                torch.digamma(safeAlpha) // digamma(α)
        );

        // 完整公式：term1 - term2 + term3 + term4
        Tensor entropy = torch.add(
                torch.add(torch.sub(term1, term2), term3),
                term4
        );

        // 数值截断：适配 clamp API
        Scalar scalarMIN = new Scalar(MIN_VAL);
        entropy = torch.clamp(entropy, new ScalarOptional(scalarMIN), new ScalarOptional(scalarMAX));

        // 释放临时张量/Scalar
        safeClose(safeAlpha);
        safeClose(safeBeta);
        safeClose(term1);
        safeClose(term2);
        safeClose(term3);
        safeClose(term4);
        safeClose(scalarEPS);
        safeClose(scalarMAX);
        safeClose(scalarMIN);
        safeClose(scalar1);

        return entropy.clone().detach();
    }

    /**
     * 均值：严格对齐公式 α/β，适配 clamp API 做除零保护
     */
    @Override
    public Tensor mean() {
        // 除零保护：β小于EPS时替换为EPS（适配 clamp API）
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalarMAX = new Scalar(MAX_VAL);
        Tensor safeBeta = torch.clamp(rate, new ScalarOptional(scalarEPS), new ScalarOptional(scalarMAX));
        Tensor mean = torch.div(concentration.clone(), safeBeta);

        // 释放临时张量/Scalar
        safeClose(safeBeta);
        safeClose(scalarEPS);
        safeClose(scalarMAX);

        return mean;
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
                System.err.println("Gamma资源释放警告：" + e.getMessage());
            }
        }
    }

    // 资源释放：仅释放实例张量，避免内存泄漏
    @Override
    public void close() {
        safeClose(concentration);
        safeClose(rate);
    }

    // Getter方法（便于测试）
    public Tensor getConcentration() {
        return concentration.clone().detach();
    }

    public Tensor getRate() {
        return rate.clone().detach();
    }

    public long[] getBatchShape() {
        return Arrays.copyOf(batchShape, batchShape.length);
    }
}
