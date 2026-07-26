package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.internal.GammaSampler;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Chi2（卡方分布）稳定版：
 * 1. 修复异常参数检测（df=0/x=0）
 * 2. 修复采样算法错误（改用原生gamma采样，均值回归理论值）
 * 3. 优化数值稳定性和资源管理
 * 4. 完善批量参数支持
 */
public class Chi2 extends Distribution implements AutoCloseable {
    private final Tensor df;          // 自由度（必须>0）
    private final TensorOptions baseOptions; // 基础设备/类型配置
    private final long[] batchShape;  // 批量形状

    // 预定义标量（static全局复用，避免重复创建）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_2 = new Scalar(2.0f);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f); // 数值稳定性极小值
    private static final Scalar SCALAR_NEG_INF = new Scalar(Float.NEGATIVE_INFINITY);

    // Optional包装器（避免重复创建）
    private static final ScalarOptional OPTIONAL_EPS = new ScalarOptional(SCALAR_EPS);
    private static final ScalarOptional OPTIONAL_1_MINUS_EPS = new ScalarOptional(new Scalar(1.0f - 1e-8f));

    // 构造函数：严格参数校验 + 深拷贝 + 批量形状计算
    public Chi2(Tensor df) {
        // 1. 统一转换为Float32+CPU，避免类型/设备不匹配
        this.df = df.to(kFloat()).to(torch.device(new Device(DeviceType.CPU)), false, true, new MemoryFormatOptional()).clone().detach();
        this.baseOptions = this.df.options();
        this.batchShape = this.df.sizes().vec().get();

        // 2. 严格校验：df必须>0（包括检测0值）
        Tensor dfLe0 = torch.le(this.df, SCALAR_0);
        try {
            if (torch.any(dfLe0).item().toBool()) {
                throw new IllegalArgumentException("卡方分布自由度df必须大于0！");
            }
        } finally {
            safeClose(dfLe0); // 及时释放临时张量
        }

//        System.out.println("chi2 construct.df shape..." + Arrays.toString(this.batchShape));
    }

    @Override
    public String name() {
        return "Chi2";
    }

    /**
     * 采样：修复算法错误，改用原生gamma采样（保证均值正确）
     * 核心公式：Chi2(df) = 2 * Gamma(df/2, 1) = Gamma(df/2, 2)
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：处理采样形状（sampleShape + batchShape）

        long[] safeSampleShape = sampleShape == null || sampleShape.length == 0 ? new long[]{1} : sampleShape;
        long[] extendedShape = concatLongArrays(safeSampleShape, batchShape);
        // 步骤2：计算Gamma分布形状参数alpha=df/2，并扩展到采样形状
        Tensor alpha = torch.mul(df, SCALAR_0_5);
        Tensor expandedAlpha = expandToShape(alpha, extendedShape);
  
        // 步骤3：Gamma(alpha, 1)采样（改用原生gamma函数，修复算法错误）
        Tensor gamma1Sample = GammaSampler.gamma(expandedAlpha,torch.ones_like(expandedAlpha));
//        Tensor gamma1Sample = torch.gamma(expandedAlpha, baseOptions);

        // 步骤4：Chi2采样 = 2 * Gamma(alpha, 1)
        Tensor chi2Sample = torch.mul(gamma1Sample, SCALAR_2);
        // 数值稳定化：避免极小值（卡方分布>0）
        chi2Sample = torch.clamp(chi2Sample, new ScalarOptional(SCALAR_EPS), new ScalarOptional());

        // 步骤5：调整形状（移除多余维度）
        if (safeSampleShape.length == 1 && safeSampleShape[0] == 1 && sampleShape.length == 0) {
            chi2Sample = chi2Sample.squeeze(0);
        }

        // 释放临时张量
        safeClose(alpha);
        safeClose(expandedAlpha);
        safeClose(gamma1Sample);

        return chi2Sample.clone().detach();
    }

    /**
     * 对数概率：修复x=0检测，增强数值稳定性
     */
    @Override
    public Tensor log_prob(Tensor v) {
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 步骤1：统一输入类型/设备
        Tensor input = v.to(baseOptions,false, true,new MemoryFormatOptional()).clone().detach();
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

        // 步骤3：广播输入和df到相同形状
        TensorVector broadcastTensors = torch.broadcast_tensors(new TensorVector(input, df));
        Tensor inputBroadcast = broadcastTensors.get(0);
        Tensor dfBroadcast = broadcastTensors.get(1);

        // 步骤4：数值稳定性处理（避免log(0)）
        Tensor safeV = torch.clamp(inputBroadcast, new ScalarOptional(SCALAR_EPS), new ScalarOptional());

        // 步骤5：计算对数概率公式
        Tensor halfDf = torch.mul(dfBroadcast, SCALAR_0_5);
        Tensor log2 = torch.log(torch.tensor(2.0f));

        // term1 = (df/2 - 1) * log(v)
        Tensor term1 = torch.mul(torch.sub(halfDf, SCALAR_1), torch.log(safeV));
        // term2 = v/2
        Tensor term2 = torch.mul(safeV, SCALAR_0_5);
        // term3 = (df/2) * log(2)
        Tensor term3 = torch.mul(halfDf, log2);
        // term4 = lgamma(df/2)
        Tensor term4 = torch.lgamma(halfDf);

        // 完整对数概率：term1 - term2 - term3 - term4
        Tensor logProb = torch.sub(torch.sub(torch.sub(term1, term2), term3), term4);

        // 步骤6：恢复原始形状
        logProb = logProb.reshape(originalShape);

        // 释放所有临时张量
        safeClose(input);
        safeClose(safeV);
        safeClose(halfDf);
        safeClose(log2);
        safeClose(term1);
        safeClose(term2);
        safeClose(term3);
        safeClose(term4);
//        for (Tensor t : broadcastTensors) {
//            safeClose(t);
//        }

        return logProb;
    }

    /**
     * 熵：增强数值稳定性
     */
    @Override
    public Tensor entropy() {
        // 熵公式：df/2 + log(2) + lgamma(df/2) + (1 - df/2)*digamma(df/2)
        Tensor halfDf = torch.mul(df, SCALAR_0_5);
        Tensor log2 = torch.log(torch.tensor(2.0f));
        Tensor oneTensor = torch.tensor(1.0f, baseOptions);

        // 数值稳定化：避免gamma函数溢出
        halfDf = torch.clamp(halfDf, new ScalarOptional(new Scalar(SCALAR_EPS)), new ScalarOptional(new Scalar(1e6f)));

        // 逐项计算
        Tensor term1 = halfDf; // df/2
        Tensor term2 = log2;   // log(2)
        Tensor term3 = torch.lgamma(halfDf); // lgamma(df/2)
        Tensor term4 = torch.mul(torch.sub(oneTensor, halfDf), torch.digamma(halfDf)); // (1-df/2)*digamma(df/2)

        // 完整熵 + 数值截断
        Tensor entropy = torch.add(torch.add(torch.add(term1, term2), term3), term4);
        entropy = torch.clamp(entropy, new ScalarOptional(new Scalar(-1e6f)), new ScalarOptional(new Scalar(1e6f)));

        // 释放临时张量
        safeClose(halfDf);
        safeClose(log2);
        safeClose(oneTensor);
        safeClose(term1);
        safeClose(term2);
        safeClose(term3);
        safeClose(term4);

        return entropy.clone().detach();
    }

    @Override
    public Tensor mean() {
        // 卡方分布均值=df（返回拷贝避免外部修改）
        return df.clone().detach();
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
                System.err.println("Chi2资源释放警告：" + e.getMessage());
            }
        }
    }

    // 资源释放：仅释放实例张量，static标量全局复用不释放
    @Override
    public void close() {
        safeClose(df);
    }

    // Getter方法
    public Tensor getDf() {
        return df.clone().detach();
    }

    public long[] getBatchShape() {
        return Arrays.copyOf(batchShape, batchShape.length);
    }
}
