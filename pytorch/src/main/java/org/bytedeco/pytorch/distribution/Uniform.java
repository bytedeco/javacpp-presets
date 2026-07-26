package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Uniform（均匀分布）实现
 * low(a)：下界（形状：batch_shape）
 * high(b)：上界（形状：batch_shape，必须满足b > a）
 * 支持批量参数、批量采样，具备完整的合法性校验和数值稳定性
 */
public class Uniform extends Distribution implements AutoCloseable {
    private final Tensor low;                // 下界a
    private final Tensor high;               // 上界b
    private final Tensor range;              // 预计算b-a，提升效率
    private final Tensor logRange;           // 预计算log(b-a)，提升效率

    // 预定义标量（复用避免重复创建，提升性能+规范）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值容忍度
    private static final Scalar SCALAR_NEG_INF = new Scalar(Double.NEGATIVE_INFINITY);
    private static final LongOptional DIM_NEG_1 = new LongOptional(-1);

    /**
     * 构造函数：严格校验参数合法性 + 预计算关键值 + 深拷贝
     * @param low 下界a（必须 < high）
     * @param high 上界b（必须 > low）
     * @throws IllegalArgumentException 参数非法/设备不匹配抛出异常
     */
    public Uniform(Tensor low, Tensor high) {
        // 1. 空值校验
        if (low == null || high == null) {
            throw new IllegalArgumentException("low和high参数不能为空！");
        }

        // 2. 校验设备一致性
        if (!low.device().equals(high.device())) {
            throw new IllegalArgumentException(
                    String.format("low和high设备不匹配：low=%s, high=%s",
                            low.device().toString(), high.device().toString())
            );
        }

        // 3. 校验形状可广播
        try {
            torch.broadcast_tensors(new TensorVector(low, high));
        } catch (Exception e) {
            throw new IllegalArgumentException("low和high形状无法广播：" + e.getMessage());
        }

        // 4. 校验high > low（添加数值容忍度，避免浮点误差）
        Tensor rangeRaw = high.sub(low);
        Tensor rangeLe0 = torch.le(rangeRaw, torch.tensor(1e-8, low.options()));
        if (torch.any(rangeLe0).item().toBool()) {
            rangeRaw.close();
            rangeLe0.close();
            throw new IllegalArgumentException("high必须严格大于low（数值容忍度1e-8）！");
        }

        // 5. 初始化核心参数（深拷贝避免外部修改）
        this.low = low.clone();
        this.high = high.clone();

        // 6. 预计算关键值（数值稳定处理）
        this.range = torch.clamp(rangeRaw, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e10)));
        this.logRange = torch.log(this.range);

        // 释放校验临时张量
        rangeRaw.close();
        rangeLe0.close();
    }

    @Override
    public String name() {
        return "Uniform";
    }

    /**
     * 采样：实现均匀分布的精确采样，支持任意批量采样形状
     * 公式：X = a + (b-a) * U，U~Uniform(0,1)
     * @param sampleShape 批量采样形状
     * @return 采样结果张量（形状：sampleShape + batch_shape）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展形状（sampleShape + batch_shape）
        long[] extendedShape = getExtendedShape(low, sampleShape);

        // 步骤2：扩展参数到采样形状
        Tensor expandedLow = low.expand(extendedShape);
        Tensor expandedRange = range.expand(extendedShape);

        // 步骤3：生成[0,1)均匀分布U
        Tensor u = torch.rand(extendedShape, low.options());

        // 步骤4：计算采样值 X = a + (b-a)*U
        Tensor sample = torch.add(expandedLow, torch.mul(expandedRange, u));

        // 释放临时张量
        expandedLow.close();
        expandedRange.close();
        u.close();

        return sample;
    }

    /**
     * 对数概率：实现均匀分布的精确对数概率公式，校验输入值域
     * 公式：logP(x) = -log(b-a)（a≤x≤b），否则返回-∞
     * @param v 输入张量（形状需与参数可广播）
     * @return 对数概率张量（形状：batch_shape）
     */
    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：扩展参数到输入形状
        Tensor expandedLow = low.expand(v.sizes());
        Tensor expandedHigh = high.expand(v.sizes());
        Tensor expandedLogRange = logRange.expand(v.sizes());

        // 步骤2：校验输入值域（a ≤ x ≤ b）
        Tensor xGeLow = torch.ge(v, expandedLow);
        Tensor xLeHigh = torch.le(v, expandedHigh);
        Tensor inRange = torch.logical_and(xGeLow, xLeHigh);

        // 步骤3：计算合法输入的对数概率（-log(b-a)）
        Tensor logProbValid = torch.neg(expandedLogRange);

        // 步骤4：处理非法输入（返回-∞）
        Tensor logProb = torch.where(
                inRange,
                logProbValid,
                torch.full_like(logProbValid, new Scalar(Float.NEGATIVE_INFINITY), logProbValid.options(),new MemoryFormatOptional())
        );

        // 释放所有临时张量
        expandedLow.close();
        expandedHigh.close();
        expandedLogRange.close();
        xGeLow.close();
        xLeHigh.close();
        inRange.close();
        logProbValid.close();

        return logProb;
    }

    /**
     * 均值：均匀分布的均值 = (a + b)/2
     * @return 均值张量（返回拷贝避免外部修改）
     */
    @Override
    public Tensor mean() {
        // 规范的张量运算（替代原代码标量直接运算）
        Tensor sum = torch.add(low, high);
        Tensor mean = torch.mul(sum, torch.tensor(0.5f, low.options()));
        sum.close();
        return mean;
    }

    /**
     * 熵：均匀分布的熵 = log(b-a)
     * @return 熵张量（返回拷贝避免外部修改）
     */
    @Override
    public Tensor entropy() {
        return logRange.clone();
    }

    /**
     * 额外实用方法：计算均匀分布的方差
     * 公式：Var(X) = (b-a)² / 12
     * @return 方差张量
     */
    public Tensor variance() {
        Tensor rangeSq = torch.pow(range, torch.tensor(1.0f).mul(new Scalar(1.0f))); // (b-a)²
        Tensor var = torch.div(rangeSq, torch.tensor(12.0, low.options()));
        rangeSq.close();
        return var;
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     */
    @Override
    public void close() {
        low.close();
        high.close();
        range.close();
        logRange.close();
        // 释放预定义常量

    }

    // Getter方法（提升易用性）
    public Tensor getLow() { return low; }
    public Tensor getHigh() { return high; }
    public Tensor getRange() { return range; }
    public Tensor getLogRange() { return logRange; }
}
