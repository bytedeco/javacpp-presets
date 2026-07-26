package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Logistic（逻辑斯蒂）分布实现
 * 位置参数loc(μ)：分布的中心位置
 * 尺度参数scale(s)：分布的扩散程度（必须>0）
 * 修复点：
 * 1. 修复scale=0的校验逻辑（包含≤0的所有情况）
 * 2. 优化预定义Scalar的使用（避免重复释放）
 * 3. 增强数值稳定性处理
 */
public class Logistic extends Distribution implements AutoCloseable {
    private final Tensor loc;   // 位置参数μ
    private final Tensor scale; // 尺度参数s（必须>0）
    private boolean isClosed = false; // 防止重复释放

    // 预定义标量（静态常量，仅初始化一次，不随实例释放）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_2 = new Scalar(2.0f);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f);
    private static final Scalar SCALAR_1E6_NEG = new Scalar(1e-6f);
    private static final Scalar SCALAR_1E6_POS = new Scalar(1.0f - 1e-6f);
    private static final Scalar SCALAR_20 = new Scalar(20.0f);

    /**
     * 构造函数：校验参数合法性 + 深拷贝
     * @param l 位置参数loc(μ)
     * @param s 尺度参数scale(s)（必须>0）
     * @throws IllegalArgumentException 尺度参数≤0时抛出异常
     */
    public Logistic(Tensor l, Tensor s) {
        // 校验输入张量非空
        if (l == null || s == null) {
            throw new IllegalArgumentException("loc和scale参数不能为空！");
        }

        // 统一转换为Float32（避免类型不匹配）
        Tensor locCpu = l.to(new Device(DeviceType.CPU),kFloat()).clone().detach();
        Tensor scaleCpu = s.to(new Device(DeviceType.CPU),kFloat()).clone().detach();

        // 核心修复：校验scale ≤ 0（包含0和负数）
        Tensor scaleLe0 = torch.le(scaleCpu, torch.tensor(0.0f, scaleCpu.options()));
        try {
            if (torch.any(scaleLe0).item().toBool()) {
                throw new IllegalArgumentException("逻辑斯蒂分布scale(s)必须大于0！");
            }
        } finally {
            scaleLe0.close(); // 确保临时张量释放
        }

        // 数值保护：避免scale过小导致后续计算溢出
        Tensor safeScale = torch.clamp(scaleCpu, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 深拷贝避免外部修改内部状态
        this.loc = locCpu.clone().detach();
        this.scale = safeScale.clone().detach();

        // 释放临时张量
        locCpu.close();
        scaleCpu.close();
        safeScale.close();
    }

    @Override
    public String name() {
        return "Logistic";
    }

    /**
     * 采样：实现逻辑斯蒂分布的标准采样公式，增加数值稳定性处理
     * 公式：x = μ + s * log(U/(1-U))，U~Uniform(ε, 1-ε)
     * @param sampleShape 批量采样形状
     * @return 采样结果张量（形状：sampleShape + loc/scale的形状）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();
        // 步骤1：复用父类方法扩展采样形状
        long[] extendedShape = getExtendedShape(loc, sampleShape);
        // 扩展loc/scale到批量形状（保证维度对齐）
        Tensor expandedLoc = loc.expand(extendedShape).clone().detach();
        Tensor expandedScale = scale.expand(extendedShape).clone().detach();

        // 步骤2：生成受限Uniform(ε,1-ε)随机数（避免log(0)或log(1)）
        Tensor u = torch.rand(extendedShape, loc.options())
                .clamp(new ScalarOptional(SCALAR_1E6_NEG), new ScalarOptional(SCALAR_1E6_POS));

        // 步骤3：计算log(U/(1-U))，保证数值稳定性
        Tensor oneMinusU = torch.tensor(1.0f).sub( u); // 1-U
        // 数值稳定：避免1-U→0导致除零
        oneMinusU = torch.clamp(oneMinusU, new ScalarOptional(SCALAR_EPS), new ScalarOptional(SCALAR_1));

        Tensor logU = torch.log(u); // log(U)
        Tensor logOneMinusU = torch.log(oneMinusU); // log(1-U)
        Tensor logRatio = torch.sub(logU, logOneMinusU); // log(U/(1-U))

        // 步骤4：最终采样结果：μ + s * log(U/(1-U))
        Tensor scaleMulLogRatio = torch.mul(expandedScale, logRatio);
        Tensor logisticSample = torch.add(expandedLoc, scaleMulLogRatio);

        // 释放所有临时张量
        expandedLoc.close();
        expandedScale.close();
        u.close();
        oneMinusU.close();
        logU.close();
        logOneMinusU.close();
        logRatio.close();
        scaleMulLogRatio.close();

        return logisticSample.clone().detach();
    }

    /**
     * 对数概率：实现逻辑斯蒂分布的对数概率密度公式，增加数值稳定性
     * 公式：log(f(x)) = -z - log(s) - 2*log(1+e^(-z))，其中 z=(x-μ)/s
     * @param v 输入张量
     * @return 对数概率张量
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 步骤1：统一类型并扩展loc/scale到v的形状
        Tensor vCpu = v.to(loc.options(),false, true, new MemoryFormatOptional()).clone().detach();
        Tensor expandedLoc = loc.expand_as(vCpu).clone().detach();
        Tensor expandedScale = scale.expand_as(vCpu).clone().detach();

        // 数值稳定性：避免scale→0导致除零
        Tensor safeScale = torch.clamp(expandedScale, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));

        // 步骤2：计算z = (x-μ)/s（修正原公式符号错误）
        Tensor vMinusLoc = torch.sub(vCpu, expandedLoc); // x-μ
        Tensor z = torch.div(vMinusLoc, safeScale); // z=(x-μ)/s

        // 步骤3：计算对数概率各项（数值稳定版）
        Tensor logScale = torch.log(safeScale); // log(s)

        // 数值稳定版log(1+e^(-z))：避免exp(-z)溢出
        Tensor negZ = torch.neg(z); // -z
        Tensor clampedNegZ = torch.clamp(negZ, new ScalarOptional(new Scalar(Float.NEGATIVE_INFINITY)), new ScalarOptional(SCALAR_20));
        Tensor expNegZ = torch.exp(clampedNegZ); // e^(-z)
        Tensor onePlusExpNegZ = torch.add(expNegZ, SCALAR_1); // 1+e^(-z)
        Tensor logOnePlusExpNegZ = torch.log(onePlusExpNegZ);

        // 步骤4：完整对数概率公式（修正后）
        // log(f(x)) = -z - log(s) - 2*log(1+e^(-z))
        Tensor term1 = torch.neg(z); // -z
        Tensor term2 = torch.neg(logScale); // -log(s)
        Tensor term3 = torch.neg(torch.tensor(2.0f).mul(logOnePlusExpNegZ)); // -2*log(1+e^(-z))
        Tensor logProb = torch.add(torch.add(term1, term2), term3);

        // 释放所有临时张量
        vCpu.close();
        expandedLoc.close();
        expandedScale.close();
        safeScale.close();
        vMinusLoc.close();
        z.close();
        logScale.close();
        negZ.close();
        clampedNegZ.close();
        expNegZ.close();
        onePlusExpNegZ.close();
        logOnePlusExpNegZ.close();
        term1.close();
        term2.close();
        term3.close();

        return logProb.clone().detach();
    }

    /**
     * 熵：逻辑斯蒂分布的熵公式 H = log(s) + 2
     * @return 熵张量
     */
    @Override
    public Tensor entropy() {
        checkClosed();
        // 数值稳定性：避免scale→0导致log(0)
        Tensor safeScale = torch.clamp(scale, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Float.MAX_VALUE)));
        Tensor logScale = torch.log(safeScale);

        // 熵公式：log(s) + 2
        Tensor entropy = torch.add(logScale, SCALAR_2);

        // 释放临时张量
        safeScale.close();
        logScale.close();

        return entropy.clone().detach();
    }

    /**
     * 均值：逻辑斯蒂分布的均值等于位置参数μ
     * @return 均值张量（返回拷贝避免外部修改）
     */
    @Override
    public Tensor mean() {
        checkClosed();
        return loc.clone().detach();
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 检查实例是否已释放，避免重复使用
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("Logistic实例已释放，无法继续使用！");
        }
    }

    /**
     * 扩展采样形状（复用父类逻辑）
     */

    /**
     * 资源释放：实现AutoCloseable，避免内存泄漏
     */
    @Override
    public void close() {
        if (!isClosed) {
            loc.close();
            scale.close();
            isClosed = true;
        }
    }

    // Getter方法（便于外部获取核心参数）
    public Tensor getLoc() {
        checkClosed();
        return loc.clone().detach();
    }

    public Tensor getScale() {
        checkClosed();
        return scale.clone().detach();
    }
}
