package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Pareto（帕累托）分布实现
 * 优化点：
 * 1. 修复资源释放逻辑（预定义Scalar不能在close()中释放）
 * 2. 规范TensorOptions/Device初始化（匹配PyTorch Java API）
 * 3. 优化数值稳定性（避免不必要的clamp）
 * 4. 修复batchDim计算错误（单批次应为0，批量为对应维度数）
 * 5. 增加空指针校验、重复释放保护
 */
public class Pareto extends Distribution implements AutoCloseable {
    private final Tensor scale;              // 尺度参数x_m（>0，batch_shape）
    private final Tensor alpha;              // 形状参数α（>0，batch_shape）
    private final long[] batchShape;         // 批量形状（更精准）
    private final int batchDim;              // 批量维度数
    private boolean isClosed = false;        // 防止重复释放

    // 预定义标量（全局复用，仅在类卸载时释放）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f);
    private static final Scalar SCALAR_INF = new Scalar(Float.POSITIVE_INFINITY);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Float.NEGATIVE_INFINITY);
    private static final Device CPU_DEVICE = new Device(torch.kCPU());

    /**
     * 构造函数：严格校验参数合法性 + 深拷贝
     * @param scale 尺度参数x_m（必须>0）
     * @param alpha 形状参数α（必须>0）
     * @throws IllegalArgumentException 参数非法/设备/形状不匹配抛出异常
     */
    public Pareto(Tensor scale, Tensor alpha) {
        // 1. 空指针校验
        if (scale == null || alpha == null) {
            throw new IllegalArgumentException("scale和alpha参数不能为空！");
        }

        // 2. 统一转换为CPU/Float（避免设备/类型不一致）
        Tensor scaleFloatCPU = scale.to(CPU_DEVICE, torch.kFloat()).clone().detach();
        Tensor alphaFloatCPU = alpha.to(CPU_DEVICE, torch.kFloat()).clone().detach();

        // 3. 校验参数>0（添加数值容忍度，避免浮点误差）
        Tensor scaleLeEps = torch.le(scaleFloatCPU, torch.tensor(1e-8, scaleFloatCPU.options()));
        Tensor alphaLeEps = torch.le(alphaFloatCPU, torch.tensor(1e-8, alphaFloatCPU.options()));

        try {
            if (torch.any(scaleLeEps).item().toBool()) {
                throw new IllegalArgumentException("scale(x_m)必须大于0（数值容忍度1e-8）！");
            }
            if (torch.any(alphaLeEps).item().toBool()) {
                throw new IllegalArgumentException("alpha(α)必须大于0（数值容忍度1e-8）！");
            }

            // 4. 校验形状可广播（保证批量运算合法）
            try {
                torch.broadcast_tensors(new TensorVector(scaleFloatCPU, alphaFloatCPU));
            } catch (Exception e) {
                throw new IllegalArgumentException("scale和alpha形状无法广播：" + e.getMessage());
            }

            // 5. 初始化核心参数
            this.scale = scaleFloatCPU;
            this.alpha = alphaFloatCPU;
            this.batchShape = getTensorShape(this.scale);
            this.batchDim = (int) this.scale.dim();

        } finally {
            // 释放校验临时张量
            scaleLeEps.close();
            alphaLeEps.close();
        }
    }

    @Override
    public String name() {
        return "Pareto";
    }

    /**
     * 采样：基于均匀分布的精确采样，支持任意批量采样形状
     * 公式：X = x_m / U^(1/α)，U ~ Uniform(0,1)
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();
        // 步骤1：扩展形状（sampleShape + batch_shape）
        long[] extendedShape = concatShapes(sampleShape, batchShape);

        // 步骤2：生成均匀分布随机数U（0,1），避免U=0导致除零
        TensorOptions randOptions = torch.tensor(0.0f).options()
                .dtype(new ScalarTypeOptional(torch.kFloat()))
                .device(new DeviceOptional(CPU_DEVICE));
        Tensor u = torch.rand(extendedShape, randOptions);

        // 安全clamp：避免U=0/1
        ScalarOptional uMin = new ScalarOptional(SCALAR_EPS);
        ScalarOptional uMax = new ScalarOptional(new Scalar(1.0f - 1e-8f));
        Tensor uSafe = torch.clamp(u, uMin, uMax);

        // 步骤3：扩展参数到采样形状（保证维度对齐）
        Tensor expandedScale = scale.expand(extendedShape);
        Tensor expandedAlpha = alpha.expand(extendedShape);

        // 步骤4：计算采样结果 X = x_m / U^(1/α)
        Tensor invAlpha = torch.reciprocal(expandedAlpha);
        Tensor uPow = torch.pow(uSafe, invAlpha);
        Tensor sample = torch.div(expandedScale, uPow);

        // 释放临时张量
        randOptions.close();
        u.close();
        uMin.close();
        uMax.close();
        uSafe.close();
        expandedScale.close();
        expandedAlpha.close();
        invAlpha.close();
        uPow.close();

        return sample;
    }

    /**
     * 对数概率：实现帕累托分布精确对数概率公式
     * 公式：log f(x) = logα + αlogx_m - (α+1)logx（x≥x_m），否则返回-∞
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 统一转换为CPU/Float
        Tensor vFloatCPU = v.to(CPU_DEVICE, torch.kFloat()).clone().detach();
        long[] vShape = getTensorShape(vFloatCPU);
        long[] broadcastShape = broadcastShapes(batchShape, vShape);

        // 扩展参数到输入形状
        Tensor expandedScale = scale.expand(broadcastShape);
        Tensor expandedAlpha = alpha.expand(broadcastShape);
        Tensor vExpanded = vFloatCPU.expand(broadcastShape);

        // 输入合法性校验（x ≥ x_m，添加微小容忍度避免浮点误差）
        Tensor vGeScale = torch.ge(vExpanded, torch.sub(expandedScale, torch.tensor(1e-10f)));

        // 数值稳定性处理（仅避免log(0)）
        Tensor safeV = torch.where(vGeScale, vExpanded, torch.add(expandedScale, torch.tensor(1e-10f)));

        // 计算对数概率各项
        Tensor logAlpha = torch.log(expandedAlpha);
        Tensor logScale = torch.log(expandedScale);
        Tensor alphaLogScale = torch.mul(expandedAlpha, logScale);
        Tensor alphaPlus1 = torch.add(expandedAlpha, torch.tensor(1.0f));
        Tensor logV = torch.log(safeV);
        Tensor negAlphaPlus1LogV = torch.neg(torch.mul(alphaPlus1, logV));

        // 完整对数概率（合法输入）
        Tensor logProbValid = torch.add(torch.add(logAlpha, alphaLogScale), negAlphaPlus1LogV);

        // 处理非法输入（x < x_m → 返回-∞）
        Tensor logProb = torch.where(
                vGeScale,
                logProbValid,
                torch.full_like(logProbValid, SCALAR_NEG_INF, logProbValid.options(), new MemoryFormatOptional())
        );

        // 释放临时张量
        vFloatCPU.close();
        expandedScale.close();
        expandedAlpha.close();
        vExpanded.close();
        vGeScale.close();
        safeV.close();
        logAlpha.close();
        logScale.close();
        alphaLogScale.close();
        alphaPlus1.close();
        logV.close();
        negAlphaPlus1LogV.close();
        logProbValid.close();

        return logProb;
    }

    /**
     * 均值：帕累托分布的精确均值公式
     * 公式：E[X] = αx_m/(α-1)（α>1），否则为+∞
     */
    @Override
    public Tensor mean() {
        checkClosed();
        // 步骤1：判断α>1（添加数值容忍度避免浮点误差）
        Tensor alphaGt1 = torch.gt(alpha, torch.tensor(1.0f + 1e-8f));

        // 步骤2：计算α>1时的均值
        Tensor alphaMinus1 = torch.sub(alpha, torch.tensor(1.0f));
        Tensor meanValid = torch.div(torch.mul(alpha, scale), alphaMinus1);

        // 步骤3：处理α≤1的情况（返回+∞）
        Tensor mean = torch.where(
                alphaGt1,
                meanValid,
                torch.full_like(meanValid, SCALAR_INF, meanValid.options(), new MemoryFormatOptional())
        );

        // 释放临时张量
        alphaGt1.close();
        alphaMinus1.close();
        meanValid.close();

        return mean.clone().detach();
    }

    /**
     * 熵：实现帕累托分布的精确熵公式
     * 公式：H = log(x_m/α) + 1/α + 1
     */
    @Override
    public Tensor entropy() {
        checkClosed();
        // 计算各项
        Tensor scaleOverAlpha = torch.div(scale, alpha);
        Tensor logScaleOverAlpha = torch.log(scaleOverAlpha);
        Tensor invAlpha = torch.reciprocal(alpha);
        Tensor one = torch.tensor(1.0f).expand(batchShape).to(CPU_DEVICE, torch.kFloat());

        // 完整熵公式
        Tensor entropy = torch.add(torch.add(logScaleOverAlpha, invAlpha), one);

        // 释放临时张量
        scaleOverAlpha.close();
        logScaleOverAlpha.close();
        invAlpha.close();
        one.close();

        return entropy;
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     * 注意：预定义Scalar是静态全局的，不能在实例close中释放！
     */
    @Override
    public void close() {
        if (!isClosed) {
            scale.close();
            alpha.close();
            isClosed = true;
        }
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 检查是否已释放，避免访问已释放资源
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("Pareto实例已释放，无法继续使用！");
        }
    }

    /**
     * 获取张量形状（long[]）
     */
    private long[] getTensorShape(Tensor tensor) {
        long[] shape = new long[(int) tensor.dim()];
        for (int i = 0; i < tensor.dim(); i++) {
            shape[i] = tensor.size(i);
        }
        return shape;
    }

    /**
     * 拼接形状（sampleShape + batchShape）
     */
    private long[] concatShapes(long[] sampleShape, long[] batchShape) {
        long[] result = new long[sampleShape.length + batchShape.length];
        System.arraycopy(sampleShape, 0, result, 0, sampleShape.length);
        System.arraycopy(batchShape, 0, result, sampleShape.length, batchShape.length);
        return result;
    }

    /**
     * 计算广播后的形状
     */
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
                throw new IllegalArgumentException("形状无法广播：" + arrayToString(a) + " 和 " + arrayToString(b));
            }
            result[i] = Math.max(sA, sB);
        }
        return result;
    }

    /**
     * 数组转字符串
     */
    private String arrayToString(long[] array) {
        if (array == null || array.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i]);
            if (i < array.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    // Getter方法（返回拷贝，避免外部修改内部张量）
    public Tensor getScale() { checkClosed(); return scale.clone().detach(); }
    public Tensor getAlpha() { checkClosed(); return alpha.clone().detach(); }
    public int getBatchDim() { return batchDim; }
    public long[] getBatchShape() { return batchShape.clone(); }

    // 额外实用方法：获取众数（帕累托分布众数为x_m）
    public Tensor mode() { checkClosed(); return scale.clone().detach(); }

    /**
     * 静态资源释放（仅在程序退出时调用）
     */
    public static void releaseStaticResources() {
        SCALAR_0.close();
        SCALAR_1.close();
        SCALAR_EPS.close();
        SCALAR_INF.close();
        SCALAR_NEG_INF.close();
    }
}
