package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Cauchy extends Distribution implements AutoCloseable {
    private final Tensor loc;  // 位置参数μ
    private final Tensor scale; // 尺度参数γ（必须>0）
    private final long[] batchShape; // 显式存储批量形状

    // 预定义常量（静态常量，不释放）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_2 = new Scalar(2.0);
    private static final Scalar SCALAR_4 = new Scalar(4.0);
    private static final Scalar SCALAR_PI = new Scalar(Math.PI);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8);
    private static final Scalar SCALAR_NAN = new Scalar(Float.NaN);

    // 构造函数：修复标量张量形状解析 + expand 维度适配
    public Cauchy(Tensor loc, Tensor scale) {
        // 1. 校验scale>0
        Tensor scaleLe0 = torch.lt(scale, torch.tensor(0.0f, scale.options()));
        if (torch.any(scaleLe0).item().toBool()) {
            scaleLe0.close();
            throw new IllegalArgumentException("柯西分布scale(γ)必须大于0！");
        }
        scaleLe0.close();

        // 2. 深拷贝参数
        this.loc = loc.clone();
        this.scale = scale.clone();

        // 3. 核心修复：解析形状时，先判断是否为标量（numel()=1 且 dim()=0）
        long[] locShape = getRealTensorShape(this.loc);
        long[] scaleShape = getRealTensorShape(this.scale);
        this.batchShape = computeBroadcastShape(locShape, scaleShape);
    }

    // 核心修复1：获取张量的真实形状（适配bytedeco标量张量）
    private long[] getRealTensorShape(Tensor tensor) {
        // bytedeco中，标量张量的dim()=0，即使sizes()返回[1]，也视为标量（形状为空数组）
        if (tensor.dim() == 0) {
            return new long[0]; // 标量的真实形状是空数组
        }
        // 一维张量且size=1，但dim()=1 → 视为标量（广播时按1处理）
        if (tensor.dim() == 1 && tensor.numel() == 1) {
            return new long[0];
        }
        // 其他情况返回原始形状
        return tensor.sizes().vec().get();
    }

    // 核心修复2：手动实现广播形状计算（适配标量空数组）
    private long[] computeBroadcastShape(long[] shape1, long[] shape2) {
        // 步骤1：反转形状（从后往前对齐）
        int len1 = shape1.length;
        int len2 = shape2.length;
        int maxLen = Math.max(len1, len2);

        long[] reversed1 = new long[maxLen];
        long[] reversed2 = new long[maxLen];

        // 填充反转数组（短的形状前面补1）
        for (int i = 0; i < maxLen; i++) {
            reversed1[i] = (i < len1) ? shape1[len1 - 1 - i] : 1;
            reversed2[i] = (i < len2) ? shape2[len2 - 1 - i] : 1;
        }

        // 步骤2：逐维度计算广播形状
        long[] reversedResult = new long[maxLen];
        for (int i = 0; i < maxLen; i++) {
            long s1 = reversed1[i];
            long s2 = reversed2[i];
            if (s1 == 1) {
                reversedResult[i] = s2;
            } else if (s2 == 1) {
                reversedResult[i] = s1;
            } else if (s1 == s2) {
                reversedResult[i] = s1;
            } else {
                throw new IllegalArgumentException("形状无法广播：" + arrayToString(shape1) + " 和 " + arrayToString(shape2));
            }
        }

        // 步骤3：反转回原顺序
        long[] result = new long[maxLen];
        for (int i = 0; i < maxLen; i++) {
            result[i] = reversedResult[maxLen - 1 - i];
        }

        return result;
    }

    // 核心修复3：适配expand的维度规则（空数组→[1]）
    private long[] adaptExpandShape(long[] targetShape) {
        // bytedeco中，一维张量（dim=1）不能expand到空数组，需适配为[1]
        if (targetShape.length == 0) {
            return new long[]{1};
        }
        return targetShape;
    }

    @Override
    public String name() {
        return "Cauchy";
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // sampleShape + batchShape (scalar batch stays empty → sample(4) has shape [4])
        long[] extendedShape = new long[sampleShape.length + batchShape.length];
        System.arraycopy(sampleShape, 0, extendedShape, 0, sampleShape.length);
        System.arraycopy(batchShape, 0, extendedShape, sampleShape.length, batchShape.length);
        if (extendedShape.length == 0) {
            extendedShape = new long[]{1};
        }

        // 生成Uniform(0,1)随机数（数值稳定性）
        Tensor u = torch.rand(extendedShape, loc.options())
                .clamp(new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(1.0 - 1e-8)));

        // 采样公式：x = μ + γ * tan(π*(u-0.5))
        Tensor piTensor = torch.tensor(Math.PI, u.options());
        Tensor uMinus05 = torch.sub(u, torch.tensor(0.5f, u.options()));
        Tensor tanTerm = torch.tan(torch.mul(uMinus05, piTensor));

        // 扩展loc/scale到最终形状（适配维度）
        Tensor expandedLoc = loc.expand(extendedShape);
        Tensor expandedScale = scale.expand(extendedShape);
        Tensor sample = torch.add(expandedLoc, torch.mul(expandedScale, tanTerm));

        // 释放临时张量
        u.close();
        piTensor.close();
        uMinus05.close();
        tanTerm.close();
        expandedLoc.close();
        expandedScale.close();

        return sample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        // Broadcast loc/scale/v to a common shape (sample shape + batch shape)
        TensorVector tv = new TensorVector(loc, scale, v);
        Tensor[] b = torch.broadcast_tensors(tv).get();
        tv.close();
        Tensor expandedLoc = b[0];
        Tensor expandedScale = b[1];
        Tensor expandedV = b[2];

        // z = (v - μ)/γ
        Tensor vMinusLoc = torch.sub(expandedV, expandedLoc);
        Tensor z = torch.div(vMinusLoc, expandedScale);

        Tensor zClamped = z.clamp(
                new ScalarOptional(new Scalar(-1e6)),
                new ScalarOptional(new Scalar(1e6))
        );

        // log p = -log(πγ) - log(1+z²)
        Tensor piTensor = torch.tensor(Math.PI, expandedScale.options());
        Tensor piScale = torch.mul(expandedScale, piTensor);
        Tensor logPiScale = torch.log(piScale);

        Tensor zSquared = torch.pow(zClamped, torch.tensor(2.0f, zClamped.options()));
        Tensor onePlusZSquared = torch.add(zSquared, torch.tensor(1.0f, zSquared.options()));
        Tensor logOnePlusZSquared = torch.log(onePlusZSquared);

        Tensor logProb = torch.sub(torch.neg(logPiScale), logOnePlusZSquared);

        expandedLoc.close();
        expandedScale.close();
        expandedV.close();
        vMinusLoc.close();
        z.close();
        zClamped.close();
        piTensor.close();
        piScale.close();
        logPiScale.close();
        zSquared.close();
        onePlusZSquared.close();
        logOnePlusZSquared.close();

        return logProb;
    }

    @Override
    public Tensor entropy() {
        // 适配expand形状（空数组→[1]）
        long[] adaptedBatchShape = adaptExpandShape(batchShape);

        // 扩展scale到批量形状
        Tensor expandedScale = scale.expand(adaptedBatchShape);

        // 熵公式：log(4πγ)
        Tensor fourTensor = torch.tensor(4.0f, expandedScale.options());
        Tensor piTensor = torch.tensor(Math.PI, expandedScale.options());
        Tensor fourPiScale = torch.mul(torch.mul(fourTensor, piTensor), expandedScale);
        Tensor entropy = torch.log(fourPiScale);

        // 释放临时张量
        expandedScale.close();
        fourTensor.close();
        piTensor.close();
        fourPiScale.close();

        return entropy;
    }

    @Override
    public Tensor mean() {
        // 适配mean的形状（空数组→[1]）
        long[] adaptedBatchShape = adaptExpandShape(batchShape);
        return torch.full(adaptedBatchShape, SCALAR_NAN, loc.options());
    }

    // 仅释放实例变量
    @Override
    public void close() {
        loc.close();
        scale.close();
    }

    // 暴露批量形状（供测试验证）
    public long[] getBatchShape() {
        return batchShape;
    }

    // 辅助方法：数组转字符串
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
}
