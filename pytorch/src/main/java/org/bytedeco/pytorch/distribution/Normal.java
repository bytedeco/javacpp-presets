package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Normal（正态/高斯）分布实现 - 修复数值容忍度问题
 */
public class Normal extends Distribution implements AutoCloseable {
    private final Tensor loc;                // 均值μ（形状：batch_shape）
    private final Tensor scale;              // 标准差σ（形状：batch_shape，必须σ>0）

    // 预定义张量（复用避免重复创建，保证设备/类型一致）
    private Tensor scalar05;
    private Tensor scalar2;
    private Tensor scalarLog2Pi;
    private Tensor scalar1e9; // 调整容忍度为1e-9，避免1e-8被误判

    /**
     * 构造函数：严格校验参数合法性 + 深拷贝
     * 核心修复：将scale的容忍度阈值从1e-8改为1e-9，避免1e-8的scale被误判为非法
     */
    public Normal(Tensor loc, Tensor scale) {
        // 1. 初始化预定义张量（与输入同设备/类型）
        this.scalar05 = torch.tensor(0.5, loc.options());
        this.scalar2 = torch.tensor(2.0, loc.options());
        this.scalarLog2Pi = torch.tensor(Math.log(2 * Math.PI), loc.options());
        this.scalar1e9 = torch.tensor(1e-9, loc.options()); // 关键修改：1e-9替代1e-8

        // 2. 校验scale>0（核心修复：判断scale < 1e-9才非法，而非≤1e-8）
        Tensor scaleLe0 = torch.lt(scale, scalar1e9); // lt = 小于（<），而非le（≤）
        boolean hasInvalidScale = torch.any(scaleLe0).item().toBool();
        if (hasInvalidScale) {
            scaleLe0.close();
            throw new IllegalArgumentException("scale(σ)必须大于0（数值容忍度1e-9）！");
        }

        // 3. 校验设备一致性
        if (!loc.device().equals(scale.device())) {
            scaleLe0.close();
            throw new IllegalArgumentException(
                    String.format("loc和scale设备不匹配：loc=%s, scale=%s",
                            loc.device().toString(), scale.device().toString())
            );
        }

        // 4. 校验形状可广播（保证批量运算合法）
        try {
            TensorVector broadcastCheck = new TensorVector(loc, scale);
            torch.broadcast_tensors(broadcastCheck);
            broadcastCheck.close();
        } catch (Exception e) {
            scaleLe0.close();
            throw new IllegalArgumentException("loc和scale形状无法广播：" + e.getMessage());
        }

        // 5. 深拷贝避免外部修改内部状态
        this.loc = loc.clone();
        this.scale = scale.clone();

        // 释放校验临时张量
        scaleLe0.close();
    }

    @Override
    public String name() {
        return "Normal";
    }

    /**
     * 正确的维度扩展逻辑：sampleShape + batch_shape
     */
    protected long[] getExtendedShape(Tensor baseTensor, long[] sampleShape) {
        long[] batchShape = baseTensor.sizes().vec().get();
        long[] extendedShape = new long[sampleShape.length + batchShape.length];

        // 拼接：sampleShape在前，batchShape在后
        System.arraycopy(sampleShape, 0, extendedShape, 0, sampleShape.length);
        System.arraycopy(batchShape, 0, extendedShape, sampleShape.length, batchShape.length);

        return extendedShape;
    }

    /**
     * 采样：基于标准正态分布的高效采样，支持任意批量采样形状
     * 公式：X = μ + σ * ε，ε ~ N(0, 1)
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展形状（sampleShape + batch_shape）
        long[] extendedShape = getExtendedShape(loc, sampleShape);

        // 步骤2：生成标准正态随机数ε（形状匹配扩展后）
        Tensor eps = randn(extendedShape, loc.options());

        // 步骤3：扩展loc和scale到采样形状（保证维度对齐）
        Tensor expandedLoc = loc.expand(extendedShape);
        Tensor expandedScale = scale.expand(extendedShape);

        // 步骤4：计算采样结果 X = μ + σ*ε
        Tensor sample = expandedLoc.add(eps.mul(expandedScale));

        // 释放临时张量（避免native内存泄漏）
        eps.close();
        expandedLoc.close();
        expandedScale.close();

        return sample;
    }

    /**
     * 对数概率：实现正态分布精确对数概率公式，增强数值稳定性
     * 公式：log p(X) = -logσ - 0.5*log(2π) - 0.5*((X-μ)/σ)^2
     */
    @Override
    public Tensor log_prob(Tensor value) {
        // 步骤1：广播loc/scale到value的形状（核心修复：维度对齐）
        TensorVector broadcastTensors = new TensorVector(loc, scale, value);
        Tensor[] broadcasted = torch.broadcast_tensors(broadcastTensors).get();
        broadcastTensors.close();

        Tensor expandedLoc = broadcasted[0].clone();
        Tensor expandedScale = broadcasted[1].clone();
        Tensor broadcastedValue = broadcasted[2].clone();

        // 释放broadcast_tensors返回的临时张量
        for (Tensor t : broadcasted) {
            t.close();
        }

        // 步骤2：数值稳定性处理（避免log(0)/除零）
        Tensor safeScale = torch.clamp(expandedScale, new TensorOptional(scalar1e9), new TensorOptional(expandedScale.max()));

        // 步骤3：计算各项
        Tensor logScale = torch.log(safeScale);
        Tensor term1 = logScale.neg(); // -logσ

        Tensor term2 = scalarLog2Pi.mul(scalar05).neg(); // -0.5*log(2π)

        Tensor diff = broadcastedValue.sub(expandedLoc);
        Tensor diffOverScale = diff.div(safeScale);
        Tensor squaredTerm = diffOverScale.pow(scalar2);
        Tensor term3 = squaredTerm.mul(scalar05).neg(); // -0.5*((X-μ)/σ)^2

        // 步骤4：完整对数概率 = term1 + term2 + term3
        Tensor logProb = term1.add(term2).add(term3);

        // 释放所有临时张量
        expandedLoc.close();
        expandedScale.close();
        broadcastedValue.close();
        safeScale.close();
        logScale.close();
        term1.close();
        term2.close();
        diff.close();
        diffOverScale.close();
        squaredTerm.close();
        term3.close();

        return logProb;
    }

    /**
     * 熵：实现正态分布的精确熵公式，规范张量运算
     * 公式：H = logσ + 0.5*(1 + log(2π))
     */
    @Override
    public Tensor entropy() {
        // 数值稳定性处理
        Tensor safeScale = torch.clamp(scale, new TensorOptional(scalar1e9), new TensorOptional(scale.max()));

        // 计算熵
        Tensor logScale = torch.log(safeScale);
        Tensor constTerm = scalar05.mul(torch.tensor(1.0, loc.options()).add(scalarLog2Pi));
        Tensor entropy = logScale.add(constTerm);

        // 释放临时张量
        safeScale.close();
        logScale.close();
        constTerm.close();

        return entropy;
    }

    /**
     * 均值：正态分布的均值等于loc(μ)
     */
    @Override
    public Tensor mean() {
        return loc.clone();
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     */
    @Override
    public void close() {
        loc.close();
        scale.close();
        scalar05.close();
        scalar2.close();
        scalarLog2Pi.close();
        scalar1e9.close();
    }

    // Getter方法（便于外部获取核心参数）
    public Tensor getLoc() { return loc.clone(); } // 返回拷贝避免外部修改
    public Tensor getScale() { return scale.clone(); }

    // 额外实用方法：获取方差（σ²）
    public Tensor variance() {
        return scale.pow(scalar2).clone();
    }
}
