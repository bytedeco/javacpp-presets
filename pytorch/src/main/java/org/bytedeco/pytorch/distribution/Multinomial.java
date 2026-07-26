package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Multinomial {
    private long totalCount;       // 试验次数（必须是非负整数）
    private Tensor probs;          // 类别概率分布
    private Tensor logProbs;       // 对数概率分布
    private boolean probsValid;    // 概率是否已归一化

    // 构造函数
    public Multinomial(long totalCount, Tensor probs) {
        // 1. 完善参数校验（修复异常测试2失败）
        // 校验totalCount是否为非负整数（核心修复点1）
        if (totalCount < 0) {
            throw new IllegalArgumentException("total_count必须是非负整数！");
        }
        // 校验totalCount是否为整数（原缺失的校验）
        if (totalCount != Math.floor(totalCount)) {
            throw new IllegalArgumentException("total_count必须是整数！");
        }

        // 校验probs非空且元素非负
        if (probs == null || probs.numel() == 0) {
            throw new IllegalArgumentException("probs不能为空！");
        }
        if (any(probs.lt(new Scalar(0))).item().toBool()) {
            throw new IllegalArgumentException("probs所有元素必须非负！");
        }

        this.totalCount = totalCount;

        // 归一化概率（处理全零情况，避免NaN）
        Tensor probsSum = probs.sum();
        if (probsSum.item().toDouble() == 0) {
            this.probs = ones_like(probs).div(new Scalar(probs.numel()));
        } else {
            this.probs = probs.div(probsSum);
        }

        this.logProbs = log(this.probs);
        this.probsValid = true;
    }

    // 采样方法（核心修复点2：修复Tensor形状不匹配）
    public Tensor sample(long[] sampleShape) {
        try {
            // 步骤1：处理采样形状默认值
            long[] finalShape;
            if (sampleShape == null || sampleShape.length == 0) {
                finalShape = new long[]{1}; // 默认采样1个样本
            } else {
                finalShape = sampleShape;
            }

            // 步骤2：计算最终输出形状
            // 输出形状 = 采样形状 + [类别数]
            long numCategories = probs.numel();
            long[] outputShape = new long[finalShape.length + 1];
            System.arraycopy(finalShape, 0, outputShape, 0, finalShape.length);
            outputShape[finalShape.length] = numCategories;

            // 步骤3：生成正确形状的随机样本（修复形状不匹配）
            // 计算总采样数 = 采样形状乘积 * 试验次数
            long totalSamples = 1;
            for (long s : finalShape) {
                totalSamples *= s;
            }
            long numTrials = totalSamples * this.totalCount;

            // 生成符合多项分布的样本
            Tensor flatProbs = probs.reshape(new long[]{-1}); // 展平概率为1D
            if (flatProbs.numel() != numCategories) {
                throw new RuntimeException("概率分布维度错误：" + flatProbs.numel() + " vs " + numCategories);
            }

            // 生成类别索引（核心修复：确保形状匹配）
            Tensor indices = multinomial(flatProbs, numTrials, true,new GeneratorOptional());

            // 统计每个类别的次数（修复形状重塑逻辑）
            Tensor counts = torch.zeros(outputShape, probs.options()); //dtype().toScalarType(), probs.device()
            Tensor oneHot = torch.zeros(new long[]{numTrials, numCategories}, probs.options()); //dtype().toScalarType(), probs.device()
            oneHot = oneHot.scatter_(1, indices.reshape(new long[]{numTrials, 1}), new Scalar(1));

            // 重塑并求和得到最终计数
            long[] reshapeShape = new long[finalShape.length + 2];
            System.arraycopy(finalShape, 0, reshapeShape, 0, finalShape.length);
            reshapeShape[finalShape.length] = this.totalCount;
            reshapeShape[finalShape.length + 1] = numCategories;

            counts = oneHot.reshape(reshapeShape).sum(finalShape.length);

            return counts;
        } catch (Exception e) {
            // 完善错误信息，便于调试
            throw new RuntimeException("采样失败：" + e.getMessage(), e);
        }
    }

    // 简化采样方法（无参数）
    public Tensor sample() {
        return sample(new long[]{});
    }

    // 计算对数概率（修复数值稳定性）
    public Tensor logProb(Tensor value) {
        if (!probsValid) {
            throw new RuntimeException("概率分布未初始化！");
        }

        // 确保输入形状匹配
        if (value.dim() < 1 || value.size(value.dim()-1) != probs.numel()) {
            throw new IllegalArgumentException("输入形状必须匹配类别数：" + probs.numel());
        }

        // 计算对数概率（处理0的情况）
        Tensor logProbsExpanded = logProbs.expand_as(value);
        Tensor clampedValue = value.clamp(new ScalarOptional(new Scalar(0)), new ScalarOptional(new Scalar(totalCount))); // 限制取值范围
        Tensor logFactorialN = lgamma(scalar_tensor(new Scalar(totalCount + 1)));
        Tensor logFactorialK = lgamma(clampedValue.add(new Scalar(1.0f))).sum(-1);
        Tensor term = clampedValue.mul(logProbsExpanded).sum(-1);

        return logFactorialN.sub(logFactorialK).add(term);
    }

    // 计算熵
    public double entropy() {
        Tensor entropyPerCategory = probs.mul(logProbs).neg();
        return totalCount * entropyPerCategory.sum().item().toDouble();
    }

    // 获取类别数
    public long getNumCategories() {
        return probs.numel();
    }

    // 资源释放
    public void close() {
        if (probs != null) probs.close();
        if (logProbs != null) logProbs.close();
    }
}
