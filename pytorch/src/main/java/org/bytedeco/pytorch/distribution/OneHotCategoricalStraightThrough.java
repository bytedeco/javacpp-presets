package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * 最终版OneHotCategoricalStraightThrough
 * 1. 彻底修复全零概率NaN问题
 * 2. 简化ST逻辑，保证梯度回流
 * 3. 适配javacpp-presets所有限制
 */
public class OneHotCategoricalStraightThrough extends Distribution implements AutoCloseable {
    private final Tensor probs;
    private final Tensor normalizedProbs;
    private final int numCategories;

    public OneHotCategoricalStraightThrough(Tensor probs) {
        // 1. 校验非负
        Tensor negMask = torch.lt(probs, torch.tensor(0.0f, probs.options()));
        if (torch.any(negMask).item().toBool()) {
            negMask.close();
            throw new IllegalArgumentException("probs所有元素必须非负！");
        }
        negMask.close();

        // 2. 归一化（彻底修复全零概率NaN）
        this.numCategories = (int) probs.size(-1);
        Tensor sum = probs.sum(new long[]{-1}, true,new ScalarTypeOptional()); // 最后一维求和
        // 全零概率时，直接返回均匀分布（1/numCategories）
        Tensor uniformProbs = torch.full(probs.sizes(),new Scalar( 1.0f / numCategories), probs.options());
        // 非零概率时，正常归一化；全零概率时，用均匀分布替代
        this.normalizedProbs = torch.where(
                sum.lt(new Scalar(1e-8)), // 和小于1e-8视为全零
                uniformProbs,
                probs.div(sum)
        );
        this.probs = probs; // 保留原始张量，保证梯度链路

        // 释放临时张量
        sum.close();
        uniformProbs.close();
    }

    @Override
    public String name() {
        return "OneHotCategoricalStraightThrough";
    }

    /**
     * 最终版ST采样：极简 + 保证梯度回流
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 1. 基础采样（适配batch逻辑，同时保留梯度）
        Tensor expandedProbs = normalizedProbs;
        // 处理sampleShape（扩展概率形状）
        if (sampleShape.length > 0) {
            long[] newShape = new long[sampleShape.length + (int)normalizedProbs.dim()];
            System.arraycopy(sampleShape, 0, newShape, 0, sampleShape.length);
            System.arraycopy(normalizedProbs.sizes().vec().get(), 0, newShape, sampleShape.length, (int)normalizedProbs.dim());
            expandedProbs = normalizedProbs.expand(newShape);
        }

        // 2. 扁平化采样（适配multinomial）
        long[] flatShape = new long[]{-1, numCategories};
        Tensor flatProbs = expandedProbs.reshape(flatShape);
        Tensor flatIndices = multinomial(flatProbs, 1, true, new GeneratorOptional()).to(kLong());
        Tensor flatOneHot = one_hot(flatIndices, numCategories).to(probs.dtype()).reshape(flatShape);

        // 3. 恢复形状
        Tensor oneHot = flatOneHot.reshape(expandedProbs.sizes());
        if (sampleShape.length == 0 && oneHot.dim() > probs.dim()) {
            oneHot = oneHot.squeeze(); // 单样本时压缩维度
        }

        // 4. 核心ST逻辑（直接基于原始probs，保证梯度回流）
        // 适配batch：扩展原始probs到采样形状
        Tensor expandedOriginProbs = probs;
        if (sampleShape.length > 0) {
            expandedOriginProbs = probs.expand(expandedProbs.sizes());
        }
        Tensor sample = oneHot.sub(expandedOriginProbs).detach().add(expandedOriginProbs);

        // 释放临时张量
        if (!expandedProbs.equals(normalizedProbs)) expandedProbs.close();
        flatProbs.close();
        flatIndices.close();
        flatOneHot.close();
        oneHot.close();
        if (!expandedOriginProbs.equals(probs)) expandedOriginProbs.close();

        return sample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        if (v.size(-1) != numCategories) {
            throw new IllegalArgumentException("输入最后一维必须为类别数" + numCategories + "，实际为" + v.size(-1));
        }

        // 校验独热向量合法性
        Tensor vSum = v.sum(-1);
        Tensor vIsBinary = torch.logical_or(torch.eq(v, new Scalar(0.0f)), torch.eq(v, new Scalar(1.0f)));
        Tensor vIsBinaryAll = torch.all(vIsBinary, -1);
        Tensor isValid = torch.logical_and(torch.eq(vSum, new Scalar(1.0f)), vIsBinaryAll);

        // 数值稳定的log_prob计算
        Tensor safeProbs = torch.clamp(normalizedProbs,  new ScalarOptional(new Scalar(1e-8f)), new ScalarOptional(new Scalar(1.0f)));
        Tensor logProbs = torch.log(safeProbs);
        Tensor logProbValid = torch.sum(torch.mul(v, logProbs), -1);

        // 非法输入返回-Inf
        Tensor logProb = torch.where(
                isValid,
                logProbValid,
                torch.full_like(logProbValid, new Scalar( Float.NEGATIVE_INFINITY))
        );

        // 释放临时张量
        vSum.close();
        vIsBinary.close();
        vIsBinaryAll.close();
        isValid.close();
        safeProbs.close();
        logProbs.close();
        logProbValid.close();

        return logProb;
    }

    @Override
    public Tensor mean() {
        return normalizedProbs.clone();
    }

    @Override
    public Tensor entropy() {
        Tensor safeProbs = torch.clamp(normalizedProbs, new ScalarOptional(new Scalar(1e-8f)), new ScalarOptional(new Scalar(1.0f)));
        Tensor entropy = torch.neg(torch.sum(torch.mul(safeProbs, torch.log(safeProbs)), -1));
        safeProbs.close();
        return entropy;
    }

    // Getter
    public Tensor getProbs() { return probs; }
    public Tensor getNormalizedProbs() { return normalizedProbs; }
    public int getNumCategories() { return numCategories; }

    @Override
    public void close() {
        if (probs != null) probs.close();
        if (normalizedProbs != null) normalizedProbs.close();
    }
}
