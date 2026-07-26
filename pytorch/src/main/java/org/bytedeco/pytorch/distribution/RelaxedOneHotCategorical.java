package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * RelaxedOneHotCategorical（Gumbel-Softmax分布）
 * 离散独热类别分布的连续松弛版本，温度τ控制松弛程度（τ→0收敛到离散分布）
 * temperature(τ)：温度参数（>0，batch_shape）
 * probs(p)：类别概率（0<p_i<1，最后一维和为1，形状：batch_shape + [k]）
 */
public class RelaxedOneHotCategorical extends Distribution implements AutoCloseable {
    private final Tensor temperature;       // 温度参数τ（>0，batch_shape）
    private final Tensor probs;             // 类别概率p（batch_shape + [k]）
    private final Tensor normalizedProbs;   // 归一化后的概率（保证最后一维和为1）
    private final int numCategories;        // 类别数k

    // 预定义标量（复用避免重复创建，提升性能+规范）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值
    private static final Scalar SCALAR_INF = new Scalar(Double.POSITIVE_INFINITY);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Double.NEGATIVE_INFINITY);
    private static final ScalarTypeOptional DIM_NEG_1 = new ScalarTypeOptional(new Scalar(-1.0f)); // new LongOptional(-1);

    /**
     * 构造函数：严格校验参数合法性 + 概率归一化 + 深拷贝
     * @param temperature 温度参数τ（必须>0）
     * @param probs 类别概率（0<p_i<1，最后一维和为1）
     * @throws IllegalArgumentException 参数非法/设备不匹配抛出异常
     */
    public RelaxedOneHotCategorical(Tensor temperature, Tensor probs) {
        // 1. 校验temperature>0（添加数值容忍度）
        Tensor tempLe0 = torch.le(temperature, torch.tensor(1e-8, temperature.options()));
        if (torch.any(tempLe0).item().toBool()) {
            tempLe0.close();
            throw new IllegalArgumentException("temperature(τ)必须大于0（数值容忍度1e-8）！");
        }

        // 2. 校验probs非负且形状合法
        Tensor probsNeg = torch.lt(probs, torch.tensor(0.0f, probs.options()));
        if (torch.any(probsNeg).item().toBool()) {
            tempLe0.close();
            probsNeg.close();
            throw new IllegalArgumentException("probs所有元素必须非负！");
        }

        // 3. 概率归一化（保证最后一维和为1，处理数值误差）
        Tensor probsSum = probs.sum(new long[]{-1}, true, new ScalarTypeOptional());
        Tensor probsSumSafe = torch.clamp(probsSum, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(probsSum.max().item()));
        this.normalizedProbs = probs.div(probsSumSafe).clone();

        // 4. 初始化核心参数
        // temperature must broadcast to probs: allow scalar, [B], or [B,1] → store as [B,1] when needed
        Tensor tempCloned = temperature.clone();
        if (tempCloned.dim() == 0) {
            // scalar → [1,1] then expand
            tempCloned = tempCloned.reshape(1, 1);
        } else if (tempCloned.dim() == 1 && this.normalizedProbs.dim() >= 2
                && tempCloned.size(0) == this.normalizedProbs.size(0)) {
            // [B] → [B,1] so expand to [B,K] works
            tempCloned = tempCloned.unsqueeze(-1);
        }
        this.temperature = tempCloned;
        this.probs = probs.clone();
        this.numCategories = (int) this.normalizedProbs.size(-1);

        // 5. 校验设备一致性（广播后）
        Tensor broadcastedTemp = this.temperature.expand(this.normalizedProbs.sizes());
        if (!broadcastedTemp.device().equals(this.normalizedProbs.device())) {
            tempLe0.close();
            probsNeg.close();
            probsSum.close();
            probsSumSafe.close();
            broadcastedTemp.close();
            throw new IllegalArgumentException(
                    String.format("temperature和probs设备不匹配：temp=%s, probs=%s",
                            this.temperature.device().toString(), this.normalizedProbs.device().toString())
            );
        }

        // 释放校验/归一化临时张量
        tempLe0.close();
        probsNeg.close();
        probsSum.close();
        probsSumSafe.close();
        broadcastedTemp.close();
    }

    @Override
    public String name() {
        return "RelaxedOneHotCategorical";
    }

    /**
     * 采样：实现Gumbel-Softmax的精确采样，支持任意批量采样形状
     * 核心公式：y = Softmax( (log(p_i) + Gumbel(0,1)) / τ )
     * Gumbel采样：g = -log(-log(U))，U~Uniform(0,1)（修正原代码公式错误）
     * @param sampleShape 批量采样形状
     * @return 采样结果张量（形状：sampleShape + batch_shape + [k]，值域(0,1)，最后一维和为1）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展形状（sampleShape + batch_shape + [k]）
        long[] extendedShape = getExtendedShape(normalizedProbs, sampleShape);

        // 步骤2：扩展参数到采样形状
        Tensor expandedTemp = temperature.expand(extendedShape);
        Tensor expandedProbs = normalizedProbs.expand(extendedShape);

        // 步骤3：生成均匀分布U~(0,1)，数值稳定处理（避免U=0/1）
        Tensor u = torch.rand(extendedShape, probs.options());
        Tensor uSafe = torch.clamp(u, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));

        // 步骤4：采样Gumbel噪声（修正原代码公式错误：g = -log(-log(U))）
        Tensor logU = torch.log(uSafe);
        Tensor negLogU = torch.neg(logU);
        Tensor logNegLogU = torch.log(negLogU);
        Tensor gumbelNoise = torch.neg(logNegLogU); // 正确的Gumbel采样公式

        // 步骤5：计算log(p_i) + gumbel_noise
        Tensor safeProbs = torch.clamp(expandedProbs, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0f)));
        Tensor logProbs = torch.log(safeProbs);
        Tensor logits = torch.add(logProbs, gumbelNoise);

        // 步骤6：Gumbel-Softmax核心计算
        Tensor scaledLogits = torch.div(logits, expandedTemp);
        Tensor sample = torch.softmax(scaledLogits,-1);// DIM_NEG_1); // 最后一维Softmax

        // 释放所有临时张量
        expandedTemp.close();
        expandedProbs.close();
        u.close();
        uSafe.close();
        logU.close();
        negLogU.close();
        logNegLogU.close();
        gumbelNoise.close();
        safeProbs.close();
        logProbs.close();
        logits.close();
        scaledLogits.close();

        return sample;
    }
    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：输入合法性校验
        // 1.1 校验最后一维为类别数k
        if (v.size(-1) != numCategories) {
            throw new IllegalArgumentException(
                    "输入最后一维必须为类别数" + numCategories + "，实际为" + v.size(-1)
            );
        }

        // 临时张量声明（统一在finally释放）
        Tensor vClone = null;
        Tensor vSum = null;
        Tensor vSumSqueezed = null;
        Tensor vSumExpanded = null;
        Tensor vNormalized = null;
        Tensor vNeg = null;
        Tensor vSumValid = null;
        Tensor anyNeg = null;
        Tensor invalid = null;
        Tensor expandedTemp = null;
        Tensor expandedProbs = null;
        Tensor safeV = null;
        Tensor safeProbs = null;
        Tensor safeTemp = null;
        Tensor kScalar = null;
        Tensor lgammaK = null;
        Tensor term1 = null;
        Tensor kMinus1 = null;
        Tensor logTemp = null;
        Tensor term2 = null;
        Tensor logV = null;
        Tensor tempLogV = null;
        Tensor logProbs = null;
        Tensor tempLogVMinusLogProbs = null;
        Tensor term3 = null;
        Tensor vMinus1 = null;
        Tensor vMinus1LogV = null;
        Tensor term4 = null;
        Tensor logProbValid = null;
        Tensor logProb = null;
        Tensor fullNegInf = null;
        Tensor term1Expanded = null;

        try {
            vClone = v.clone(); // 避免修改原始输入

            // ==============================================
            // 核心修复1：严格校验非归一化输入（返回-∞）
            // ==============================================
            vSum = torch.sum(vClone, new long[]{-1}, true,new ScalarTypeOptional()); // [batch_shape, 1]
            vSumSqueezed = vSum.squeeze(-1); // [batch_shape]
            // 严格校验和是否偏离1（容忍度1e-3）
            vSumValid = torch.logical_and(
                    torch.ge(vSumSqueezed, new Scalar(1.0 - 1e-3)),
                    torch.le(vSumSqueezed, new Scalar(1.0 + 1e-3))
            ); // [batch_shape]

            // 校验是否含明显负数
            vNeg = torch.lt(vClone, new Scalar(-1e-6)); // [batch_shape + [k]]
            anyNeg = torch.any(vNeg, -1); // [batch_shape]

            // 最终invalid掩码：和偏离1 或 含负数 → 返回-∞
            invalid = torch.logical_or(anyNeg, torch.logical_not(vSumValid));

            // ==============================================
            // 核心修复2：仅对合法输入做归一化修正
            // ==============================================
            // 创建掩码：仅合法输入需要归一化
            Tensor validMask = torch.logical_not(invalid).unsqueeze(-1).expand(vClone.sizes());
            // 初始化归一化输入为原始输入
            vNormalized = vClone.clone();
            // 对合法输入做归一化修正（避免非法输入参与计算）
            vSumExpanded = vSum.expand(vClone.sizes());
            Tensor normalizedPart = torch.div(vClone, vSumExpanded);
            // 仅替换合法输入的部分
            vNormalized = torch.where(validMask, normalizedPart, vNormalized);

            // 步骤2：扩展参数到输入形状
            long[] vShape = vNormalized.sizes().vec().get();
            expandedTemp = temperature.expand(vShape);
            expandedProbs = normalizedProbs.expand(vShape);

            // 步骤3：数值稳定处理输入和参数
            safeV = torch.clamp(vNormalized, new ScalarOptional(new Scalar(1e-10)), new ScalarOptional(new Scalar(1.0f)));
            safeProbs = torch.clamp(expandedProbs, new ScalarOptional(new Scalar(1e-10)), new ScalarOptional(new Scalar(1.0f)));
            safeTemp = torch.clamp(expandedTemp, new ScalarOptional(new Scalar(1e-10)), new ScalarOptional(new Scalar(1e6)));

            // 步骤4：计算对数概率各项（精确公式）
            // term1：lgamma(k)（排列数的对数，k为类别数）
            kScalar = torch.tensor(numCategories, safeProbs.options());
            lgammaK = lgamma(kScalar);
            term1 = torch.neg(lgammaK); // -lgamma(k) → 标量/1维张量

            // 统一term1的维度，兼容标量/批量输入
            if (term1.dim() == 1 && term1.size(0) == 1) {
                term1 = term1.squeeze(0); // 1维→0维标量
            }
            if (invalid.dim() == 0) {
                term1Expanded = term1;
            } else {
                term1Expanded = term1.expand(invalid.sizes());
            }

            // term2：(k-1)*log(τ) → [batch_shape]
            kMinus1 = torch.sub(kScalar, new Scalar(1.0f));
            logTemp = torch.log(safeTemp);
            term2 = torch.mul(kMinus1, logTemp).sum(DIM_NEG_1); // [batch_shape]

            // term3：sum( τ*log(v_i) - log(p_i) ) → [batch_shape]
            logV = torch.log(safeV);
            tempLogV = torch.mul(safeTemp, logV);
            logProbs = torch.log(safeProbs);
            tempLogVMinusLogProbs = torch.sub(tempLogV, logProbs);
            term3 = torch.sum(tempLogVMinusLogProbs, DIM_NEG_1); // [batch_shape]

            // term4：-sum( (v_i - 1) * log(v_i) ) → [batch_shape]
            vMinus1 = torch.sub(safeV, new Scalar(1.0f));
            vMinus1LogV = torch.mul(vMinus1, logV);
            term4 = torch.neg(torch.sum(vMinus1LogV, DIM_NEG_1)); // [batch_shape]

            // 步骤5：完整对数概率（合法输入）→ [batch_shape]
            logProbValid = torch.add(
                    torch.add(
                            torch.add(term1Expanded, term2),
                            term3
                    ),
                    term4
            );

            // 步骤6：处理非法输入（返回-∞）
            fullNegInf = torch.full_like(
                    logProbValid,
                    new Scalar(Float.NEGATIVE_INFINITY),
                    logProbValid.options(),
                    new MemoryFormatOptional()
            );

            // 最终log_prob：非法输入返回-∞，合法输入返回计算值
            logProb = torch.where(invalid, fullNegInf, logProbValid);

            return logProb.clone(); // 返回拷贝避免外部释放影响

        } finally {
            // 释放所有临时张量（避免内存泄漏）
            safeClose(vClone);
            safeClose(vSum);
            safeClose(vSumSqueezed);
            safeClose(vSumExpanded);
            safeClose(vNormalized);
            safeClose(vNeg);
            safeClose(vSumValid);
            safeClose(anyNeg);
            safeClose(invalid);
            safeClose(expandedTemp);
            safeClose(expandedProbs);
            safeClose(safeV);
            safeClose(safeProbs);
            safeClose(safeTemp);
            safeClose(kScalar);
            safeClose(lgammaK);
            safeClose(term1);
            safeClose(kMinus1);
            safeClose(logTemp);
            safeClose(term2);
            safeClose(logV);
            safeClose(tempLogV);
            safeClose(logProbs);
            safeClose(tempLogVMinusLogProbs);
            safeClose(term3);
            safeClose(vMinus1);
            safeClose(vMinus1LogV);
            safeClose(term4);
            safeClose(logProbValid);
            safeClose(fullNegInf);
            safeClose(term1Expanded);
        }
    }

    private void safeClose(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                // 释放失败仅打印警告，不中断流程
                System.err.println("张量释放警告: " + e.getMessage());
            }
        }
    }
    


    /**
     * 均值：Gumbel-Softmax分布的均值等于类别概率p（与离散分布一致）
     * @return 均值张量（返回拷贝避免外部修改）
     */
    @Override
    public Tensor mean() {
        return normalizedProbs.clone();
    }

    /**
     * 熵：实现Gumbel-Softmax分布的解析熵公式，体现温度和离散熵的联合影响
     * 公式：H = 离散熵 + (k-1)*τ + τ*sum(p_i * log(p_i))（简化高精度近似）
     * @return 熵张量（形状：batch_shape）
     */
    @Override
    public Tensor entropy() {
        // 步骤1：数值稳定处理参数
        Tensor safeProbs = torch.clamp(normalizedProbs, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0f)));
        Tensor safeTemp = torch.clamp(temperature, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e6)));

        // 步骤2：计算离散独热类别分布的熵（-sum(p_i*log(p_i))）
        Tensor logProbs = torch.log(safeProbs);
        Tensor pLogP = torch.mul(safeProbs, logProbs);
        Tensor discreteEntropy = torch.neg(torch.sum(pLogP, DIM_NEG_1));

        // 步骤3：计算温度相关的松弛熵项
        // term1：(k-1)*τ（类别数相关的熵增）
        Tensor kMinus1 = torch.tensor(numCategories - 1, safeTemp.options());
        Tensor tempTerm1 = torch.mul(kMinus1, safeTemp);
        // term2：τ*sum(p_i*log(p_i))（温度修正项）
        Tensor tempTerm2 = torch.mul(safeTemp, torch.sum(pLogP, DIM_NEG_1));

        // 步骤4：完整熵公式（离散熵 + 温度松弛项）
        Tensor entropy = torch.add(torch.add(discreteEntropy, tempTerm1), tempTerm2);

        // 释放临时张量
        safeProbs.close();
        safeTemp.close();
        logProbs.close();
        pLogP.close();
        discreteEntropy.close();
        kMinus1.close();
        tempTerm1.close();
        tempTerm2.close();

        return entropy;
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     */
    @Override
    public void close() {
        temperature.close();
        probs.close();
        normalizedProbs.close();
        // 释放预定义常量

    }

    // Getter方法（提升易用性）
    public Tensor getTemperature() { return temperature; }
    public Tensor getProbs() { return probs; }
    public Tensor getNormalizedProbs() { return normalizedProbs; }
    public int getNumCategories() { return numCategories; }

    // 额外实用方法：获取离散独热类别分布的熵（基准对比）
    public Tensor discreteEntropy() {
        Tensor safeProbs = torch.clamp(normalizedProbs, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0f)));
        Tensor entropy = torch.neg(torch.sum(torch.mul(safeProbs, torch.log(safeProbs)), DIM_NEG_1));
        safeProbs.close();
        return entropy;
    }
}
