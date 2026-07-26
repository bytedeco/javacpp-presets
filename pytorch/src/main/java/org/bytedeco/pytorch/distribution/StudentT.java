package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.internal.GammaSampler;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * StudentT（学生t分布）实现
 * df(ν)：自由度（>0，形状：batch_shape）
 * loc(μ)：位置参数（形状：batch_shape）
 * scale(σ)：尺度参数（>0，形状：batch_shape）
 * 支持批量参数、批量采样，具备完整的合法性校验和数值稳定性
 */
public class StudentT extends Distribution implements AutoCloseable {
    private final Tensor df;                // 自由度ν（>0）
    private final Tensor loc;               // 位置参数μ
    private final Tensor scale;             // 尺度参数σ（>0）
    private final Tensor normalizedScale;   // 归一化后的尺度参数（避免外部修改）

    // 预定义标量（复用避免重复创建，提升性能+规范）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值
    private static final Scalar SCALAR_INF = new Scalar(Double.POSITIVE_INFINITY);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Double.NEGATIVE_INFINITY);
    private static final LongOptional DIM_NEG_1 = new LongOptional(-1);

    /**
     * 构造函数：严格校验参数合法性 + 深拷贝
     * @param df 自由度ν（必须>0）
     * @param loc 位置参数μ
     * @param scale 尺度参数σ（必须>0）
     * @throws IllegalArgumentException 参数非法/设备不匹配抛出异常
     */
    public StudentT(Tensor df, Tensor loc, Tensor scale) {
        // 1. 校验df>0（添加数值容忍度，避免浮点误差）
        Tensor dfLe0 = torch.le(df, torch.tensor(1e-8, df.options()));
        if (torch.any(dfLe0).item().toBool()) {
            dfLe0.close();
            throw new IllegalArgumentException("自由度df(ν)必须大于0（数值容忍度1e-8）！");
        }

        // 2. 校验scale>0
        Tensor scaleLe0 = torch.le(scale, torch.tensor(1e-8, scale.options()));
        if (torch.any(scaleLe0).item().toBool()) {
            dfLe0.close();
            scaleLe0.close();
            throw new IllegalArgumentException("尺度参数scale(σ)必须大于0（数值容忍度1e-8）！");
        }

        // 3. 校验设备一致性
        Tensor[] tensors = {df, loc, scale};
        for (int i = 1; i < tensors.length; i++) {
            if (!tensors[0].device().equals(tensors[i].device())) {
                dfLe0.close();
                scaleLe0.close();
                throw new IllegalArgumentException(
                        String.format("参数设备不匹配：df=%s, loc=%s, scale=%s",
                                df.device().toString(), loc.device().toString(), scale.device().toString())
                );
            }
        }

        // 4. 校验形状可广播（保证批量运算合法）
        try {
            torch.broadcast_tensors(new TensorVector(df, loc, scale));
        } catch (Exception e) {
            dfLe0.close();
            scaleLe0.close();
            throw new IllegalArgumentException("参数形状无法广播：" + e.getMessage());
        }

        // 5. 初始化核心参数（深拷贝避免外部修改）
        this.df = df.clone();
        this.loc = loc.clone();
        this.scale = scale.clone();
        // 数值稳定化处理尺度参数
        this.normalizedScale = torch.clamp(
                this.scale,
                new ScalarOptional(new Scalar(1e-8)),
                new ScalarOptional(new Scalar(1e10)) // 限制上限避免数值溢出
        ).clone();

        // 释放校验临时张量
        dfLe0.close();
        scaleLe0.close();
    }

    @Override
    public String name() {
        return "StudentT";
    }

    /**
     * 采样：实现学生t分布的精确采样，支持任意批量采样形状
     * 公式：X = μ + σ * Z / sqrt(V/ν)
     * Z ~ Normal(0,1)，V ~ Chi2(ν)（卡方分布）
     * @param sampleShape 批量采样形状
     * @return 采样结果张量（形状：sampleShape + batch_shape）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展形状（sampleShape + batch_shape）
        long[] extendedShape = getExtendedShape(df, sampleShape);

        // 步骤2：扩展参数到采样形状
        Tensor expandedDf = df.expand(extendedShape);
        Tensor expandedLoc = loc.expand(extendedShape);
        Tensor expandedScale = normalizedScale.expand(extendedShape);

        // 步骤3：采样标准正态分布Z ~ N(0,1)
        Tensor z = torch.randn(extendedShape, df.options());

        // 步骤4：采样卡方分布V ~ Chi2(ν)（V = sum_{i=1}^ν Z_i²）
        // 实现方式：gamma(ν/2, 2) 等价于 Chi2(ν)
        Tensor dfHalf = torch.mul(expandedDf, new Scalar(0.5f));
        Tensor chiOp = torch.full_like(dfHalf, new Scalar(2.0f), dfHalf.options(),new MemoryFormatOptional());
        Tensor chi2 = GammaSampler.gamma(dfHalf, chiOp);

        // 步骤5：计算t分布采样值 X = μ + σ * Z / sqrt(V/ν)
        Tensor vOverDf = torch.div(chi2, expandedDf);
        Tensor sqrtVOverDf = torch.sqrt(vOverDf);
        // 数值稳定处理：避免除零
        Tensor sqrtVOverDfSafe = torch.clamp(sqrtVOverDf, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(Float.POSITIVE_INFINITY)));
        Tensor tSample = torch.div(z, sqrtVOverDfSafe);
        Tensor sample = torch.add(expandedLoc, torch.mul(expandedScale, tSample));

        // 释放所有临时张量
        expandedDf.close();
        expandedLoc.close();
        expandedScale.close();
        z.close();
        dfHalf.close();
        chi2.close();
        vOverDf.close();
        sqrtVOverDf.close();
        sqrtVOverDfSafe.close();
        tSample.close();

        return sample;
    }

    /**
     * 对数概率：实现学生t分布的精确对数概率公式，增强数值稳定性
     * 公式严格遵循学生t分布的概率密度函数对数形式
     * @param v 输入张量（形状需与参数可广播）
     * @return 对数概率张量（形状：batch_shape）
     */
    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：扩展参数到输入形状
        Tensor expandedDf = df.expand(v.sizes());
        Tensor expandedLoc = loc.expand(v.sizes());
        Tensor expandedScale = normalizedScale.expand(v.sizes());

        // 步骤2：数值稳定性处理
        Tensor safeDf = torch.clamp(expandedDf, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e10)));
        Tensor safeScale = torch.clamp(expandedScale, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e10)));
        Tensor safeV = v.clone();

        // 步骤3：计算y = (x - μ)/σ
        Tensor y = torch.div(torch.sub(safeV, expandedLoc), safeScale);

        // 步骤4：计算对数概率各项
        // term1 = lgamma((ν+1)/2)
        Tensor dfPlus1 = torch.add(safeDf, new Scalar(1.0f));
        Tensor dfPlus1Half = torch.mul(dfPlus1, new Scalar(0.5f));
        Tensor term1 = lgamma(dfPlus1Half);

        // term2 = -lgamma(ν/2)
        Tensor dfHalf = torch.mul(safeDf, new Scalar(0.5f));
        Tensor term2 = torch.neg(lgamma(dfHalf));

        // term3 = -0.5 * log(νπ)
        Tensor piScalar = torch.tensor(Math.PI, safeDf.options());
        Tensor dfPi = torch.mul(safeDf, piScalar);
        Tensor logDfPi = torch.log(dfPi);
        Tensor term3 = torch.neg(torch.mul(logDfPi, new Scalar(0.5f)));

        // term4 = -log(σ)
        Tensor logScale = torch.log(safeScale);
        Tensor term4 = torch.neg(logScale);

        // term5 = -((ν+1)/2) * log(1 + y²/ν)
        Tensor ySq = torch.pow(y, torch.tensor(1.0f).mul(new Scalar(2.0f)));
        Tensor ySqOverDf = torch.div(ySq, safeDf);
        Tensor logArg = torch.add(ySqOverDf, new Scalar(1.0f));
        Tensor logLogArg = torch.log(torch.clamp(logArg, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(Float.POSITIVE_INFINITY)))); // 避免log(0)
        Tensor term5 = torch.neg(torch.mul(dfPlus1Half, logLogArg));

        // 步骤5：完整对数概率
        Tensor logProb = torch.add(torch.add(torch.add(torch.add(term1, term2), term3), term4), term5);

        // 释放所有临时张量
        expandedDf.close();
        expandedLoc.close();
        expandedScale.close();
        safeDf.close();
        safeScale.close();
        safeV.close();
        y.close();
        dfPlus1.close();
        dfPlus1Half.close();
        term1.close();
        dfHalf.close();
        term2.close();
        piScalar.close();
        dfPi.close();
        logDfPi.close();
        term3.close();
        logScale.close();
        term4.close();
        ySq.close();
        ySqOverDf.close();
        logArg.close();
        logLogArg.close();
        term5.close();

        return logProb;
    }

    /**
     * 均值：学生t分布的均值
     * 公式：E[X] = μ（ν>1），否则为+∞（无定义）
     * @return 均值张量（返回拷贝避免外部修改）
     */
    @Override
    public Tensor mean() {
        // 步骤1：判断ν>1（添加数值容忍度）
        Tensor dfGt1 = torch.gt(df, torch.tensor(1.0 + 1e-8, df.options()));

        // 步骤2：扩展loc到df形状（保证维度对齐）
        Tensor expandedLoc = loc.expand(df.sizes());

        // 步骤3：ν>1返回μ，否则返回+∞（而非NaN，符合数学定义）
        Tensor mean = torch.where(
                dfGt1,
                expandedLoc,
                torch.full_like(expandedLoc, new Scalar(Float.POSITIVE_INFINITY), expandedLoc.options(),new MemoryFormatOptional())
        );

        // 释放临时张量
        dfGt1.close();
        expandedLoc.close();

        return mean.clone();
    }

    /**
     * 熵：实现学生t分布的精确解析熵公式
     * 公式：H = ((ν+1)/2)(ψ((ν+1)/2) - ψ(ν/2)) + 0.5log(νπ) + logσ
     * ψ为digamma函数（lgamma的导数）
     * @return 熵张量（形状：batch_shape）
     */
    @Override
    public Tensor entropy() {
        // 步骤1：数值稳定性处理
        Tensor safeDf = torch.clamp(df, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e10)));
        Tensor safeScale = torch.clamp(normalizedScale, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e10)));

        // 步骤2：计算digamma项
        Tensor dfHalf = torch.mul(safeDf, new Scalar(0.5f));
        Tensor dfPlus1Half = torch.mul(torch.add(safeDf, new Scalar(1.0f)), new Scalar(0.5f));
        Tensor digammaDfHalf = digamma(dfHalf);
        Tensor digammaDfPlus1Half = digamma(dfPlus1Half);
        Tensor digammaDiff = torch.sub(digammaDfPlus1Half, digammaDfHalf);

        // 步骤3：计算各项
        // term1 = ((ν+1)/2) * (ψ((ν+1)/2) - ψ(ν/2))
        Tensor term1 = torch.mul(dfPlus1Half, digammaDiff);

        // term2 = 0.5 * log(νπ)
        Tensor piScalar = torch.tensor(Math.PI, safeDf.options());
        Tensor dfPi = torch.mul(safeDf, piScalar);
        Tensor logDfPi = torch.log(dfPi);
        Tensor term2 = torch.mul(logDfPi, new Scalar(0.5f));

        // term3 = log(σ)
        Tensor logScale = torch.log(safeScale);

        // 步骤4：完整熵公式
        Tensor entropy = torch.add(torch.add(term1, term2), logScale);

        // 释放临时张量
        safeDf.close();
        safeScale.close();
        dfHalf.close();
        dfPlus1Half.close();
        digammaDfHalf.close();
        digammaDfPlus1Half.close();
        digammaDiff.close();
        term1.close();
        piScalar.close();
        dfPi.close();
        logDfPi.close();
        term2.close();
        logScale.close();

        return entropy;
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     */
    @Override
    public void close() {
        df.close();
        loc.close();
        scale.close();
        normalizedScale.close();
        // 释放预定义常量

    }

    // Getter方法（提升易用性）
    public Tensor getDf() { return df; }
    public Tensor getLoc() { return loc; }
    public Tensor getScale() { return scale; }
    public Tensor getNormalizedScale() { return normalizedScale; }

    // 额外实用方法：获取方差
    // 方差：σ²ν/(ν-2)（ν>2），+∞（1<ν≤2），无定义（ν≤1）
    public Tensor variance() {
        Tensor safeDf = torch.clamp(df, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e10)));
        Tensor safeScale = torch.clamp(normalizedScale, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e10)));

        // 判断ν>2
        Tensor dfGt2 = torch.gt(safeDf, torch.tensor(2.0 + 1e-8, safeDf.options()));
        // 判断1<ν≤2
        Tensor dfGt1Le2 = torch.logical_and(
                torch.gt(safeDf, new Scalar(1.0f)),
                torch.le(safeDf, new Scalar(2.0 + 1e-8))
        );

        // 计算ν>2时的方差
        Tensor dfMinus2 = torch.sub(safeDf, new Scalar(2.0f));
        Tensor varValid = torch.mul(torch.pow(safeScale, new Scalar(2.0f)), torch.div(safeDf, dfMinus2));

        // 构建最终方差：ν>2→varValid；1<ν≤2→+∞；否则→NaN
        Tensor variance = torch.where(
                dfGt2,
                varValid,
                torch.where(
                        dfGt1Le2,
                        torch.full_like(varValid, new Scalar(Float.POSITIVE_INFINITY)),
                        torch.full_like(varValid, new Scalar(Float.NaN))
                )
        );

        // 释放临时张量
        safeDf.close();
        safeScale.close();
        dfGt2.close();
        dfGt1Le2.close();
        dfMinus2.close();
        varValid.close();

        return variance;
    }
}
