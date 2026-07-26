package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Binomial extends Distribution implements AutoCloseable {
    private final Tensor total_count; // n（试验次数，≥0）
    private final Tensor probs;       // p（单次成功概率，0≤p≤1）

    // 预定义标量（通过数值计算后创建，避免Scalar直接运算）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    private static final double EPS_VALUE = 1e-8;
    private static final Scalar EPS = new Scalar(EPS_VALUE);
    private static final Scalar ONE_MINUS_EPS = new Scalar(1.0 - EPS_VALUE); // 提前计算1-EPS
    private static final Scalar TWO_PI_E = new Scalar(2 * Math.PI * Math.E); // 预计算常数

    // 构造函数：校验核心参数合法性（严格符合JavaCPP API）
    public Binomial(Tensor count, Tensor p) {
        // 校验total_count ≥ 0（使用torch.lt + any + item + toBool）
        if (torch.any(torch.lt(count, new Scalar(0.0f))).item().toBool()) {
            throw new IllegalArgumentException("二项分布total_count(n)必须≥0！");
        }
        // 校验probs ∈ [0,1]
        Tensor pLt0 = torch.lt(p, new Scalar(0.0f));
        Tensor pGt1 = torch.gt(p, new Scalar(1.0f));
        Tensor pInvalid = torch.logical_or(pLt0, pGt1);
        if (torch.any(pInvalid).item().toBool()) {
            // 释放临时张量
            pLt0.close();
            pGt1.close();
            pInvalid.close();
            throw new IllegalArgumentException("二项分布probs(p)必须满足0≤p≤1！");
        }
        // 释放校验张量
        pLt0.close();
        pGt1.close();
        pInvalid.close();

        // 深拷贝避免外部修改内部状态
        this.total_count = count.clone();
        this.probs = p.clone();
    }

    @Override
    public String name() {
        return "Binomial";
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展采样形状（与其他分布逻辑对齐）
        long[] extendedShape = getExtendedShape(total_count, sampleShape);

        // 步骤2：扩展total_count和probs到目标形状（广播）
        Tensor expandedCount = total_count.expand(extendedShape);
        Tensor expandedProbs = probs.expand(extendedShape);

        // 步骤3：二项分布采样（torch.binomial）
        Tensor sample = torch.binomial(expandedCount, expandedProbs);

        // 释放临时张量
        expandedCount.close();
        expandedProbs.close();

        return sample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：校验输入v的合法性
        // 1.1 校验v ≥ 0 且 v ≤ total_count
        Tensor vLt0 = torch.lt(v, new Scalar(0.0));
        Tensor vGtN = torch.gt(v, total_count);
        Tensor invalidRange = torch.logical_or(vLt0, vGtN);
        if (torch.any(invalidRange).item().toBool()) {
            vLt0.close();
            vGtN.close();
            invalidRange.close();
            throw new IllegalArgumentException("log_prob输入v必须满足0≤v≤total_count！");
        }

        // 1.2 校验v为整数（二项分布成功次数必须是整数）
        Tensor vRounded = torch.round(v);
        Tensor vIsInt = torch.eq(v, vRounded);
        if (!torch.all(vIsInt).item().toBool()) {
            vRounded.close();
            vIsInt.close();
            vLt0.close();
            vGtN.close();
            invalidRange.close();
            throw new IllegalArgumentException("log_prob输入v必须是整数！");
        }

        // 步骤2：数值稳定性处理（clamp使用ScalarOptional）
        // safeProbs: 限制在[EPS, 1-EPS]，避免log(0)
        Tensor safeProbs = probs.clamp(new ScalarOptional(new Scalar(EPS_VALUE)), new ScalarOptional(new Scalar(1.0 - EPS_VALUE)));
        // 1-p: 同样限制范围
        Tensor oneMinusP = torch.tensor(1.0f).sub(safeProbs);
        oneMinusP = oneMinusP.clamp(new ScalarOptional(new Scalar(EPS_VALUE)), new ScalarOptional(new Scalar(1.0 - EPS_VALUE)));

        // 步骤3：计算组合数的对数 log(C_n^k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        Tensor nPlus1 = torch.add(total_count, new Scalar(1.0f));
        Tensor logNFact = torch.lgamma(nPlus1);

        Tensor kPlus1 = torch.add(v, new Scalar(1.0f));
        Tensor logKFact = torch.lgamma(kPlus1);

        Tensor nMinusK = torch.sub(total_count, v);
        Tensor nMinusKPlus1 = torch.add(nMinusK, new Scalar(1.0f));
        Tensor logNkFact = torch.lgamma(nMinusKPlus1);

        Tensor logFact = torch.sub(torch.sub(logNFact, logKFact), logNkFact);

        // 步骤4：计算对数概率项
        Tensor logP = torch.log(safeProbs);
        Tensor term1 = torch.mul(v, logP); // k*log(p)

        Tensor log1MinusP = torch.log(oneMinusP);
        Tensor term2 = torch.mul(nMinusK, log1MinusP); // (n-k)*log(1-p)

        Tensor logProb = torch.add(torch.add(logFact, term1), term2);

        // 释放所有临时张量
        vLt0.close();
        vGtN.close();
        invalidRange.close();
        vRounded.close();
        vIsInt.close();
        safeProbs.close();
        oneMinusP.close();
        nPlus1.close();
        logNFact.close();
        kPlus1.close();
        logKFact.close();
        nMinusK.close();
        nMinusKPlus1.close();
        logNkFact.close();
        logFact.close();
        logP.close();
        term1.close();
        log1MinusP.close();
        term2.close();

        return logProb;
    }

    @Override
    public Tensor entropy() {
        // 步骤1：计算方差项 var = n*p*(1-p)
        Tensor oneMinusP =  torch.tensor(1.0f).sub(probs);
        Tensor np = torch.mul(total_count, probs);
        Tensor var = torch.mul(np, oneMinusP);

        // 步骤2：数值稳定性处理（clamp使用ScalarOptional）
        Tensor safeVar = var.clamp(new ScalarOptional(new Scalar(EPS_VALUE)), new ScalarOptional(torch.max(var).item()));

        // 步骤3：正态近似熵公式 H ≈ 0.5 * log(2πe * var)
        Tensor twoPiEVar = torch.mul(safeVar, new Scalar(2 * Math.PI * Math.E));
        Tensor logTwoPiEVar = torch.log(twoPiEVar);
        Tensor entropy = torch.mul(logTwoPiEVar, new Scalar(0.5f));

        // 步骤4：边界处理：var=0时（p=0/p=1或n=0），熵为0
        Tensor varEq0 = torch.eq(var, new Scalar(0.0f));
        Tensor zeros = torch.zeros_like(entropy);
        entropy.masked_scatter_(varEq0, zeros);

        // 释放临时张量
        oneMinusP.close();
        np.close();
        var.close();
        safeVar.close();
        twoPiEVar.close();
        logTwoPiEVar.close();
        varEq0.close();
        zeros.close();

        return entropy;
    }

    @Override
    public Tensor mean() {
        // 均值公式：n*p（返回拷贝避免外部修改）
        Tensor mean = torch.mul(total_count, probs).clone();
        return mean;
    }

    // 辅助工具：扩展形状（复用Distribution抽象类的方法）
    protected long[] getExtendedShape(Tensor baseTensor, long... sampleShape) {
        long[] baseShape = baseTensor.sizes().vec().get();
        long[] extended = new long[sampleShape.length + baseShape.length];
        System.arraycopy(sampleShape, 0, extended, 0, sampleShape.length);
        System.arraycopy(baseShape, 0, extended, sampleShape.length, baseShape.length);
        return extended;
    }

    // 资源释放：实现AutoCloseable
    @Override
    public void close() {
        total_count.close();
        probs.close();
        // 释放预定义Scalar（JavaCPP资源管理）

    }
}
