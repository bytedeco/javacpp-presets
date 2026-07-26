package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class ContinuousBernoulli extends Distribution implements AutoCloseable {
    private final Tensor probs;  // 分布参数p∈(0,1)

    // 预定义标量（复用避免重复创建）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_2 = new Scalar(2.0);
    private static final Scalar SCALAR_LOG2 = new Scalar(Math.log(2.0));
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值

    // 构造函数：校验参数合法性 + 深拷贝
    public ContinuousBernoulli(Tensor probs) {
        // 校验p∈(0,1)（连续伯努利分布核心约束）
        Tensor pLt0 = torch.lt(probs, torch.tensor(0.0f, probs.options()));
        Tensor pGt1 = torch.gt(probs, torch.tensor(1.0f, probs.options()));
        Tensor pInvalid = torch.logical_or(pLt0, pGt1);
        if (torch.any(pInvalid).item().toBool()) {
            pLt0.close();
            pGt1.close();
            pInvalid.close();
            throw new IllegalArgumentException("连续伯努利分布probs(p)必须满足0<p<1！");
        }
        pLt0.close();
        pGt1.close();
        pInvalid.close();

        // 深拷贝避免外部修改内部状态
        this.probs = probs.clone();
    }

    @Override
    public String name() {
        return "ContinuousBernoulli";
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // Inverse-CDF (torch.distributions.ContinuousBernoulli.icdf):
        //   x = log( ((1-p) + u*(2p-1)) / (1-p) ) / log( p/(1-p) )
        //   when p≈0.5 → Uniform(0,1)
        long[] extendedShape = getExtendedShape(probs, sampleShape);
        Tensor expandedProbs = probs.expand(extendedShape);
        Tensor u = torch.rand(extendedShape, probs.options())
                .clamp(new ScalarOptional(new Scalar(1e-6)), new ScalarOptional(new Scalar(1.0 - 1e-6)));

        Tensor one = torch.tensor(1.0f, expandedProbs.options());
        Tensor two = torch.tensor(2.0f, expandedProbs.options());
        Tensor p = expandedProbs.clamp(new ScalarOptional(new Scalar(1e-6)), new ScalarOptional(new Scalar(1.0 - 1e-6)));
        Tensor oneMinusP = torch.sub(one, p);
        Tensor twoPMinus1 = torch.sub(torch.mul(p, two), one);

        Tensor nearHalf = torch.abs(twoPMinus1).lt(new Scalar(1e-4));

        // num = (1-p) + u*(2p-1)
        Tensor num = torch.add(oneMinusP, torch.mul(twoPMinus1, u));
        Tensor safeNum = num.clamp(new ScalarOptional(new Scalar(1e-12)), new ScalarOptional(new Scalar(1e12)));
        Tensor logNumOverQ = torch.log(torch.div(safeNum, oneMinusP));
        Tensor logPOverQ = torch.log(torch.div(p, oneMinusP));
        Tensor general = torch.div(logNumOverQ, logPOverQ);

        Tensor sample = torch.where(nearHalf, u, general);
        sample = sample.clamp(new ScalarOptional(new Scalar(0.0f)), new ScalarOptional(new Scalar(1.0f)));

        expandedProbs.close();
        u.close();
        one.close();
        two.close();
        p.close();
        oneMinusP.close();
        twoPMinus1.close();
        nearHalf.close();
        num.close();
        safeNum.close();
        logNumOverQ.close();
        logPOverQ.close();
        general.close();

        return sample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：校验输入v∈[0,1]
        Tensor vLt0 = torch.lt(v, torch.tensor(0.0f, v.options()));
        Tensor vGt1 = torch.gt(v, torch.tensor(1.0f, v.options()));
        Tensor vInvalid = torch.logical_or(vLt0, vGt1);
        if (torch.any(vInvalid).item().toBool()) {
            vLt0.close();
            vGt1.close();
            vInvalid.close();
            throw new IllegalArgumentException("log_prob输入v必须满足0≤v≤1！");
        }
        vLt0.close();
        vGt1.close();
        vInvalid.close();

        // 步骤2：数值稳定性处理（避免log(0)）
        Tensor safeProbs = probs.clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));
        Tensor oneMinusP = torch.sub(torch.tensor(1.0f, safeProbs.options()), safeProbs);
        Tensor safeV = v.clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));

        // 步骤3：计算核心项 x*log(p) + (1-x)*log(1-p)
        Tensor logP = torch.log(safeProbs);
        Tensor log1MinusP = torch.log(oneMinusP);
        Tensor logCore = torch.add(
                torch.mul(safeV, logP),
                torch.mul(torch.sub(torch.tensor(1.0f, safeV.options()), safeV), log1MinusP)
        );

        // 步骤4：计算归一化常数的对数 log(C(p))
        Tensor twoTensor = torch.tensor(2.0f, safeProbs.options());
        Tensor twoPMinus1 = torch.sub(torch.mul(safeProbs, twoTensor), torch.tensor(1.0f, safeProbs.options()));
        Tensor logRatio = torch.log(torch.div(safeProbs, oneMinusP).clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e6))));

        // 处理p=0.5的情况（log(C(p))=log(2)）
        Tensor mask = torch.abs(twoPMinus1).lt(new Scalar(1e-8));
        Tensor logC = torch.where(
                mask,
                torch.tensor(Math.log(2.0f), safeProbs.options()),
                torch.div(logRatio, twoPMinus1).log() // log(C(p)) = log( log(p/(1-p))/(2p-1) )
        );

        // 步骤5：完整对数概率 = logCore + logC
        Tensor logProb = torch.add(logCore, logC);

        // 释放所有临时张量
        safeProbs.close();
        oneMinusP.close();
        safeV.close();
        logP.close();
        log1MinusP.close();
        logCore.close();
        twoTensor.close();
        twoPMinus1.close();
        logRatio.close();
        mask.close();
        logC.close();

        return logProb;
    }

    @Override
    public Tensor mean() {
        // PyTorch: mean = p/(2p-1) + 1/(log(1-p)-log(p))   ;  0.5 when p≈0.5
        Tensor safeProbs = probs.clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));
        Tensor oneTensor = torch.tensor(1.0f, safeProbs.options());
        Tensor twoTensor = torch.tensor(2.0f, safeProbs.options());
        Tensor twoPMinus1 = torch.sub(torch.mul(safeProbs, twoTensor), oneTensor);
        Tensor oneMinusP = torch.sub(oneTensor, safeProbs);

        Tensor term1 = torch.div(safeProbs, twoPMinus1);
        Tensor log1MinusP = torch.log(oneMinusP);
        Tensor logP = torch.log(safeProbs);
        Tensor denom = torch.sub(log1MinusP, logP); // log(1-p)-log(p)
        Tensor term2 = torch.div(oneTensor, denom);

        Tensor mask = torch.abs(twoPMinus1).lt(new Scalar(1e-8));
        Tensor mean = torch.where(
                mask,
                torch.tensor(0.5f, safeProbs.options()),
                torch.add(term1, term2)
        );

        safeProbs.close();
        oneTensor.close();
        twoTensor.close();
        twoPMinus1.close();
        oneMinusP.close();
        term1.close();
        log1MinusP.close();
        logP.close();
        denom.close();
        term2.close();
        mask.close();

        return mean;
    }

    @Override
    public Tensor entropy() {
        // 连续伯努利分布熵的数值计算（工程实现，避免复杂积分）
        // 核心逻辑：H = -E[log(f(X;p))] = -∫₀¹ f(x;p) * (logC + xlogp + (1-x)log(1-p)) dx
        Tensor safeProbs = probs.clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));
        Tensor oneTensor = torch.tensor(1.0f, safeProbs.options());
        Tensor twoTensor = torch.tensor(2.0f, safeProbs.options());
        Tensor twoPMinus1 = torch.sub(torch.mul(safeProbs, twoTensor), oneTensor);
        Tensor oneMinusP = torch.sub(oneTensor, safeProbs);

        // 计算log(C(p))
        Tensor logRatio = torch.log(torch.div(safeProbs, oneMinusP).clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1e6))));
        Tensor mask = torch.abs(twoPMinus1).lt(new Scalar(1e-8));
        Tensor logC = torch.where(
                mask,
                torch.tensor(Math.log(2.0f), safeProbs.options()),
                torch.div(logRatio, twoPMinus1).log()
        );

        // 计算E[xlogp + (1-x)log(1-p)] = mean*logp + (1-mean)*log(1-p)
        Tensor mean = this.mean();
        Tensor logP = torch.log(safeProbs);
        Tensor log1MinusP = torch.log(oneMinusP);
        Tensor eLogCore = torch.add(
                torch.mul(mean, logP),
                torch.mul(torch.sub(oneTensor, mean), log1MinusP)
        );

        // 熵 = - (logC + E[xlogp + (1-x)log(1-p)])
        Tensor entropy = torch.neg(torch.add(logC, eLogCore));

        // 释放临时张量
        safeProbs.close();
        oneTensor.close();
        twoTensor.close();
        twoPMinus1.close();
        oneMinusP.close();
        logRatio.close();
        mask.close();
        logC.close();
        mean.close();
        logP.close();
        log1MinusP.close();
        eLogCore.close();

        return entropy;
    }

    // 资源释放：实现AutoCloseable
    @Override
    public void close() {
        probs.close();
        // 释放预定义Scalar

    }
}
