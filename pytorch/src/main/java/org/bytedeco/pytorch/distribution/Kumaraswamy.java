package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Kumaraswamy extends Distribution implements AutoCloseable {
    private final Tensor a;  // 形状参数a（必须>0）
    private final Tensor b;  // 形状参数b（必须>0）

    // 预定义标量（复用避免重复创建）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值
    private static final Scalar SCALAR_NEG_INF = new Scalar(Double.NEGATIVE_INFINITY);

    // 构造函数：校验参数合法性 + 深拷贝
    public Kumaraswamy(Tensor a, Tensor b) {
        // 校验形状参数a>0
        Tensor aLe0 = torch.lt(a, torch.tensor(0.0f, a.options()));
        // 校验形状参数b>0
        Tensor bLe0 = torch.lt(b, torch.tensor(0.0f, b.options()));
        Tensor paramInvalid = torch.logical_or(aLe0, bLe0);

        if (torch.any(paramInvalid).item().toBool()) {
            aLe0.close();
            bLe0.close();
            paramInvalid.close();
            throw new IllegalArgumentException("库玛斯瓦米分布a和b必须大于0！");
        }

        // 释放校验临时张量
        aLe0.close();
        bLe0.close();
        paramInvalid.close();

        // 深拷贝避免外部修改内部状态
        this.a = a.clone();
        this.b = b.clone();
    }

    @Override
    public String name() {
        return "Kumaraswamy";
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：复用父类方法扩展采样形状
        long[] extendedShape = getExtendedShape(a, sampleShape);
        // 扩展参数到批量形状（保证维度对齐）
        Tensor expandedA = a.expand(extendedShape);
        Tensor expandedB = b.expand(extendedShape);

        // 步骤2：生成受限Uniform(ε,1-ε)随机数（避免数值溢出）
        Tensor u = torch.rand(extendedShape, a.options())
                .clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));

        // 步骤3：库玛斯瓦米采样公式：[1 - (1-U)^(1/b)]^(1/a)
        Tensor oneTensor = torch.tensor(1.0f, u.options());
        Tensor oneMinusU = torch.sub(oneTensor, u); // 1-U

        // 数值稳定版(1-U)^(1/b)：避免(1-U)→0时幂运算溢出
        Tensor invB = torch.reciprocal(expandedB); // 1/b
        Tensor term1 = torch.pow(
                torch.clamp(oneMinusU, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(oneMinusU.max().item())),
                invB
        );

        Tensor oneMinusTerm1 = torch.sub(oneTensor, term1); // 1 - (1-U)^(1/b)
        Tensor invA = torch.reciprocal(expandedA); // 1/a

        // 最终采样结果：[1 - (1-U)^(1/b)]^(1/a)
        Tensor sample = torch.pow(
                torch.clamp(oneMinusTerm1, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(oneMinusTerm1.max().item())),
                invA
        );

        // 释放所有临时张量
        expandedA.close();
        expandedB.close();
        u.close();
        oneTensor.close();
        oneMinusU.close();
        invB.close();
        term1.close();
        oneMinusTerm1.close();
        invA.close();

        return sample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：校验输入0<v<1
        Tensor vLe0 = torch.le(v, torch.tensor(0.0f, v.options()));
        Tensor vGe1 = torch.ge(v, torch.tensor(1.0f, v.options()));
        Tensor vInvalid = torch.logical_or(vLe0, vGe1);

        if (torch.any(vInvalid).item().toBool()) {
            vLe0.close();
            vGe1.close();
            vInvalid.close();
            // 返回全-∞张量
            Tensor negInf = torch.full_like(v, new Scalar(Float.NEGATIVE_INFINITY), v.options(), new MemoryFormatOptional());
            return negInf;
        }
        vLe0.close();
        vGe1.close();
        vInvalid.close();

        // 步骤2：扩展参数到v的形状
        Tensor expandedA = a.expand(v.sizes());
        Tensor expandedB = b.expand(v.sizes());
        // 数值稳定性处理
        Tensor safeV = torch.clamp(v, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));

        // 步骤3：计算对数概率公式（严格对齐数学定义）
        // term1 = log(a) + log(b)
        Tensor logA = torch.log(expandedA);
        Tensor logB = torch.log(expandedB);
        Tensor term1 = torch.add(logA, logB);

        // term2 = (a-1) * log(v)
        Tensor aMinus1 = torch.sub(expandedA, torch.tensor(1.0f, expandedA.options()));
        Tensor logV = torch.log(safeV);
        Tensor term2 = torch.mul(aMinus1, logV);

        // term3 = (b-1) * log(1 - v^a)
        Tensor vPowA = torch.pow(safeV, expandedA); // v^a
        Tensor oneMinusVPowA = torch.sub(torch.tensor(1.0f, vPowA.options()), vPowA); // 1 - v^a
        // 数值稳定：避免1 - v^a→0导致log(0)
        Tensor safeOneMinusVPowA = torch.clamp(oneMinusVPowA, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(oneMinusVPowA.max().item()));
        Tensor bMinus1 = torch.sub(expandedB, torch.tensor(1.0f, expandedB.options()));
        Tensor logOneMinusVPowA = torch.log(safeOneMinusVPowA);
        Tensor term3 = torch.mul(bMinus1, logOneMinusVPowA);

        // 完整对数概率：log(a)+log(b) + (a-1)logv + (b-1)log(1-v^a)
        Tensor logProb = torch.add(torch.add(term1, term2), term3);

        // 释放临时张量
        expandedA.close();
        expandedB.close();
        safeV.close();
        logA.close();
        logB.close();
        term1.close();
        aMinus1.close();
        logV.close();
        term2.close();
        vPowA.close();
        oneMinusVPowA.close();
        safeOneMinusVPowA.close();
        bMinus1.close();
        logOneMinusVPowA.close();
        term3.close();

        return logProb;
    }

    @Override
    public Tensor entropy() {
        // 熵公式：H = 1 - 1/a - (ψ(b)+1)/b + (log(a)+log(b))/b
        Tensor oneTensor = torch.tensor(1.0f, a.options());

        // 逐项计算
        Tensor term1 = oneTensor; // 1
        Tensor term2 = torch.neg(torch.reciprocal(a)); // -1/a

        // term3 = -(ψ(b)+1)/b
        Tensor digammaB = torch.digamma(b); // ψ(b)
        Tensor digammaBPlus1 = torch.add(digammaB, oneTensor); // ψ(b)+1
        Tensor term3 = torch.neg(torch.div(digammaBPlus1, b));

        // term4 = (log(a)+log(b))/b
        Tensor logA = torch.log(a);
        Tensor logB = torch.log(b);
        Tensor logAPlusLogB = torch.add(logA, logB);
        Tensor term4 = torch.div(logAPlusLogB, b);

        // 完整熵
        Tensor entropy = torch.add(torch.add(torch.add(term1, term2), term3), term4);

        // 释放临时张量
        oneTensor.close();
        term1.close();
        term2.close();
        digammaB.close();
        digammaBPlus1.close();
        term3.close();
        logA.close();
        logB.close();
        logAPlusLogB.close();
        term4.close();

        return entropy;
    }

    @Override
    public Tensor mean() {
        // 均值公式：b * Γ(1+1/a) * Γ(b) / Γ(1+1/a + b)
        Tensor oneTensor = torch.tensor(1.0f, a.options());

        // 计算1+1/a
        Tensor invA = torch.reciprocal(a); // 1/a
        Tensor onePlusInvA = torch.add(oneTensor, invA); // 1+1/a

        // 计算各项lgamma
        Tensor lgammaOnePlusInvA = torch.lgamma(onePlusInvA); // lgamma(1+1/a)
        Tensor lgammaB = torch.lgamma(b); // lgamma(b)
        Tensor sumTerm = torch.add(onePlusInvA, b); // 1+1/a + b
        Tensor lgammaSumTerm = torch.lgamma(sumTerm); // lgamma(1+1/a + b)

        // 合并lgamma：lgamma(1+1/a) + lgamma(b) - lgamma(1+1/a + b)
        Tensor lgammaCombined = torch.sub(torch.add(lgammaOnePlusInvA, lgammaB), lgammaSumTerm);

        // 指数化得到Beta函数值：exp(lgammaCombined)
        Tensor betaFunc = torch.exp(lgammaCombined);

        // 最终均值：b * Beta(1+1/a, b)
        Tensor mean = torch.mul(b, betaFunc);

        // 释放临时张量
        oneTensor.close();
        invA.close();
        onePlusInvA.close();
        lgammaOnePlusInvA.close();
        lgammaB.close();
        sumTerm.close();
        lgammaSumTerm.close();
        lgammaCombined.close();
        betaFunc.close();

        return mean;
    }

    // 资源释放：实现AutoCloseable
    @Override
    public void close() {
        a.close();
        b.close();
        // 释放预定义标量

    }
}
