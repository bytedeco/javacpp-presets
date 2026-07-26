package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

public class Dirichlet extends Distribution implements AutoCloseable {
    private final Tensor alpha;  // 分布参数α（所有元素>0）
    private final long numCategories; // 类别数K（最后一维长度）

    // 预定义常量（复用避免重复创建）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值
    private static final ScalarTypeOptional SCALAR_TYPE_OPT = new ScalarTypeOptional(); // 复用空Optional

    // 构造函数：校验参数合法性 + 深拷贝
    public Dirichlet(Tensor alpha) {
        // 校验α所有元素>0（狄利克雷分布核心约束）
        Tensor alphaLe0 = torch.lt(alpha, torch.tensor(0.0f, alpha.options()));
        if (torch.any(alphaLe0).item().toBool()) {
            alphaLe0.close();
            throw new IllegalArgumentException("狄利克雷分布alpha所有元素必须大于0！");
        }
        alphaLe0.close();

        // 深拷贝避免外部修改内部状态
        this.alpha = alpha.clone();
        this.numCategories = alpha.size(-1); // 记录类别数K

        // 校验类别数≥2（狄利克雷分布要求K≥2）
        if (numCategories < 2) {
            throw new IllegalArgumentException("狄利克雷分布类别数必须≥2！");
        }
    }

    @Override
    public String name() {
        return "Dirichlet";
    }

    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：复用父类方法扩展采样形状
        long[] extendedShape = getExtendedShape(alpha, sampleShape);
        Tensor expandedAlpha = alpha.expand(extendedShape); // 扩展alpha到批量形状

        // 步骤2：生成Uniform(ε,1-ε)随机数（避免log(0)/log(1)）
        Tensor u = torch.rand(extendedShape, alpha.options())
                .clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0 - 1e-8)));

        // 步骤3：Gamma(alpha_i, 1)采样（数值稳定版）
        Tensor oneTensor = torch.tensor(1.0f, u.options());
        Tensor oneMinusU = torch.sub(oneTensor, u);
        Tensor logOneMinusU = torch.log(oneMinusU);
        Tensor gammaSamples = torch.mul(expandedAlpha, torch.neg(logOneMinusU)); // Gamma(alpha,1) = -alpha*log(1-u)

        // 步骤4：归一化（按最后一维求和，保持维度）
        Tensor gammaSum = torch.sum(gammaSamples, new long[]{-1}, true, SCALAR_TYPE_OPT);
        // 数值稳定性：避免sum=0（理论上不可能，兜底处理）
        Tensor gammaSumSafe = torch.where(
                torch.eq(gammaSum, torch.tensor(0.0f, gammaSum.options())),
                torch.ones_like(gammaSum),
                gammaSum
        );
        Tensor dirichletSample = torch.div(gammaSamples, gammaSumSafe);

        // 释放所有临时张量
        expandedAlpha.close();
        u.close();
        oneTensor.close();
        oneMinusU.close();
        logOneMinusU.close();
        gammaSamples.close();
        gammaSum.close();
        gammaSumSafe.close();

        return dirichletSample;
    }

    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：校验输入合法性
        // 1.1 校验v所有元素>0
        Tensor vLe0 = torch.lt(v, torch.tensor(0.0f, v.options()));
        if (torch.any(vLe0).item().toBool()) {
            vLe0.close();
            throw new IllegalArgumentException("log_prob输入v所有元素必须大于0！");
        }
        // 1.2 校验v最后一维求和≈1；若在容差内则重新归一化
        Tensor vSum = torch.sum(v, new long[]{-1}, true, SCALAR_TYPE_OPT);
        Tensor vSumInvalid = torch.abs(torch.sub(vSum, torch.tensor(1.0f, vSum.options()))).gt(new Scalar(1e-4));
        if (torch.any(vSumInvalid).item().toBool()) {
            vLe0.close();
            vSum.close();
            vSumInvalid.close();
            throw new IllegalArgumentException("log_prob输入v最后一维求和必须等于1（允许1e-4误差）！");
        }
        vLe0.close();
        vSumInvalid.close();

        // 步骤2：重新归一化 + 数值稳定性
        Tensor vNorm = torch.div(v, vSum);
        vSum.close();
        Tensor safeV = vNorm.clamp(new ScalarOptional(new Scalar(1e-8)), new ScalarOptional(new Scalar(1.0f)));
        vNorm.close();
        Tensor safeAlpha = alpha.expand(safeV.sizes()); // 扩展alpha到v的形状

        // 步骤3：计算对数概率公式
        // term1 = sum((alpha-1)*log(v), dim=-1)
        Tensor alphaMinus1 = torch.sub(safeAlpha, torch.tensor(1.0f, safeAlpha.options()));
        Tensor logV = torch.log(safeV);
        Tensor term1 = torch.sum(torch.mul(alphaMinus1, logV), -1);

        // term2 = lgamma(sum(alpha, dim=-1))
        Tensor alphaSum = torch.sum(safeAlpha, -1);
        Tensor term2 = torch.lgamma(alphaSum);

        // term3 = sum(lgamma(alpha), dim=-1)
        Tensor term3 = torch.sum(torch.lgamma(safeAlpha), -1);

        // 完整对数概率：term1 + term2 - term3
        Tensor logProb = torch.add(torch.sub(term1, term3), term2);

        // 释放临时张量
        safeV.close();
        safeAlpha.close();
        alphaMinus1.close();
        logV.close();
        term1.close();
        alphaSum.close();
        term2.close();
        term3.close();

        return logProb;
    }

    @Override
    public Tensor entropy() {
        // 熵公式：
        // H = lgamma(sum(alpha)) - sum(lgamma(alpha)) - (sum(alpha)-K)*digamma(sum(alpha)) - sum((alpha-1)*digamma(alpha))
        Tensor alphaSum = torch.sum(alpha, -1); // sum(alpha)
        Tensor kTensor = torch.tensor(numCategories, alpha.options()); // 类别数K

        // term1 = lgamma(sum(alpha)) - sum(lgamma(alpha))
        Tensor term1 = torch.sub(torch.lgamma(alphaSum), torch.sum(torch.lgamma(alpha), -1));

        // term2 = (sum(alpha)-K) * digamma(sum(alpha))
        Tensor alphaSumMinusK = torch.sub(alphaSum, kTensor);
        Tensor term2 = torch.mul(alphaSumMinusK, torch.digamma(alphaSum));

        // term3 = sum((alpha-1)*digamma(alpha), dim=-1)
        Tensor alphaMinus1 = torch.sub(alpha, torch.tensor(1.0f, alpha.options()));
        Tensor term3 = torch.sum(torch.mul(alphaMinus1, torch.digamma(alpha)), -1);

        // 完整熵：term1 - term2 - term3
        Tensor entropy = torch.sub(torch.sub(term1, term2), term3);

        // 释放临时张量
        alphaSum.close();
        kTensor.close();
        term1.close();
        alphaSumMinusK.close();
        term2.close();
        alphaMinus1.close();
        term3.close();

        return entropy;
    }

    @Override
    public Tensor mean() {
        // 均值公式：alpha / sum(alpha, dim=-1, keepdim=True)
        Tensor alphaSum = torch.sum(alpha, new long[]{-1}, true, SCALAR_TYPE_OPT);
        // 数值稳定性：避免sum=0（理论上不可能，兜底处理）
        Tensor alphaSumSafe = torch.where(
                torch.eq(alphaSum, torch.tensor(0.0f, alphaSum.options())),
                torch.ones_like(alphaSum),
                alphaSum
        );
        Tensor mean = torch.div(alpha, alphaSumSafe);

        // 释放临时张量
        alphaSum.close();
        alphaSumSafe.close();

        return mean;
    }

    // 资源释放：实现AutoCloseable
    @Override
    public void close() {
        alpha.close();
        // 释放预定义常量

    }
}
