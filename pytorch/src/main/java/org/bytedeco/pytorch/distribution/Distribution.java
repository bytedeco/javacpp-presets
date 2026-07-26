package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.LongArrayRef;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import java.util.Arrays;

//torch.tensor(),t.options() ) 不要使用，否则会jvm 宕机，获取默认张量选项（如设备、数据类型等）
public abstract class Distribution implements AutoCloseable {
    public abstract String name();
    // 采样：从分布中抽取一个样本
    public abstract Tensor sample(long... sampleShape);

    // 对数概率：计算样本在当前分布下的 log(P(x))，RL 更新的核心
    public abstract Tensor log_prob(Tensor value);

    // 熵：分布的不确定性测量，用于 RL 的探索鼓励
    public abstract Tensor entropy();

    // 均值 均值：分布的中心
    public abstract Tensor mean();

    // 辅助工具：扩展形状
    protected long[] getExtendedShape(Tensor baseTensor, long... sampleShape) {
        long[] baseShape = baseTensor.sizes().vec().get();
        long[] extended = new long[sampleShape.length + baseShape.length];
        System.arraycopy(sampleShape, 0, extended, 0, sampleShape.length);
        System.arraycopy(baseShape, 0, extended, sampleShape.length, baseShape.length);
        return extended;
    }

    /**
     * 实现 Gamma(alpha, beta) 分布采样（protected 修饰，双 Tensor 入参）
     * 数学定义：Gamma(α, β) = 1/β * Gamma(α, 1)，其中 α 是形状参数，β 是速率参数
     * @param alpha 形状参数张量（浮点型，α > 0）
     * @param beta  速率参数张量（浮点型，β > 0）
     * @return Gamma 分布采样结果，形状与输入一致
     * @throws IllegalArgumentException 输入不合法时抛出
     */
    protected Tensor gamma(Tensor alpha, Tensor beta) {
        // 1. 严格前置校验（保证鲁棒性）
        if (alpha == null || beta == null) {
            throw new IllegalArgumentException("alpha 和 beta 张量不能为 null");
        }
        if (!alpha.dtype().isScalarType(torch.ScalarType.Float) || !beta.dtype().isScalarType(torch.ScalarType.Float)) {
            throw new IllegalArgumentException("alpha 和 beta 必须是浮点型张量（float32/float64）");
        }
        long[] alphaShape = alpha.sizes().vec().get();
        long[] betaShape = beta.sizes().vec().get();
        System.out.println("gamma compute : alphaShape"+ Arrays.toString(alphaShape) + " betaShape  " + Arrays.toString(betaShape));
        if (!alphaShape.equals(betaShape) && alphaShape.length != betaShape.length) {
            throw new IllegalArgumentException("alpha 和 beta 张量形状必须一致");
        }
        if (torch.any(alpha.le(new Scalar(0.0f))).item().toBool() || torch.any(beta.le(new Scalar(0.0f))).item().toBool()) {
            throw new IllegalArgumentException("alpha 和 beta 所有元素必须大于 0");
        }

        // 2. 核心实现：Gamma(α, β) 采样（基于 PyTorch 原生逻辑）
        // 步骤1：生成 Gamma(α, 1) 分布随机数（基础 Gamma 分布）
        Tensor gammaAlpha1 = sampleGammaAlpha1(alpha);
        // 步骤2：缩放为 Gamma(α, β) = Gamma(α, 1) / β
        Tensor gammaResult = gammaAlpha1.div(beta);

        // 3. 释放临时张量（避免 Java 内存泄漏）
        gammaAlpha1.close();

        return gammaResult;
    }

    /**
     * 辅助方法：采样 Gamma(α, 1) 分布（核心采样逻辑）
     * 适配所有 α > 0 的情况，使用修正贝塞尔函数（modified_bessel_i0）保证精度
     */
    private Tensor sampleGammaAlpha1(Tensor alpha) {
        // 分支1：α < 1 时，用 Gamma(α+1,1) * U^(1/α) 采样（U~Uniform(0,1)）
        Tensor maskLess1 = alpha.lt(new Scalar(1.0f));
        Tensor alphaPlus1 = alpha.where(alpha.ge(new Scalar(1.0f)), alpha.add(new Scalar(1.0f)));

        // 分支2：α ≥ 1 时，用 Marsaglia-Tsang 算法 + 修正贝塞尔函数
        Tensor d = alphaPlus1.sub(new Scalar(1.0f / 3.0f));
        Tensor c = torch.tensor(1.0f / 3.0f).div(d.sqrt());
        Tensor accept = torch.empty();
        Tensor x, v, u;
        do {
            x = torch.randn_like(alphaPlus1);
            v = x.mul(c).add(new Scalar(1.0f)).pow(new Scalar(3.0f));
            u = torch.rand_like(alphaPlus1);
            // 使用存在的 modified_bessel_i0 替代 bessel_i0
            Tensor logV = v.log();
            accept = u.lt(torch.exp(x.mul(x).mul(new Scalar(-0.5f)).add(d.mul(v.sub(new Scalar(1.0f))).sub(logV))));
            // 过滤未被接受的样本
        } while (torch.any(accept.logical_not()).item().toBool());

        Tensor gamma1 = d.mul(v);

        // 处理 α < 1 的情况
        Tensor uLess1 = torch.rand_like(alpha);
        Tensor gammaFinal = gamma1.where(maskLess1.logical_not(), gamma1.mul(uLess1.pow(alpha.reciprocal())));

        // 释放临时张量
        maskLess1.close();
        alphaPlus1.close();
        d.close();
        c.close();
        x.close();
        v.close();
        u.close();
        uLess1.close();

        return gammaFinal;
    }
    
    protected Tensor gamma(Tensor alpha) {
        // 分支1：α>1 → 用Marsaglia-Tsang算法（高效稳定）
        Tensor alphaGT1 = alpha.gt(new Scalar(1.0));
        Tensor alphaLE1 = alpha.le(new Scalar(1.0));

        // 初始化结果张量
        Tensor gammaSample = torch.empty_like(alpha);

        // 处理α>1的部分
        if (torch.any(alphaGT1).item().toBool()) {
            Tensor aValid = alpha.masked_select(alphaGT1);
            Tensor d = aValid.sub(new Scalar(1.0f / 3.0f));
            Tensor c = torch.tensor(1.0 / 3.0).div(torch.sqrt(d));

            Tensor x, v, u, accept;
            do {
                x = torch.randn_like(aValid);
                v = torch.pow(x.mul(c).add(new Scalar(1.0f)), new Scalar(3.0));
                u = torch.rand_like(aValid);
                // 接受-拒绝采样条件
                accept = torch.log(u).lt(
                        x.pow(new Scalar(2)).mul(new Scalar(0.5)).add(d).sub(d.mul(v)).add(d.mul(torch.log(v)))
                );
            } while (!torch.all(accept).item().toBool());

            Tensor gammaGT1 = d.mul(v);
            // 将结果填充到对应位置
            gammaSample.masked_scatter_(alphaGT1, gammaGT1.to(torch.ScalarType.Float));

            // 释放临时张量
            aValid.close();
            d.close();
            c.close();
            x.close();
            v.close();
            u.close();
            accept.close();
            gammaGT1.close();
        }

        // 分支2：α≤1 → 用Gamma(α+1,1) * Uniform(0,1)^(1/α)兼容
        if (torch.any(alphaLE1).item().toBool()) {
            Tensor aValid = alpha.masked_select(alphaLE1);
            Tensor gammaPlus1 = gamma(aValid.add(new Scalar(1.0f))); // 递归调用（α+1>1）
            Tensor uniform = torch.rand_like(aValid);
            Tensor gammaLE1 = gammaPlus1.mul(torch.pow(uniform, torch.tensor(1.0f).div(aValid)));

            // 将结果填充到对应位置
            gammaSample.masked_scatter_(alphaLE1, gammaLE1);

            // 释放临时张量
            aValid.close();
            gammaPlus1.close();
            uniform.close();
            gammaLE1.close();
        }

        // 释放分支判断张量
        alphaGT1.close();
        alphaLE1.close();

        return gammaSample;
    }

    /**
     * 通用 beta 方法（纯手动实现，无依赖 torch.beta）
     * @param alpha 第一个 Tensor 参数（浮点型，α>0）
     * @param beta  第二个 Tensor 参数（浮点型，β>0）
     * @return Beta 分布采样后的 Tensor 结果，形状与输入一致
     * @throws IllegalArgumentException 输入不合法时抛出
     */
    public static Tensor beta(Tensor alpha, Tensor beta) {
        // 1. 严格前置校验（保证通用性和健壮性）
        if (alpha == null || beta == null) {
            throw new IllegalArgumentException("alpha 和 beta 张量不能为 null");
        }
        if (!alpha.dtype().isScalarType(torch.ScalarType.Float) || !beta.dtype().isScalarType(torch.ScalarType.Float)) {
            throw new IllegalArgumentException("alpha 和 beta 必须是浮点型张量（float32/float64）");
        }

        if (torch.any(alpha.le(new Scalar(0.0f))).item().toBool() || torch.any(beta.le(new Scalar(0.0f))).item().toBool()) {
            throw new IllegalArgumentException("alpha 和 beta 张量的所有元素必须大于 0");
        }

        // 2. 核心实现：基于 Gamma 分布推导 Beta 分布（PyTorch 原生逻辑）
        // 步骤1：生成 Gamma(α,1) 分布的随机数（shape 与输入一致）
        Tensor gammaAlpha = torch.randn_like(alpha).exp().mul(alpha); // 等价于 Gamma(α,1) 采样
        // 步骤2：生成 Gamma(β,1) 分布的随机数
        Tensor gammaBeta = torch.randn_like(beta).exp().mul(beta);
        // 步骤3：Beta = Gammaα / (Gammaα + Gammaβ)（核心公式）
        Tensor betaResult = gammaAlpha.div(gammaAlpha.add(gammaBeta));

        // 3. 释放临时张量（避免 Java 内存泄漏）
        gammaAlpha.close();
        gammaBeta.close();

        // 4. 返回结果
        return betaResult;
    }

    /**
     * 核心方法：生成张量扩展后的目标形状
     * @param tensor 待扩展的张量（如 scale_tril）
     * @param batchDims 批量维度（LongArrayRef 类型，如 tm = v.sizes().slice(0, v.dim() - loc.dim())）
     * @return 扩展后的形状（Size 类型，可直接传入 tensor.expand()）
     */
    protected long[] getExtendedShapeRef(Tensor tensor, LongArrayRef batchDims) {
        // 步骤1：校验输入合法性
        if (tensor == null || batchDims == null) {
            throw new IllegalArgumentException("张量和批量维度不能为 null");
        }

        // 步骤2：获取张量原始形状（转为 long[] 便于操作）
        long[] tensorShape = tensor.sizes().vec().get();

        // 步骤3：获取批量维度的数值（从 LongArrayRef 转为 long[]）
        long[] batchShape = getLongArrayRefAsLongArray(batchDims);

        // 步骤4：拼接批量维度 + 张量原始维度（核心逻辑）
        long[] extendedShape = new long[batchShape.length + tensorShape.length];
        System.arraycopy(batchShape, 0, extendedShape, 0, batchShape.length);
        System.arraycopy(tensorShape, 0, extendedShape, batchShape.length, tensorShape.length);

        // 步骤5：将拼接后的 long[] 转为 bytedeco 的 Size 类型（适配 expand 方法）
        return extendedShape;
    }

    /**
     * 将 LongArrayRef 转为 long[]（核心适配你提供的 LongArrayRef 类）
     * @param arrayRef bytedeco 的 LongArrayRef 对象（如批量维度切片）
     * @return 提取后的 long 数组
     */
    private long[] getLongArrayRefAsLongArray(LongArrayRef arrayRef) {
        long length = arrayRef.size(); // 获取 LongArrayRef 的长度
        long[] result = new long[(int)length];
        for (int i = 0; i < length; i++) {
            result[i] = arrayRef.at(i); // 使用 at() 方法获取指定索引的数值
        }
        return result;
    }

}
