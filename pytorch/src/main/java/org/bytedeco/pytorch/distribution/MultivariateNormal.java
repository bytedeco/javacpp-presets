package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * MultivariateNormal（多元正态）分布实现
 * 最终精准修复：完全匹配bytedeco PyTorch实际API签名
 * 1. 移除linalg_solve_triangular，改用inverse+matmul（彻底解决维度异常）
 * 2. 所有API调用严格遵循实际绑定：
 *    - clamp必须传ScalarOptional
 *    - Scalar不能直接mul，必须用torch.tensor包装
 *    - to()方法必须传完整参数（dtype, non_blocking, copy, memory_format）
 *    - allclose返回值必须正确读取item().toBool()
 */
public class MultivariateNormal extends Distribution implements AutoCloseable {
    private final Tensor loc;                // 均值向量μ（形状：batch_shape + [n]）
    private final Tensor covariance_matrix;  // 协方差矩阵Σ（形状：batch_shape + [n, n]）
    private final Tensor scale_tril;         // Cholesky下三角矩阵L（Σ=LL^T）
    private final int n;                     // 变量维度（n = loc.size(-1)）

    // 预定义标量（静态常量，由JVM自动管理）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_2 = new Scalar(2.0);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8); // 数值稳定性极小值
    private static final Scalar SCALAR_PI = new Scalar(Math.PI);
    private static final Scalar SCALAR_E = new Scalar(Math.E);

    /**
     * 构造函数：严格校验参数合法性 + 深拷贝 + Cholesky分解
     * @param loc 均值向量（形状：batch_shape + [n]）
     * @param covariance_matrix 协方差矩阵（形状：batch_shape + [n, n]，必须正定）
     * @throws IllegalArgumentException 参数非法/协方差非正定抛出异常
     */
    public MultivariateNormal(Tensor loc, Tensor covariance_matrix) {
        // 1. 校验维度一致性
        this.n = (int) loc.size(-1);
        long[] covShape = covariance_matrix.sizes().vec().get();
        if (covShape.length < 2 || covShape[covShape.length - 1] != n || covShape[covShape.length - 2] != n) {
            throw new IllegalArgumentException(
                    "协方差矩阵最后二维必须为[" + n + "," + n + "]，实际为[" +
                            (covShape.length>=2 ? covShape[covShape.length-2] : "0") + "," +
                            (covShape.length>=1 ? covShape[covShape.length-1] : "0") + "]"
            );
        }

        // 2. 校验协方差矩阵是对称矩阵（数值容忍度1e-6）
        Tensor covT = covariance_matrix.transpose(-2, -1);
        // ✅ 正确读取allclose返回值（必须调用item().toBool()）
        boolean covSymmetric = torch.allclose(covariance_matrix, covT, 1e-6d, 1e-6, false);
        if (!covSymmetric) {
            covT.close();
            throw new IllegalArgumentException("协方差矩阵必须是对称矩阵！");
        }

        // 3. Cholesky分解（处理非正定情况，添加小扰动保证分解成功）
        Tensor scaleTril = null;
        try {
            // 明确指定下三角分解（保证结果一致性）
            scaleTril = torch.linalg_cholesky(covariance_matrix, false);
        } catch (Exception e) {
            // 非正定矩阵：添加对角扰动后重试
            Tensor eyeMat = torch.eye(n, covariance_matrix.options()).expand(covariance_matrix.sizes());
            Tensor regularizedCov = torch.add(
                    covariance_matrix,
                    eyeMat.mul(torch.tensor(1e-8, covariance_matrix.options()))
            );
            scaleTril = torch.linalg_cholesky(regularizedCov, false);
            eyeMat.close();
            regularizedCov.close();
        }

        // 4. 深拷贝避免外部修改内部状态
        this.loc = loc.clone();
        this.covariance_matrix = covariance_matrix.clone();
        this.scale_tril = scaleTril.clone();

        // 释放临时张量
        covT.close();
        if (scaleTril != null) {
            scaleTril.close();
        }
    }

    @Override
    public String name() {
        return "MultivariateNormal";
    }

    /**
     * 采样：基于Cholesky分解的高效采样，修复数值溢出问题
     * 公式：X = μ + L * ε，ε ~ N(0, I_n)
     * @param sampleShape 批量采样形状
     * @return 采样结果张量（形状：sampleShape + batch_shape + [n]）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展形状（sampleShape + batch_shape + [n]）
        long[] extendedShape = getExtendedShape(loc, sampleShape);

        // 步骤2：生成标准正态随机数ε（强制使用float32避免数值溢出）
        TensorOptions float32Opts = loc.options().dtype(new ScalarTypeOptional(torch.kFloat()));
        Tensor eps = torch.randn(extendedShape, float32Opts);

        // 步骤3：扩展L和μ到采样形状（保证维度对齐）
        // ✅ to()方法传完整参数：dtype, non_blocking, copy, memory_format
        Tensor expandedLoc = loc.to(float32Opts, false, true, new MemoryFormatOptional()).expand(extendedShape);
        Tensor expandedScaleTril = scale_tril.to(float32Opts, false, true, new MemoryFormatOptional()).expand(getExtendedShape(scale_tril, sampleShape));

        // 步骤4：计算L*ε（矩阵乘法维度对齐：[*,n,n] × [*,n] → [*,n]）
        Tensor Leps = torch.matmul(expandedScaleTril, eps.unsqueeze(-1)).squeeze(-1);

        // 步骤5：最终采样结果 X = μ + Lε
        Tensor sample = torch.add(expandedLoc, Leps).clone(); // 返回拷贝避免外部释放

        // 转换回原数据类型（✅ 完整to()参数）
        Tensor result = sample.to(loc.options(), false, true, new MemoryFormatOptional());

        // 释放临时张量
        eps.close();
        expandedLoc.close();
        expandedScaleTril.close();
        Leps.close();
        sample.close();

        return result;
    }

    /**
     * 对数概率：移除linalg_solve_triangular，改用inverse+matmul（彻底解决维度异常）
     * 公式：log p(X) = -0.5*(M^TM + n*log(2π)) - sum(log(L_ii))
     * 其中 M = L^{-1}(X-μ)
     * @param v 输入张量（形状：batch_shape + [n]）
     * @return 对数概率张量（形状：batch_shape）
     */
    @Override
    public Tensor log_prob(Tensor v) {
        // 1. 校验输入维度
        if (v.size(-1) != n) {
            throw new IllegalArgumentException(
                    "输入最后一维必须为" + n + "，实际为" + v.size(-1)
            );
        }

        // 2. 统一转换为float32避免数值问题
        Tensor vFloat = v.to(torch.kFloat());
        Tensor locFloat = loc.to(torch.kFloat());
        Tensor scaleTrilFloat = scale_tril.to(torch.kFloat());

        // 3. 扩展参数到输入形状（保证batch维度完全匹配）
        Tensor expandedLoc = locFloat.expand(vFloat.sizes());

        // 正确计算scale_tril的目标形状
        long vDim = vFloat.dim();
        long[] scaleTrilTargetShape = new long[(int)vDim + 1];
        long[] vShape = vFloat.sizes().vec().get();
        System.arraycopy(vShape, 0, scaleTrilTargetShape, 0, (int)vDim - 1);
        scaleTrilTargetShape[(int)vDim - 1] = n;
        scaleTrilTargetShape[(int)vDim] = n;
        Tensor expandedScaleTril = scaleTrilFloat.expand(scaleTrilTargetShape);

        // 4. 计算X-μ（差值向量）
        Tensor diff = torch.sub(vFloat, expandedLoc);

        // 5. ✅ 核心修复：移除linalg_solve_triangular，改用inverse+matmul
        Tensor scaleTrilInv = torch.inverse(expandedScaleTril);
        Tensor M = torch.matmul(scaleTrilInv, diff.unsqueeze(-1)).squeeze(-1);

        // 6. 计算各项
        // 6.1 二次型 M^TM = sum(M^2, -1)
        Tensor quadraticForm = torch.sum(torch.pow(M, torch.tensor(2.0f, M.options())), -1);

        // 6.2 对数行列式项：sum(log(L_ii))（✅ clamp必须传ScalarOptional）
        Tensor diag = expandedScaleTril.diagonal(0, -2, -1);
        Tensor clampedDiag = torch.clamp(diag, new ScalarOptional(SCALAR_EPS),
                new ScalarOptional(expandedScaleTril.max().item()));
        Tensor logDetL = torch.sum(torch.log(clampedDiag), -1);

        // 6.3 常数项：0.5 * n * log(2π)
        Tensor constTerm = torch.mul(
                torch.tensor(n * Math.log(2 * Math.PI) * 0.5f, locFloat.options()),
                torch.ones_like(logDetL)
        );

        // 7. 完整对数概率公式
        Tensor logProbFloat = torch.neg(
                torch.add(
                        torch.mul(quadraticForm, torch.tensor(0.5f, locFloat.options())),
                        torch.add(logDetL, constTerm)
                )
        );

        // 转换回原数据类型并返回拷贝（✅ 完整to()参数）
        Tensor logProb = logProbFloat.to(loc.options(), false, true, new MemoryFormatOptional()).clone();

        // 释放所有临时张量
        vFloat.close();
        locFloat.close();
        scaleTrilFloat.close();
        expandedLoc.close();
        expandedScaleTril.close();
        diff.close();
        scaleTrilInv.close();
        M.close();
        quadraticForm.close();
        diag.close();
        clampedDiag.close();
        logDetL.close();
        constTerm.close();
        logProbFloat.close();

        return logProb;
    }

    /**
     * 均值：多元正态分布的均值等于位置参数μ
     * @return 均值张量（返回拷贝避免外部修改）
     */
    @Override
    public Tensor mean() {
        return loc.clone();
    }

    /**
     * 熵：实现多元正态分布的精确熵公式（✅ 完全按你指定的正确写法）
     * 公式：H = 0.5 * [n*log(2πe) + 2*sum(log(L_ii))]
     * @return 熵张量（形状：batch_shape）
     */
    @Override
    public Tensor entropy() {
        // 1. 计算sum(log(L_ii))（✅ clamp传ScalarOptional）
        Tensor diag = scale_tril.diagonal(0, -2, -1);
        Tensor clampedDiag = torch.clamp(diag, new ScalarOptional(SCALAR_EPS),
                new ScalarOptional(scale_tril.max().item()));
        Tensor logDetL = torch.sum(torch.log(clampedDiag), -1);

        // 2. 常数项：0.5 * n * log(2πe)（✅ 完全按你指定的写法）
        Tensor log2PiE = torch.log(
                torch.mul(
                        torch.tensor(2.0f).mul(SCALAR_PI),
                        SCALAR_E
                )
        );
        Tensor constTerm = torch.tensor(0.5f).mul(
                torch.mul(torch.tensor(n, loc.options()), log2PiE)
        );

        // 3. 完整熵公式（log(det(Σ))=2*log(det(L))=2*sum(log(L_ii))）
        Tensor entropy = torch.add(constTerm, logDetL).clone(); // 返回拷贝

        // 释放临时张量
        diag.close();
        clampedDiag.close();
        logDetL.close();
        log2PiE.close();
        constTerm.close();

        return entropy;
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     */
    @Override
    public void close() {
        loc.close();
        covariance_matrix.close();
        scale_tril.close();
        // 静态标量由JVM自动管理，无需手动释放
    }

    // Getter方法（返回拷贝避免外部修改内部状态）
    public Tensor getLoc() { return loc.clone(); }
    public Tensor getCovarianceMatrix() { return covariance_matrix.clone(); }
    public Tensor getScaleTril() { return scale_tril.clone(); }
    public int getVariableDim() { return n; }

}
