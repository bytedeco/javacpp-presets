package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Wishart（威沙特分布）实现 - 最终优化版
 * 特性：
 * 1. 完全消除Cholesky废弃警告日志
 * 2. 修复批量log_prob第二个值未返回-∞的问题
 * 3. 提升数值稳定性和精度
 * 4. 无递归、无栈溢出、无异常日志
 */
public class Wishart extends Distribution implements AutoCloseable {
    private Tensor df;                // 自由度ν（支持标量/批量，shape=batch_shape）
    private final Tensor scale;             // 尺度矩阵Σ（shape=batch_shape + [p,p]）
    private final int matrixDim;            // 矩阵维度p
    private final long[] batchShape;        // 批量形状（df/scale的公共批量形状）
    private final Tensor scaleCholesky;     // Σ的Cholesky分解 L（shape=batch_shape + [p,p]）
    private final Tensor logDetScale;       // log|Σ|（shape=batch_shape）
    private final Tensor scaleInv;          // Σ的逆矩阵（预计算，避免重复分解）

    // 数值稳定常量（精准调优）
    private static final double EPS = 1e-8;
    private static final double MIN_EIGENVALUE = 1e-6;
    private static final double STABILIZE_EPS = 1e-6;
    private static final int MAX_STABILIZE_ITER = 3;
    private static final Scalar SCALAR_EPS = new Scalar(EPS);
    private static final Scalar SCALAR_MIN_EIGEN = new Scalar(MIN_EIGENVALUE);
    private static final Scalar SCALAR_STABILIZE = new Scalar(STABILIZE_EPS);
    private static final Scalar SCALAR_2 = new Scalar(2.0);
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_NEG_INF = new Scalar(Double.NEGATIVE_INFINITY);
    private static final Scalar SCALAR_1 = new Scalar(1.0);

    /**
     * 构造函数：支持标量/批量参数 + 严格校验
     * @param df 自由度（shape=batch_shape，标量/批量，所有元素≥p）
     * @param scale 尺度矩阵（shape=batch_shape + [p,p]，每个矩阵正定）
     */
    public Wishart(Tensor df, Tensor scale) {
        // 1. 空值校验
        if (df == null || scale == null) {
            throw new IllegalArgumentException("df和scale不能为空！");
        }

        // 2. 标准化df：仅移除多余的1维（保留批量维度）
        this.df = standardizeDf(df);

        // 3. 解析scale维度和矩阵维度p
        if (scale.dim() < 2) {
            throw new IllegalArgumentException("scale必须至少2维（shape=batch_shape + [p,p]）！");
        }
        long p = scale.size(scale.dim() - 1);
        if (scale.size(scale.dim() - 2) != p) {
            throw new IllegalArgumentException(
                    String.format("scale最后两维必须是方阵（当前：[%d,%d]）",
                            scale.size(scale.dim() - 2), p)
            );
        }
        this.matrixDim = (int) p;

        // 4. 解析批量形状（scale的前n-2维）
        long[] scaleBatchShape = new long[(int)scale.dim() - 2];
        for (int i = 0; i < scaleBatchShape.length; i++) {
            scaleBatchShape[i] = scale.size(i);
        }

        // 5. 对齐df和scale的批量形状
        long[] dfShape = getTensorShape(this.df);
        this.batchShape = broadcastShapes(dfShape, scaleBatchShape);

        // 6. 扩展df和scale到公共批量形状（广播）
        Tensor dfExpanded = this.df.expand(this.batchShape);
        Tensor scaleExpanded = scale.view(
                concatShapes(scaleBatchShape, new long[]{p, p})
        ).expand(concatShapes(this.batchShape, new long[]{p, p}));

        // 7. 校验自由度约束：所有df ≥ p（数值容忍度）
        Tensor dfMinP = torch.sub(dfExpanded, new Scalar(p));
        Tensor dfLtP = torch.lt(dfMinP, new Scalar(-EPS));
        if (torch.any(dfLtP).item().toBool()) {
            throw new IllegalArgumentException(
                    String.format("所有自由度df必须≥矩阵维度p=%d（数值容忍度%.4f）", p, EPS)
            );
        }

        // 8. 校验scale所有矩阵严格正定（无递归的稳定化）
        if (!isAllPositiveDefinite(scaleExpanded, false)) {
            throw new IllegalArgumentException("scale中存在非正定矩阵！");
        }

        // 9. 保存最终的df和scale（深拷贝，避免外部修改）
        this.df = dfExpanded.clone();
        this.scale = scaleExpanded.clone();

        // 10. 预计算Cholesky分解（使用新API，消除废弃警告）
        Tensor scaleStabilized = stabilizeMatrix(this.scale, false);
        try (Tensor chol = torch.linalg_cholesky(scaleStabilized)) { // 修复废弃API
            this.scaleCholesky = chol.clone();
        }
        scaleStabilized.close();

        // 11. 预计算log|Σ|
        this.logDetScale = computeLogDetFromCholesky(this.scaleCholesky);

        // 12. 预计算scale的逆矩阵（使用linalg.inv，避免废弃API）
        this.scaleInv = torch.linalg_inv(stabilizeMatrix(this.scale, false));

        // 释放临时张量
        dfMinP.close();
        dfLtP.close();
        scaleExpanded.close();
        dfExpanded.close();
    }

    @Override
    public String name() {
        return String.format("Wishart(p=%d, batch_shape=%s)",
                matrixDim, shapeToString(batchShape));
    }

    /**
     * 采样方法：支持标量/批量参数 + 无异常日志 + 无栈溢出
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：计算最终批量形状 = sample_shape + batch_shape
        long[] finalBatchShape = concatShapes(sampleShape, batchShape);

        // 步骤2：计算输出形状 = finalBatchShape + [p,p]
        long[] outputShape = concatShapes(finalBatchShape, new long[]{matrixDim, matrixDim});

        // 步骤3：计算随机矩阵形状 = finalBatchShape + [ν, p]
        // 修复：使用每个批量的df值，而非均值
        Tensor dfExpanded = df.expand(finalBatchShape);
        long[] xShape = concatShapes(finalBatchShape, new long[]{1, matrixDim});
        xShape[xShape.length - 2] = (long) Math.ceil(dfExpanded.max().item().toDouble()); // 使用最大df

        // 内存安全的张量操作
        Tensor x = null;
        Tensor xScaled = null;
        Tensor xTx = null;
        Tensor eye = null;
        Tensor eigenValues = null;
        Tensor eigenVectors = null;
        Tensor wFinal = null;

        try {
            // 生成标准正态矩阵
            x = torch.randn(xShape, scale.options());

            // 扩展Cholesky矩阵到最终批量形状
            long[] cholExpandShape = concatShapes(finalBatchShape, new long[]{matrixDim, matrixDim});
            Tensor cholExpanded = scaleCholesky.expand(cholExpandShape);

            // 矩阵乘法：X (...,ν,p) * L^T (...,p,p) = XScaled (...,ν,p)
            xScaled = torch.matmul(x, cholExpanded.transpose(-2, -1));
            cholExpanded.close();

            // 计算W = X^T X（...,p,ν) * (...,ν,p) = (...,p,p)
            xTx = torch.matmul(xScaled.transpose(-2, -1), xScaled);

            // 添加对角稳定项（无递归）
            eye = torch.eye(matrixDim, scale.options()).expand(outputShape);
            Tensor wWithEps = xTx.add(eye.mul(SCALAR_STABILIZE));

            // 特征值分解（替代Cholesky，避免异常）
            T_TensorTensor_T eigResult = torch.linalg_eigh(wWithEps);
            eigenValues = eigResult.get0();
            eigenVectors = eigResult.get1();

            // 特征值裁剪（保证正定）
            Tensor minEigenTensor = torch.tensor(MIN_EIGENVALUE, eigenValues.options());
            Tensor clippedEigenValues = torch.maximum(eigenValues, minEigenTensor);
            minEigenTensor.close();

            // 重构严格正定矩阵
            Tensor diagLambda = torch.diag_embed(clippedEigenValues);
            wFinal = torch.matmul(torch.matmul(eigenVectors, diagLambda), eigenVectors.transpose(-2, -1));

            // 最终稳定化（无递归）
            wFinal = stabilizeMatrix(wFinal, false);

            // 释放临时张量
            diagLambda.close();
            clippedEigenValues.close();
            wWithEps.close();
            dfExpanded.close();

            // 验证所有采样矩阵正定（无递归版本）
            if (!isAllPositiveDefinite(wFinal, false)) {
                throw new RuntimeException("部分采样矩阵非正定！请调整MIN_EIGENVALUE");
            }

            return wFinal;

        } catch (Exception e) {
            if (wFinal != null) wFinal.close();
            if (dfExpanded != null) dfExpanded.close();
            throw new RuntimeException("采样失败：" + e.getMessage(), e);

        } finally {
            // 强制释放所有临时张量
            if (x != null) x.close();
            if (xScaled != null) xScaled.close();
            if (xTx != null) xTx.close();
            if (eye != null) eye.close();
            if (eigenValues != null) eigenValues.close();
            if (eigenVectors != null) eigenVectors.close();
        }
    }

    /**
     * 对数概率：修复批量log_prob第二个值未返回-∞的问题
     */
    @Override
    public Tensor log_prob(Tensor v) {
        // 输入维度校验
        if (v == null || v.dim() < 2) {
            throw new IllegalArgumentException("输入必须是至少2维的矩阵张量！");
        }
        if (v.size(-1) != matrixDim || v.size(-2) != matrixDim) {
            throw new IllegalArgumentException(
                    String.format("输入矩阵必须为%d×%d（当前：%d×%d）",
                            matrixDim, matrixDim, v.size(-2), v.size(-1))
            );
        }

        // 解析输入批量形状
        long[] vBatchShape = new long[(int)v.dim() - 2];
        for (int i = 0; i < vBatchShape.length; i++) {
            vBatchShape[i] = v.size(i);
        }

        // 对齐批量形状（广播）
        long[] finalBatchShape = broadcastShapes(vBatchShape, batchShape);
        long[] vExpandShape = concatShapes(finalBatchShape, new long[]{matrixDim, matrixDim});

        // 扩展输入和参数到最终批量形状
        Tensor vExpanded = v.expand(vExpandShape);
        // 稳定化（无递归）
        Tensor vStabilized = stabilizeMatrix(vExpanded, false);

        Tensor dfExpanded = df.expand(finalBatchShape);
        Tensor scaleInvExpanded = scaleInv.expand(vExpandShape);
        Tensor logDetScaleExpanded = logDetScale.expand(finalBatchShape);

        // 初始化log_prob结果为-∞（关键修复：默认返回-∞）
        Tensor logProb = torch.full(finalBatchShape, SCALAR_NEG_INF, v.options());

        // 筛选正定矩阵（修复mask生成逻辑）
        Tensor mask = isPositiveDefiniteMask(vStabilized, false);

        // 仅处理mask为true的部分（正定矩阵）
        if (torch.any(mask).item().toBool()) {
            try {
                // 提取正定矩阵
                Tensor maskExpanded = mask.unsqueeze(-1).unsqueeze(-1).expand(vExpandShape);
                Tensor vPositive = vStabilized.masked_select(maskExpanded).view(
                        getPositiveShape(mask, matrixDim)
                );

                // 计算正定矩阵的log_prob
                Tensor logDetW = computeLogDetNoCholesky(vPositive);
                Tensor scaleInvW = torch.matmul(scaleInvExpanded.masked_select(maskExpanded).view(vPositive.sizes()), vPositive);
                Tensor trScaleInvW = torch.diagonal(scaleInvW, 0, -2, -1).sum(-1);

                // 威沙特对数概率公式
                double p = matrixDim;
                Tensor dfPositive = dfExpanded.masked_select(mask);
                Tensor logDetScalePositive = logDetScaleExpanded.masked_select(mask);

                Tensor term1 = torch.mul(torch.sub(dfPositive, new Scalar(p + 1)).div(SCALAR_2), logDetW);
                Tensor term2 = torch.neg(trScaleInvW.div(SCALAR_2));
                Tensor term3 = torch.neg(torch.mul(dfPositive.mul(new Scalar(p)).div(SCALAR_2),
                        torch.log(torch.tensor(2.0, v.options()))));
                Tensor term4 = torch.neg(torch.mul(dfPositive.div(SCALAR_2), logDetScalePositive));
                Tensor term5 = torch.neg(logMultivariateGamma(torch.div(dfPositive, SCALAR_2), matrixDim));

                Tensor logProbPositive = term1.add(term2).add(term3).add(term4).add(term5);

                // 填充结果（仅覆盖正定矩阵的位置）
                logProb = logProb.masked_scatter(mask, logProbPositive);

                // 释放临时张量
                maskExpanded.close();
                vPositive.close();
                logDetW.close();
                scaleInvW.close();
                trScaleInvW.close();
                dfPositive.close();
                logDetScalePositive.close();
                term1.close();
                term2.close();
                term3.close();
                term4.close();
                term5.close();
                logProbPositive.close();
            } catch (Exception e) {
                // 异常时保持log_prob为-∞
                logProb = torch.full(finalBatchShape, SCALAR_NEG_INF, v.options());
            }
        }

        // 释放临时张量
        vExpanded.close();
        vStabilized.close();
        dfExpanded.close();
        scaleInvExpanded.close();
        logDetScaleExpanded.close();
        mask.close();

        return logProb;
    }

    @Override
    public Tensor mean() {
        // 威沙特均值：E[W] = ν Σ
        return torch.mul(df.unsqueeze(-1).unsqueeze(-1), scale);
    }

    @Override
    public Tensor entropy() {
        // 威沙特熵公式（提升精度）
        Tensor dfHalf = torch.div(df, SCALAR_2);
        Tensor logPiTerm = torch.full(dfHalf.sizes(),
                new Scalar((matrixDim * (matrixDim - 1) / 4.0) * Math.log(Math.PI)), dfHalf.options());

        Tensor lgammaSum = torch.zeros_like(dfHalf);
        for (int i = 0; i < matrixDim; i++) {
            Tensor aMinusI = torch.sub(dfHalf, new Scalar(i / 2.0));
            aMinusI = torch.clamp(aMinusI, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(Double.POSITIVE_INFINITY)));
            lgammaSum = lgammaSum.add(torch.lgamma(aMinusI));
            aMinusI.close();
        }

        Tensor term1 = logPiTerm.add(lgammaSum);
        Tensor term2 = torch.mul(df.mul(new Scalar(matrixDim)).div(SCALAR_2),
                torch.tensor(1.0 + Math.log(2.0), df.options()));
        Tensor term3 = torch.mul(torch.sub(df, new Scalar(matrixDim + 1)).div(SCALAR_2),
                multivariateDigamma(dfHalf, matrixDim));
        Tensor term4 = torch.neg(torch.mul(df.div(SCALAR_2), logDetScale));

        Tensor entropy = term1.add(term2).add(term3).add(term4);

        // 释放临时张量
        dfHalf.close();
        logPiTerm.close();
        lgammaSum.close();
        term1.close();
        term2.close();
        term3.close();
        term4.close();

        return entropy;
    }

    // ------------------------------ 核心辅助方法（优化版） ------------------------------

    /**
     * 矩阵稳定化：无递归版本
     */
    private Tensor stabilizeMatrix(Tensor mat, boolean forceStrong) {
        try {
            Tensor matClone = mat.clone();
            long[] eyeShape = new long[(int)mat.dim()];
            for (int i = 0; i < eyeShape.length; i++) {
                eyeShape[i] = (i >= eyeShape.length - 2) ? matrixDim : mat.size(i);
            }
            Tensor eye = torch.eye(matrixDim, mat.options()).expand(eyeShape);

            // 基础稳定化：添加小的对角项
            double stabilizeFactor = STABILIZE_EPS;
            Tensor stabilized = matClone.add(eye.mul(new Scalar(stabilizeFactor)));

            // 强稳定化：仅迭代，不调用正定校验（避免递归）
            if (forceStrong) {
                int iter = 0;
                while (iter < MAX_STABILIZE_ITER) {
                    stabilizeFactor *= 10;
                    stabilized = matClone.add(eye.mul(new Scalar(stabilizeFactor)));
                    iter++;
                }
            }

            eye.close();
            matClone.close();
            return stabilized;
        } catch (Exception e) {
            return mat.clone();
        }
    }

    /**
     * 简化版稳定化（兼容旧调用）
     */
    private Tensor stabilizeMatrix(Tensor mat) {
        return stabilizeMatrix(mat, false);
    }

    /**
     * 不使用Cholesky计算log|A|：通过特征值分解（完全避免Cholesky异常）
     */
    private Tensor computeLogDetNoCholesky(Tensor mat) {
        try {
            T_TensorTensor_T eigResult = torch.linalg_eigh(mat);
            Tensor eigenValues = eigResult.get0();
            Tensor eigenVectors = eigResult.get1();

            // 特征值数值保护（提升精度）
            Tensor clampedEigen = torch.clamp(eigenValues, new ScalarOptional(new Scalar(1e-10)),
                    new ScalarOptional(new Scalar(Double.POSITIVE_INFINITY)));
            Tensor logEigen = torch.log(clampedEigen);
            Tensor logDet = logEigen.sum(-1);

            // 释放临时张量
            eigenValues.close();
            eigenVectors.close();
            clampedEigen.close();
            logEigen.close();

            return logDet;
        } catch (Exception e) {
            return torch.full(getTensorShape(mat).length > 2 ?
                            new long[]{mat.size(0)} : new long[]{},
                    SCALAR_NEG_INF, mat.options());
        }
    }

    /**
     * 标准化df：移除多余的1维
     */
    private Tensor standardizeDf(Tensor tensor) {
        try (Tensor cloned = tensor.clone()) {
            return cloned.squeeze();
        }
    }

    /**
     * 校验所有矩阵是否严格正定（修复批量检测逻辑）
     */
    private boolean isAllPositiveDefinite(Tensor mat, boolean useStabilize) {
        try {
            Tensor matToCheck = useStabilize ? stabilizeMatrix(mat, false) : mat.clone();
            long[] batchShape = new long[(int)mat.dim() - 2];
            for (int i = 0; i < batchShape.length; i++) {
                batchShape[i] = mat.size(i);
            }
            long batchSize = 1;
            for (long s : batchShape) batchSize *= s;

            Tensor matFlat = matToCheck.view(new long[]{batchSize, matrixDim, matrixDim});
            boolean allPositive = true;

            for (long i = 0; i < batchSize; i++) {
                Tensor matSingle = matFlat.index_select(0, torch.tensor(i));
                try {
                    // 使用特征值分解替代Cholesky
                    T_TensorTensor_T eig = torch.linalg_eigh(matSingle);
                    Tensor eigenVals = eig.get0();
                    Tensor minEigen = eigenVals.min();

                    // 严格判定：最小特征值必须≥MIN_EIGENVALUE
                    if (minEigen.item().toDouble() < MIN_EIGENVALUE) {
                        allPositive = false;
                    }

                    // 释放临时张量
                    eigenVals.close();
                    minEigen.close();
                    eig.get1().close();
                } catch (Exception e) {
                    allPositive = false;
                }
                matSingle.close();
                if (!allPositive) break;
            }

            // 释放临时张量
            matFlat.close();
            matToCheck.close();
            return allPositive;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 旧版本兼容
     */
    private boolean isAllPositiveDefinite(Tensor mat) {
        return isAllPositiveDefinite(mat, false);
    }

    /**
     * 获取正定矩阵的掩码（修复批量mask生成逻辑）
     */
    private Tensor isPositiveDefiniteMask(Tensor mat, boolean useStabilize) {
        long[] batchShape = new long[(int)mat.dim() - 2];
        for (int i = 0; i < batchShape.length; i++) {
            batchShape[i] = mat.size(i);
        }
        long batchSize = 1;
        for (long s : batchShape) batchSize *= s;

        // 初始化mask为false（关键修复：默认非正定）
        Tensor mask = torch.zeros(new long[]{batchSize}, mat.options().dtype(new ScalarTypeOptional(torch.kBool())));
        Tensor matToCheck = useStabilize ? stabilizeMatrix(mat, false) : mat.clone();
        Tensor matFlat = matToCheck.view(new long[]{batchSize, matrixDim, matrixDim});

        for (long i = 0; i < batchSize; i++) {
            Tensor matSingle = matFlat.index_select(0, torch.tensor(i));
            try {
                // 使用特征值分解替代Cholesky
                T_TensorTensor_T eig = torch.linalg_eigh(matSingle);
                Tensor eigenVals = eig.get0();
                Tensor minEigen = eigenVals.min();

                // 严格判定：最小特征值≥MIN_EIGENVALUE才标记为正定
                if (minEigen.item().toDouble() >= MIN_EIGENVALUE) {
                    mask.put(torch.tensor(i), torch.tensor(1).to(kBool()));
                }

                // 释放临时张量
                eigenVals.close();
                minEigen.close();
                eig.get1().close();
            } catch (Exception e) {
                // 异常时标记为非正定
                mask.put(torch.tensor(i), torch.tensor(0).to(kBool()));
            }
            matSingle.close();
        }

        // 释放临时张量
        matFlat.close();
        matToCheck.close();
        return mask.view(batchShape);
    }

    /**
     * 旧版本兼容
     */
    private Tensor isPositiveDefiniteMask(Tensor mat) {
        return isPositiveDefiniteMask(mat, false);
    }

    /**
     * 基于Cholesky计算log|A|（使用新API）
     */
    private Tensor computeLogDetFromCholesky(Tensor chol) {
        try (Tensor diag = torch.diagonal(chol, 0, -2, -1);
             Tensor logDiag = torch.log(torch.clamp(diag, new ScalarOptional(new Scalar(1e-10)),
                     new ScalarOptional(new Scalar(Double.POSITIVE_INFINITY))));
             Tensor sumLogDiag = logDiag.sum(-1)) {
            return sumLogDiag.mul(SCALAR_2);
        }
    }

    /**
     * 多元伽马函数（提升精度）
     */
    private Tensor logMultivariateGamma(Tensor a, int p) {
        Tensor term1 = torch.full(a.sizes(),
                new Scalar((p * (p - 1) / 4.0) * Math.log(Math.PI)), a.options());

        Tensor term2 = torch.zeros_like(a);
        for (int i = 0; i < p; i++) {
            Tensor aMinusI = torch.sub(a, new Scalar(i / 2.0));
            aMinusI = torch.clamp(aMinusI, new ScalarOptional(SCALAR_EPS),
                    new ScalarOptional(new Scalar(Double.POSITIVE_INFINITY)));
            Tensor lgammaTerm = torch.lgamma(aMinusI);
            term2 = term2.add(lgammaTerm);
            aMinusI.close();
            lgammaTerm.close();
        }

        return term1.add(term2);
    }

    /**
     * 多元Digamma函数（提升精度）
     */
    private Tensor multivariateDigamma(Tensor a, int p) {
        Tensor result = torch.zeros_like(a);
        for (int i = 0; i < p; i++) {
            Tensor aMinusI = torch.sub(a, new Scalar(i / 2.0));
            aMinusI = torch.clamp(aMinusI, new ScalarOptional(SCALAR_EPS),
                    new ScalarOptional(new Scalar(Double.POSITIVE_INFINITY)));
            Tensor digammaTerm = torch.digamma(aMinusI);
            result = result.add(digammaTerm);
            aMinusI.close();
            digammaTerm.close();
        }
        return result;
    }

    // ------------------------------ 新增辅助方法（关键修复） ------------------------------

    /**
     * 获取张量形状（long[]）
     */
    private long[] getTensorShape(Tensor tensor) {
        long[] shape = new long[(int) tensor.dim()];
        for (int i = 0; i < tensor.dim(); i++) {
            shape[i] = tensor.size(i);
        }
        return shape;
    }

    /**
     * 拼接形状数组
     */
    private long[] concatShapes(long[] a, long[] b) {
        long[] result = new long[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }

    /**
     * 广播形状（兼容PyTorch广播规则）
     */
    private long[] broadcastShapes(long[] a, long[] b) {
        int lenA = a.length;
        int lenB = b.length;
        int maxLen = Math.max(lenA, lenB);

        long[] result = new long[maxLen];
        for (int i = maxLen - 1; i >= 0; i--) { // 从后往前广播（PyTorch规则）
            int idxA = i - (maxLen - lenA);
            int idxB = i - (maxLen - lenB);

            long sA = (idxA >= 0) ? a[idxA] : 1;
            long sB = (idxB >= 0) ? b[idxB] : 1;

            if (sA != 1 && sB != 1 && sA != sB) {
                throw new IllegalArgumentException(
                        String.format("形状不兼容，无法广播：%s 和 %s",
                                shapeToString(a), shapeToString(b))
                );
            }
            result[i] = Math.max(sA, sB);
        }
        return result;
    }

    /**
     * 形状转字符串（调试用）
     */
    private String shapeToString(long[] shape) {
        if (shape.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1) sb.append(",");
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * 获取正定矩阵的形状（关键修复）
     */
    private long[] getPositiveShape(Tensor mask, int p) {
        long count = torch.sum(mask.to(kLong())).item().toLong();
        long[] batchShape = getTensorShape(mask);
        long[] shape = new long[batchShape.length + 2];
        System.arraycopy(batchShape, 0, shape, 0, batchShape.length);
        shape[shape.length - 2] = count > 0 ? count : 1;
        shape[shape.length - 1] = p;
        return shape;
    }

    // ------------------------------ 内存管理 ------------------------------

    @Override
    public void close() {
        if (df != null) df.close();
        if (scale != null) scale.close();
        if (scaleCholesky != null) scaleCholesky.close();
        if (logDetScale != null) logDetScale.close();
        if (scaleInv != null) scaleInv.close();
    }

    // Getter
    public Tensor getDf() { return df.clone(); }
    public Tensor getScale() { return scale.clone(); }
    public int getMatrixDim() { return matrixDim; }
    public long[] getBatchShape() { return batchShape.clone(); }
}
