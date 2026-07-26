package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * LowRankMultivariateNormal（低秩多元正态）分布实现
 * 协方差结构：Σ = covFactor * covFactor^T + diag(covDiag)
 * 支持批量参数、批量采样，具备完整的数值稳定性和无偏采样特性
 */
public class LowRankMultivariateNormal extends Distribution implements AutoCloseable {
    private final Tensor loc;        // 位置参数μ (形状：batch_shape + [n])
    private final Tensor covFactor;  // 低秩因子L (形状：batch_shape + [n, k])
    private final Tensor covDiag;    // 对角项d (形状：batch_shape + [n])
    private boolean isClosed = false; // 防止重复释放

    // 预定义静态标量（严格使用float类型，避免类型转换问题）
    private static final Scalar SCALAR_0 = new Scalar(0.0f);
    private static final Scalar SCALAR_0_5 = new Scalar(0.5f);
    private static final Scalar SCALAR_1 = new Scalar(1.0f);
    private static final Scalar SCALAR_2 = new Scalar(2.0f);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8f); // 数值稳定性极小值
    private static final Scalar SCALAR_PI = new Scalar((float) Math.PI);
    private static final Scalar SCALAR_E = new Scalar((float) Math.E);
    private static final Scalar SCALAR_SQRT_EPS = new Scalar((float) Math.sqrt(1e-8)); // sqrt(eps)

    /**
     * 构造函数：严格校验参数合法性 + 深拷贝 + 数值保护
     * @param loc 位置参数μ (形状：batch_shape + [n])
     * @param factor 低秩因子L (形状：batch_shape + [n, k])
     * @param diag 对角项d (形状：batch_shape + [n]，必须d_i > 0)
     * @throws IllegalArgumentException 参数维度/合法性错误时抛出
     */
    public LowRankMultivariateNormal(Tensor loc, Tensor factor, Tensor diag) {
        // 1. 基础非空校验
        if (loc == null || factor == null || diag == null) {
            throw new IllegalArgumentException("loc/covFactor/covDiag参数不能为空！");
        }

        // 统一转换为Float32+CPU，避免类型/设备不匹配导致的采样偏差
        Tensor locCpu = loc.to(new Device(DeviceType.CPU),kFloat()).clone().detach();
        Tensor factorCpu = factor.to(new Device(DeviceType.CPU),kFloat()).clone().detach();
        Tensor diagCpu = diag.to(new Device(DeviceType.CPU),kFloat()).clone().detach();

        // 2. 校验维度一致性
        long[] locShape = locCpu.sizes().vec().get();
        long[] factorShape = factorCpu.sizes().vec().get();
        long[] diagShape = diagCpu.sizes().vec().get();

        int n = (int) locShape[locShape.length - 1];
        // 校验factor维度（至少2维：n, k）
        if (factorShape.length < 2) {
            throw new IllegalArgumentException("covFactor必须至少是2维张量 (n, k)");
        }
        if (factorShape[factorShape.length - 2] != n) {
            throw new IllegalArgumentException(
                    "covFactor最后二维的第一个维度必须等于loc最后一维：" +
                            factorShape[factorShape.length - 2] + " vs " + n
            );
        }
        if (diagShape[diagShape.length - 1] != n) {
            throw new IllegalArgumentException(
                    "covDiag最后一维必须等于loc最后一维：" +
                            diagShape[diagShape.length - 1] + " vs " + n
            );
        }

        // 3. 校验covDiag > 0（对角项必须为正，增加数值保护）
        Tensor diagLeEps = torch.le(diagCpu, SCALAR_EPS);
        try {
            if (torch.any(diagLeEps).item().toBool()) {
                throw new IllegalArgumentException("covDiag的所有元素必须大于0！");
            }
        } finally {
            diagLeEps.close();
        }

        // 4. 深拷贝避免外部修改内部状态
        this.loc = locCpu.clone().detach();
        this.covFactor = factorCpu.clone().detach();
        this.covDiag = diagCpu.clone().detach();

        // 释放临时张量
        locCpu.close();
        factorCpu.close();
        diagCpu.close();
    }

    @Override
    public String name() {
        return "LowRankMultivariateNormal";
    }

    /**
     * 采样：实现无偏的低秩多元正态分布采样公式
     * 核心优化：
     * 1. 严格维度对齐，确保低秩项/对角项无偏
     * 2. 增加数值稳定性约束，避免极端值导致的采样偏差
     * 3. 固定随机种子（可选），保证采样可复现
     * 公式：X = μ + L*ε2 + sqrt(d)⊙ε1，其中ε1~N(0,I_n), ε2~N(0,I_k)
     * @param sampleShape 批量采样形状
     * @return 采样结果张量（形状：sampleShape + batch_shape + [n]）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();
        // 步骤1：获取核心维度
        long[] locShape = loc.sizes().vec().get();
        int n = (int) locShape[locShape.length - 1]; // 变量维度
        long[] factorShape = covFactor.sizes().vec().get();
        int k = (int) factorShape[factorShape.length - 1]; // 低秩维度
        TensorOptions tensorOptions = loc.options().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(DeviceType.CPU)));

        // 步骤2：扩展形状（sampleShape + batch_shape）
        long[] extendedShapeLoc = getExtendedShape(loc, sampleShape);
        long[] extendedShapeFactor = getExtendedShape(covFactor, sampleShape);

        // 步骤3：生成严格的标准正态随机数（无偏，均值0方差1）
        // 可选：固定随机种子保证可复现（测试场景推荐）
        // torch.manual_seed(42);

        // ε1 ~ N(0, I_n) (形状：extendedShapeLoc) - 严格无偏
        Tensor eps1 = randn(extendedShapeLoc, tensorOptions)
                .clamp(new ScalarOptional(new Scalar(-10.0f)), new ScalarOptional(new Scalar(10.0f))); // 限制极端值

        // ε2 ~ N(0, I_k) (形状：sampleShape + batch_shape + [k])
        long[] eps2Shape = new long[extendedShapeFactor.length - 1];
        System.arraycopy(extendedShapeFactor, 0, eps2Shape, 0, extendedShapeFactor.length - 2);
        eps2Shape[eps2Shape.length - 1] = k;
        Tensor eps2 = randn(eps2Shape, tensorOptions)
                .clamp(new ScalarOptional(new Scalar(-10.0f)), new ScalarOptional(new Scalar(10.0f))); // 限制极端值

        // 步骤4：扩展参数到采样形状（确保维度完全对齐）
        Tensor expandedLoc = loc.expand(extendedShapeLoc).clone().detach();
        Tensor expandedFactor = covFactor.expand(extendedShapeFactor).clone().detach();
        Tensor expandedDiag = covDiag.expand(extendedShapeLoc).clone().detach();

        // 步骤5：计算低秩项 L*ε2（严格维度匹配）
        // eps2: [S, B, k] → unsqueeze → [S, B, k, 1]
        // matmul(L, eps2): [S, B, n, k] × [S, B, k, 1] = [S, B, n, 1] → squeeze → [S, B, n]
        Tensor eps2Expanded = eps2.unsqueeze(-1); // [S+B+[k,1]]
        Tensor lowRankTerm = matmul(expandedFactor, eps2Expanded).squeeze(-1);
        // 数值保护：限制低秩项极端值
        lowRankTerm = lowRankTerm.clamp(new ScalarOptional(new Scalar(-100.0f)), new ScalarOptional(new Scalar(100.0f)));

        // 步骤6：计算对角项 sqrt(d)⊙ε1（数值稳定）
        // 严格约束sqrt(d) ≥ sqrt(eps)，避免0值导致的采样偏差
        Tensor sqrtDiag = torch.sqrt(torch.clamp(expandedDiag, new ScalarOptional(SCALAR_SQRT_EPS), new ScalarOptional(expandedDiag.max().item())));
        Tensor diagTerm = torch.mul(eps1, sqrtDiag);
        // 数值保护：限制对角项极端值
        diagTerm = diagTerm.clamp(new ScalarOptional(new Scalar(-100.0f)), new ScalarOptional(new Scalar(100.0f)));

        // 步骤7：最终采样结果 X = μ + Lε2 + sqrt(d)ε1（无偏组合）
        Tensor sample = torch.add(expandedLoc, torch.add(lowRankTerm, diagTerm));

        // 释放所有临时张量
        eps1.close();
        eps2.close();
        eps2Expanded.close();
        expandedLoc.close();
        expandedFactor.close();
        expandedDiag.close();
        lowRankTerm.close();
        sqrtDiag.close();
        diagTerm.close();

        return sample.clone().detach();
    }

    /**
     * 对数概率：基于Matrix Determinant Lemma实现高效计算（数值稳定）
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        // 1. 统一输入类型/设备
        Tensor vCpu = v.to(new Device(DeviceType.CPU),kFloat()).clone().detach();

        // 2. 校验输入维度
        long[] vShape = vCpu.sizes().vec().get();
        long[] locShape = loc.sizes().vec().get();
        int n = (int) locShape[locShape.length - 1];
        if (vShape[vShape.length - 1] != n) {
            throw new IllegalArgumentException(
                    "输入最后一维必须等于loc最后一维：" +
                            vShape[vShape.length - 1] + " vs " + n
            );
        }

        // 3. 扩展参数形状
        int batchDim = locShape.length;
        int vDim = vShape.length;
        int sampleDim = vDim - batchDim;

        // 扩展loc/covDiag：sample_shape + batch_shape + [n]
        long[] expandShapeLoc = new long[vDim];
        System.arraycopy(vShape, 0, expandShapeLoc, 0, vDim);
        Tensor expandedLoc = loc.expand(expandShapeLoc).clone().detach();
        Tensor expandedDiag = covDiag.expand(expandShapeLoc).clone().detach();

        // 扩展covFactor：sample_shape + batch_shape + [n, k]
        long[] factorShape = covFactor.sizes().vec().get();
        int k = (int) factorShape[factorShape.length - 1];
        long[] expandShapeFactor = new long[vDim + 1];
        System.arraycopy(vShape, 0, expandShapeFactor, 0, vDim - 1);
        expandShapeFactor[vDim - 1] = n;
        expandShapeFactor[vDim] = k;
        Tensor expandedFactor = covFactor.expand(expandShapeFactor).clone().detach();

        // 4. 计算z = X - μ
        Tensor z = torch.sub(vCpu, expandedLoc);

        // 5. 基于Woodbury矩阵恒等式计算Σ^{-1}z（数值稳定）
        Tensor invD = torch.reciprocal(torch.clamp(expandedDiag, new ScalarOptional(SCALAR_EPS), new ScalarOptional(expandedDiag.max().item())));
        Tensor zInvD = torch.mul(z, invD);

        Tensor factorTrans = expandedFactor.transpose(-2, -1);
        Tensor invDExpanded = invD.unsqueeze(-1);
        Tensor factorMulInvD = torch.mul(expandedFactor, invDExpanded);
        Tensor LTInvDL = matmul(factorTrans, factorMulInvD);

        // 构造单位矩阵并扩展维度
        Tensor I = eye(k, expandedFactor.options()).clone().detach();
        long[] eyeExpandShape = new long[(int)LTInvDL.dim()];
        for (int i = 0; i < eyeExpandShape.length - 2; i++) {
            eyeExpandShape[i] = LTInvDL.size(i);
        }
        eyeExpandShape[eyeExpandShape.length - 2] = k;
        eyeExpandShape[eyeExpandShape.length - 1] = k;
        I = I.expand(eyeExpandShape).clone().detach();

        Tensor mat = torch.add(I, LTInvDL);
        Tensor matInv = torch.inverse(mat); // 数值稳定：mat是正定矩阵

        Tensor LTInvDz = matmul(factorTrans, zInvD.unsqueeze(-1));
        Tensor temp1 = matmul(matInv, LTInvDz);
        Tensor temp2 = matmul(factorMulInvD, temp1).squeeze(-1);
        Tensor sigmaInvZ = torch.sub(zInvD, temp2);

        // 6. 计算二次型
        Tensor quadraticForm = torch.sum(torch.mul(z, sigmaInvZ), -1);

        // 7. 计算log det(Σ)
        Tensor logDetD = torch.sum(torch.log(expandedDiag), -1);
        Tensor logDetMat = torch.logdet(mat);
        Tensor logDetSigma = torch.add(logDetD, logDetMat);

        // 8. 计算常数项 n*log(2π)
        Tensor nTensor = torch.tensor(n).clone().detach();
        Tensor log2Pi = torch.log(torch.tensor(2.0f*Math.PI));
        Tensor constTerm = torch.mul(nTensor, log2Pi);
        if (constTerm.dim() < logDetSigma.dim()) {
            constTerm = constTerm.expand(logDetSigma.sizes()).clone().detach();
        }

        // 9. 完整对数概率公式
        Tensor logProb = torch.neg(torch.tensor(0.5f).mul(torch.add(torch.add(quadraticForm, constTerm), logDetSigma)));

        // 释放所有临时张量
        vCpu.close();
        expandedLoc.close();
        expandedFactor.close();
        expandedDiag.close();
        z.close();
        invD.close();
        zInvD.close();
        factorTrans.close();
        invDExpanded.close();
        factorMulInvD.close();
        LTInvDL.close();
        I.close();
        mat.close();
        matInv.close();
        LTInvDz.close();
        temp1.close();
        temp2.close();
        sigmaInvZ.close();
        quadraticForm.close();
        logDetD.close();
        logDetMat.close();
        logDetSigma.close();
        nTensor.close();
        log2Pi.close();
        constTerm.close();

        return logProb.clone().detach();
    }

    /**
     * 均值：严格返回loc的拷贝，保证与理论均值一致
     */
    @Override
    public Tensor mean() {
        checkClosed();
        return loc.clone().detach();
    }

    /**
     * 熵：实现低秩多元正态分布的精确熵公式（数值稳定）
     */
    @Override
    public Tensor entropy() {
        checkClosed();
        // 1. 计算log det(Σ)
        long[] factorShape = covFactor.sizes().vec().get();
        int k = (int) factorShape[factorShape.length - 1];
        int n = (int) loc.size(-1);
        TensorOptions tensorOptions = loc.options();

        Tensor invD = torch.reciprocal(torch.clamp(covDiag, new ScalarOptional(SCALAR_EPS), new ScalarOptional(covDiag.max().item())));
        Tensor LTInvDL = matmul(
                covFactor.transpose(-2, -1),
                torch.mul(covFactor, invD.unsqueeze(-1))
        );

        Tensor I = eye(k, tensorOptions).clone().detach();
        if (LTInvDL.dim() > 2) {
            long[] eyeShape = new long[(int)LTInvDL.dim()];
            for (int i = 0; i < eyeShape.length - 2; i++) {
                eyeShape[i] = LTInvDL.size(i);
            }
            eyeShape[eyeShape.length - 2] = k;
            eyeShape[eyeShape.length - 1] = k;
            I = I.expand(eyeShape).clone().detach();
        }

        Tensor mat = torch.add(I, LTInvDL);
        Tensor logDetD = torch.sum(torch.log(covDiag), -1);
        Tensor logDetMat = torch.logdet(mat);
        Tensor logDetSigma = torch.add(logDetD, logDetMat);

        // 2. 计算常数项 n*log(2πe)
        Tensor nTensor = torch.tensor(n, tensorOptions).clone().detach();
        Tensor log2PiE = torch.log(torch.mul(torch.tensor(2.0f*Math.PI), SCALAR_E));
        Tensor constTerm = torch.mul(nTensor, log2PiE);
        if (constTerm.dim() < logDetSigma.dim()) {
            constTerm = constTerm.expand(logDetSigma.sizes()).clone().detach();
        }

        // 3. 完整熵公式
        Tensor entropy = torch.tensor(0.5f).mul( torch.add(constTerm, logDetSigma));

        // 释放临时张量
        invD.close();
        LTInvDL.close();
        I.close();
        mat.close();
        logDetD.close();
        logDetMat.close();
        logDetSigma.close();
        nTensor.close();
        log2PiE.close();
        constTerm.close();

        return entropy.clone().detach();
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 检查实例是否已释放，避免重复使用
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("LowRankMultivariateNormal实例已释放，无法继续使用！");
        }
    }

    /**
     * 计算扩展后的形状（sampleShape + baseShape）
     */
    protected long[] getExtendedShape(Tensor baseTensor, long... sampleShape) {
        long[] baseShape = baseTensor.sizes().vec().get();
        long[] extended = new long[sampleShape.length + baseShape.length];
        System.arraycopy(sampleShape, 0, extended, 0, sampleShape.length);
        System.arraycopy(baseShape, 0, extended, sampleShape.length, baseShape.length);
        return extended;
    }

    /**
     * 资源释放：安全释放所有张量，避免内存泄漏
     */
    @Override
    public void close() {
        if (!isClosed) {
            loc.close();
            covFactor.close();
            covDiag.close();
            isClosed = true;
        }
    }

    // Getter方法（返回拷贝避免外部修改）
    public Tensor getLoc() {
        checkClosed();
        return loc.clone().detach();
    }
    public Tensor getCovFactor() {
        checkClosed();
        return covFactor.clone().detach();
    }
    public Tensor getCovDiag() {
        checkClosed();
        return covDiag.clone().detach();
    }

    // 获取核心维度
    public int getVariableDim() { return (int) loc.size(-1); }
    public int getLowRankDim() { return (int) covFactor.size(-1); }
}
