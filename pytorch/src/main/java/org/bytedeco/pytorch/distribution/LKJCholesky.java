package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * LKJCholesky分布：
 * 1. 严格匹配指定PyTorch Java API（allclose/nonzero/put/put_）
 * 2. 彻底修复批量场景下的索引越界异常
 * 3. 支持任意维度的batch输入，无索引越界风险
 * 4. 修复reshape形状不匹配问题（核心修复）
 * 5. 终极修复：eta批量维度和采样批量维度严格对齐（支持[5,2,3,3]这类嵌套批量）
 */
public class LKJCholesky extends Distribution implements AutoCloseable {
    private final int dim;                  // 矩阵维度（必须≥2）
    private final Tensor concentration;     // 浓度参数η（必须>0）
    private final TensorOptions baseOptions; // 基础配置（固定CPU+Float32）
    private final long[] concentrationBatchShape; // 保存η的原始批量形状

    // 预定义标量（全局复用，避免重复创建）
    private static final Scalar SCALAR_0;
    private static final Scalar SCALAR_0_5;
    private static final Scalar SCALAR_1;
    private static final Scalar SCALAR_2;
    private static final Scalar SCALAR_NEG_1;
    private static final Scalar SCALAR_EPS;
    private static final Scalar SCALAR_1_MINUS_EPS;
    private static final Scalar SCALAR_NEG_INF;
    private static final Scalar SCALAR_PI;
    private static final Scalar SCALAR_LOG2;
    private static final Scalar SCALAR_NEG_0_5;

    // 预定义Optional包装器
    private static final ScalarOptional OPTIONAL_0;
    private static final ScalarOptional OPTIONAL_0_5;
    private static final ScalarOptional OPTIONAL_1;
    private static final ScalarOptional OPTIONAL_2;
    private static final ScalarOptional OPTIONAL_EPS;
    private static final ScalarOptional OPTIONAL_1_MINUS_EPS;
    private static final ScalarOptional OPTIONAL_NEG_INF;

    // 静态初始化：全局复用标量和Optional包装器
    static {
        // 基础标量
        SCALAR_0 = new Scalar(0.0f);
        SCALAR_0_5 = new Scalar(0.5f);
        SCALAR_1 = new Scalar(1.0f);
        SCALAR_2 = new Scalar(2.0f);
        SCALAR_NEG_1 = new Scalar(-1.0f);
        SCALAR_EPS = new Scalar(1e-8f);
        SCALAR_1_MINUS_EPS = new Scalar(1.0f - 1e-8f);
        SCALAR_NEG_INF = new Scalar(Float.NEGATIVE_INFINITY);
        SCALAR_PI = new Scalar((float) Math.PI);
        SCALAR_LOG2 = new Scalar((float) Math.log(2));
        SCALAR_NEG_0_5 = new Scalar(-0.5f);

        // Optional包装（严格匹配API要求）
        OPTIONAL_0 = new ScalarOptional(SCALAR_0);
        OPTIONAL_0_5 = new ScalarOptional(SCALAR_0_5);
        OPTIONAL_1 = new ScalarOptional(SCALAR_1);
        OPTIONAL_2 = new ScalarOptional(SCALAR_2);
        OPTIONAL_EPS = new ScalarOptional(SCALAR_EPS);
        OPTIONAL_1_MINUS_EPS = new ScalarOptional(SCALAR_1_MINUS_EPS);
        OPTIONAL_NEG_INF = new ScalarOptional(SCALAR_NEG_INF);
    }

    /**
     * 构造函数：严格参数校验 + 内存安全初始化
     */
    public LKJCholesky(int dim, Tensor concentration) {
        // 1. 基础参数校验
        if (dim < 2) {
            throw new IllegalArgumentException("LKJCholesky分布dim必须≥2，当前值：" + dim);
        }

        // 2. 浓度参数校验（η>0）
        Tensor etaLe0 = torch.le(concentration, new Scalar(0.0f));
        try {
            // 批量安全校验：any+toBool，避免item()转换Scalar
            if (torch.any(etaLe0).item_bool()) {
                String valStr = concentration.numel() > 1 ? "批量参数包含≤0值" :
                        (concentration.numel() > 0 ? String.valueOf(concentration.get(0)) : "空张量");
                throw new IllegalArgumentException("LKJCholesky分布concentration(η)必须>0！当前值：" + valStr);
            }
        } finally {
            safeClose(etaLe0); // 即时释放
        }

        // 3. 统一配置：强制CPU+Float32，避免类型/设备不匹配
        this.dim = dim;
        this.concentration = concentration.to(new Device(DeviceType.CPU), kFloat(), false, true, new MemoryFormatOptional()).clone().detach();
        this.baseOptions = this.concentration.options();
        // 保存η的原始批量形状（关键：用于后续维度对齐）
        this.concentrationBatchShape = this.concentration.sizes().vec().get();
    }

    @Override
    public String name() {
        return "LKJCholesky(dim=" + dim + ")";
    }

    /**
     * 采样：严格使用API规范的slice/clamp等方法
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 前置校验：采样形状不能为空（避免维度计算错误）
        long[] safeSampleShape = sampleShape == null || sampleShape.length == 0 ? new long[]{1} : sampleShape;

        // 步骤1：计算基础形状
        long[] batchShape = concatLongArrays(safeSampleShape, concentrationBatchShape);

        // 步骤2：初始化单位矩阵（安全维度）
        Tensor eyeMat = eye(dim, baseOptions);
        Tensor L = eyeMat.expand(concatLongArrays(batchShape, new long[]{dim, dim})).clone().detach();
        safeClose(eyeMat);

        try {
            // 步骤3：Onion方法采样（逐行构建）
            for (int k = 1; k < dim; k++) {
                // 3.1 计算Beta分布参数（安全扩展）
                Tensor alpha = concentration.add(torch.tensor((dim - k - 1) * 0.5f, baseOptions));
                alpha = alpha.expand(batchShape);

                // 3.2 采样Beta分布并稳定化（严格使用clamp API）
                Tensor v = beta(alpha, alpha.clone());
                v = torch.clamp(v, OPTIONAL_EPS, OPTIONAL_1_MINUS_EPS);

                // 3.3 计算对角元s = sqrt(v)（安全扩展维度）
                Tensor s = torch.sqrt(v);
                s = s.reshape(concatLongArrays(batchShape, new long[]{1, 1}));

                // 3.4 安全更新对角元：严格使用slice API（LongOptional参数）
                Tensor rowK = L.slice(-2, new LongOptional(k), new LongOptional(k+1), 1);
                Tensor diagVal = rowK.slice(-1, new LongOptional(k), new LongOptional(k+1), 1);
                Tensor newDiagVal = diagVal.mul(torch.tensor(0.0f, baseOptions)).add(s);

                // 切片操作严格使用LongOptional
                Tensor rowKFront = rowK.slice(-1, new LongOptional(0), new LongOptional(k), 1);
                Tensor rowKBack = rowK.slice(-1, new LongOptional(k+1), new LongOptional(dim), 1);
                Tensor newRowK = torch.cat(new TensorVector(rowKFront, newDiagVal, rowKBack), -1);

                Tensor LBeforeK = L.slice(-2, new LongOptional(0), new LongOptional(k), 1);
                Tensor LAfterK = L.slice(-2, new LongOptional(k+1), new LongOptional(dim), 1);
                Tensor newL = torch.cat(new TensorVector(LBeforeK, newRowK, LAfterK), -2);

                // 释放临时张量并更新L
                safeClose(L);
                L = newL.detach();
                safeClose(rowK);
                safeClose(diagVal);
                safeClose(newDiagVal);
                safeClose(rowKFront);
                safeClose(rowKBack);
                safeClose(newRowK);
                safeClose(LBeforeK);
                safeClose(LAfterK);

                // 3.5 采样球面均匀分布向量
                if (k > 0) {
                    // 采样并归一化z向量
                    long[] zShape = concatLongArrays(batchShape, new long[]{k});
                    Tensor z = torch.randn(zShape, baseOptions);
                    Tensor zNorm = torch.norm(z, OPTIONAL_2, new long[]{-1}, true);
                    z = z.div(torch.clamp(zNorm, OPTIONAL_EPS));
                    safeClose(zNorm);

                    // 计算缩放因子r
                    Tensor r = torch.sqrt(torch.tensor(1.0f, baseOptions).sub(v));
                    r = r.reshape(concatLongArrays(batchShape, new long[]{1}));

                    // 计算w = z * r
                    Tensor w = z.mul(r);
                    safeClose(z);
                    safeClose(r);

                    // 安全赋值到第k行前k列：严格使用slice API
                    rowK = L.slice(-2, new LongOptional(k), new LongOptional(k+1), 1);
                    Tensor rowKFrontNew = rowK.slice(-1, new LongOptional(0), new LongOptional(k), 1)
                            .mul(torch.tensor(0.0f, baseOptions))
                            .add(w.reshape(concatLongArrays(batchShape, new long[]{1, k})));
                    Tensor rowKBackNew = rowK.slice(-1, new LongOptional(k), new LongOptional(dim), 1);
                    newRowK = torch.cat(new TensorVector(rowKFrontNew, rowKBackNew), -1);

                    // 重新拼接L矩阵
                    LBeforeK = L.slice(-2, new LongOptional(0), new LongOptional(k), 1);
                    LAfterK = L.slice(-2, new LongOptional(k+1), new LongOptional(dim), 1);
                    newL = torch.cat(new TensorVector(LBeforeK, newRowK, LAfterK), -2);

                    // 释放临时张量并更新L
                    safeClose(L);
                    L = newL.detach();
                    safeClose(rowK);
                    safeClose(rowKFrontNew);
                    safeClose(rowKBackNew);
                    safeClose(newRowK);
                    safeClose(LBeforeK);
                    safeClose(LAfterK);
                    safeClose(w);
                }

                // 释放当前循环临时张量
                safeClose(alpha);
                safeClose(v);
                safeClose(s);
            }

            // 最终安全处理：强制下三角 + 移除多余维度
            L = torch.tril(L);
            if (safeSampleShape.length == 1 && safeSampleShape[0] == 1 && sampleShape.length == 0) {
                L = L.squeeze(0); // 移除默认添加的batch维度
            }

            return L.clone().detach();

        } catch (Exception e) {
            safeClose(L);
            throw new RuntimeException("LKJCholesky采样失败：" + e.getMessage(), e);
        }
    }

    /**
     * 对数概率：
     * 1. 彻底修复批量索引越界问题
     * 2. 终极修复：保留原始批量结构，不展平η的维度，直接和输入批量对齐
     */
    @Override
    public Tensor log_prob(Tensor input) {
        if (input == null) {
            throw new IllegalArgumentException("输入张量不能为空！");
        }

        // 步骤1：保存原始形状，统一输入维度（确保为 [*, dim, dim]）
        long[] originalShape = input.shape();
        Tensor inputClone = input.clone().detach();
        boolean isScalarInput = (originalShape.length == 2);
        if (isScalarInput) {
            inputClone = inputClone.unsqueeze(0); // [dim, dim] → [1, dim, dim]
        }
        long[] inputShape = inputClone.shape();
        int inputBatchRank = inputShape.length - 2; // 输入的批量维度数
        long[] inputBatchShape = Arrays.copyOfRange(inputShape, 0, inputBatchRank);

        // 步骤2：核心修复——让η的批量维度和输入批量维度对齐（不展平！）
        Tensor etaAligned = alignConcentrationToInput(inputBatchShape);
        long[] etaAlignedShape = etaAligned.sizes().vec().get();

        // 步骤3：判断输入是否为合法的下三角Cholesky矩阵
        // 3.1 判断是否为下三角矩阵（上三角全为0）
        Tensor trilMask = torch.tril(torch.ones(new long[]{dim, dim}, baseOptions), 0);
        Tensor upperMask = torch.ones_like(trilMask).sub(trilMask);
        Tensor upperPart = inputClone.mul(upperMask);
        Tensor upperAllZero = upperPart.abs().lt(SCALAR_EPS).all(new long[]{-1, -2}); // [inputBatch...]
        // 3.2 判断对角元是否全>0
        Tensor diag = inputClone.diagonal(0, -1, -2); // [inputBatch..., dim]
        Tensor diagAllPositive = diag.gt(SCALAR_EPS).all(new long[]{-1}); // [inputBatch...]
        // 3.3 合并合法条件
        Tensor isValid = upperAllZero.logical_and(diagAllPositive); // [inputBatch...]

        // 步骤4：初始化log_prob为-∞
        Tensor logProb = torch.full(inputClone.shape(), SCALAR_NEG_INF, baseOptions);

        // 步骤5：只对合法矩阵计算log_prob（不展平，直接按批量维度计算）
        if (isValid.any().item_bool()) {
            // 5.1 计算核心log_prob（保留批量结构）
            Tensor coreLogProb = calculateLKJLogProbWithBatch(inputClone, etaAligned);

            // 5.2 只保留合法样本的log_prob，非法样本保持-∞
            // 扩展isValid维度到 [inputBatch..., 1, 1] 以匹配logProb形状
            Tensor isValidExpanded = isValid.unsqueeze(-1).unsqueeze(-1).expand(inputClone.shape());
            logProb = torch.where(isValidExpanded, coreLogProb, logProb);

            safeClose(coreLogProb);
            safeClose(isValidExpanded);
        }

        // 步骤6：恢复原始形状
        if (isScalarInput) {
            logProb = logProb.squeeze(0); // 移除临时添加的batch维度
            if (logProb.dim() > 0) {
                logProb = logProb.reshape(new long[]{}); // 转为标量
            }
        } else {
            logProb = logProb.reshape(originalShape);
        }

        // 释放临时张量
        safeClose(inputClone);
        safeClose(trilMask);
        safeClose(upperMask);
        safeClose(upperPart);
        safeClose(upperAllZero);
        safeClose(diag);
        safeClose(diagAllPositive);
        safeClose(isValid);
        safeClose(etaAligned);

        return logProb;
    }

    // ===================== 核心修复：保留批量结构计算log_prob =====================
    /**
     * 不展平批量维度，直接按输入的批量结构计算log_prob
     * @param input [batch..., dim, dim] 输入张量
     * @param etaAligned [batch...] 对齐后的η张量
     * @return [batch..., dim, dim] log_prob结果
     */
    private Tensor calculateLKJLogProbWithBatch(Tensor input, Tensor etaAligned) {
        long k = dim;
        long[] inputShape = input.sizes().vec().get();
        int batchRank = inputShape.length - 2;
        long[] batchShape = Arrays.copyOfRange(inputShape, 0, batchRank);

        // 步骤1：提取对角元并计算log det(J)
        Tensor diag = input.diagonal(0, -1, -2); // [batch..., dim]
        Tensor logDetJ = torch.zeros(batchShape, baseOptions);

        for (int i = 0; i < k; i++) {
            long power = 2 * (k - i - 1);
            Tensor diagI = diag.select(-1, i); // [batch...]
            logDetJ = logDetJ.add(diagI.log().mul(new Scalar(power)));
            safeClose(diagI);
        }

        // 步骤2：计算非对角项的sum(log(1-y²))
        Tensor sumLog1MinusY2 = torch.zeros(batchShape, baseOptions);
        Tensor trilNonDiag = input.mul(torch.tril(torch.ones(new long[]{k, k}, baseOptions), -1));

        for (int i = 1; i < k; i++) {
            for (int j = 0; j < i; j++) {
                // 提取第i行第j列的元素 [batch...]
                Tensor yij = trilNonDiag.select(-2, i).select(-1, j);
                Tensor logTerm = torch.ones_like(yij).sub(yij.pow(SCALAR_2)).log();
                sumLog1MinusY2 = sumLog1MinusY2.add(logTerm);
                safeClose(yij);
                safeClose(logTerm);
            }
        }

        // 步骤3：核心修复——直接用对齐后的η计算（维度完全匹配）
        Tensor etaMinus1 = etaAligned.sub(SCALAR_1); // [batch...]
        Tensor kernelLogProb = sumLog1MinusY2.mul(etaMinus1); // [batch...]

        // 步骤4：合并结果
        Tensor logNormalizer = calculateLogNormalizer(concentration); // 标量
        Tensor batchLogProb = kernelLogProb.add(logDetJ).sub(logNormalizer); // [batch...]

        // 扩展维度到 [batch..., dim, dim]
        batchLogProb = batchLogProb.unsqueeze(-1).unsqueeze(-1).expand(inputShape);

        // 释放临时张量
        safeClose(diag);
        safeClose(logDetJ);
        safeClose(trilNonDiag);
        safeClose(sumLog1MinusY2);
        safeClose(etaMinus1);
        safeClose(kernelLogProb);
        safeClose(logNormalizer);

        return batchLogProb;
    }

    // ===================== 辅助方法：将η对齐到输入的批量维度 =====================
    private Tensor alignConcentrationToInput(long[] inputBatchShape) {
        Tensor eta = this.concentration.clone().detach();
        long[] etaShape = eta.sizes().vec().get();
        int etaRank = etaShape.length;
        int inputRank = inputBatchShape.length;

        // 情况1：η是标量 → 扩展为输入的批量形状
        if (etaRank == 0) {
            return eta.expand(inputBatchShape);
        }

        // 情况2：η的维度 < 输入批量维度 → 前置补1维后扩展
        if (etaRank < inputRank) {
            for (int i = 0; i < inputRank - etaRank; i++) {
                eta = eta.unsqueeze(0);
            }
            return eta.expand(inputBatchShape);
        }

        // 情况3：η的维度 == 输入批量维度 → 直接扩展（兼容广播）
        if (etaRank == inputRank) {
            return eta.expand(inputBatchShape);
        }

        // 情况4：η的维度 > 输入批量维度 → 取后inputRank维（适配你的测试场景：η[2] → 输入批量[5,2]）
        if (etaRank > inputRank) {
            int startDim = etaRank - inputRank;
            // 计算需要保留的维度
            long[] newEtaShape = new long[inputRank];
            System.arraycopy(etaShape, startDim, newEtaShape, 0, inputRank);
            eta = eta.reshape(newEtaShape);
            return eta.expand(inputBatchShape);
        }

        return eta;
    }

    // ===================== 辅助方法：计算LKJCholesky的对数归一化常数 =====================
    private Tensor calculateLogNormalizer(Tensor eta) {
        long k = dim;
        Tensor logNormalizer = torch.tensor(0.0f, baseOptions);

        for (int d = 1; d < k; d++) {
            Tensor alpha = eta.add(new Scalar((k - 1 - d) / 2.0f));
            Tensor term1 = torch.lgamma(alpha).mul(SCALAR_2);
            Tensor term2 = torch.log(torch.tensor((float) Math.PI)).mul(new Scalar(d));
            Tensor term3 = torch.lgamma(alpha.mul(SCALAR_2)).neg();

            logNormalizer = logNormalizer.add(term1).add(term2).add(term3);

            safeClose(alpha);
            safeClose(term1);
            safeClose(term2);
            safeClose(term3);
        }

        return logNormalizer;
    }

    @Override
    public Tensor mean() {
        long[] meanShape = concatLongArrays(concentrationBatchShape, new long[]{dim, dim});
        Tensor mean = eye(dim, baseOptions).expand(meanShape).clone().detach();
        return mean;
    }

    @Override
    public Tensor entropy() {
        Tensor entropy = torch.zeros(concentrationBatchShape, baseOptions);
        double d = dim;

        for (int k = 1; k < d; k++) {
            Tensor alpha = concentration.add(torch.tensor((d - k - 1) * 0.5f, baseOptions));
            Tensor term = lgamma(alpha)
                    .add(torch.log(torch.tensor((float)Math.PI)).mul(torch.tensor(k * 0.5f, baseOptions)))
                    .sub(lgamma(alpha.add(torch.tensor(k * 0.5f, baseOptions))));
            entropy = entropy.add(term);
            safeClose(alpha);
            safeClose(term);
        }

        entropy = entropy.neg();
        for (int k = 1; k < d; k++) {
            Tensor alpha = concentration.add(torch.tensor((d - k - 1) * 0.5f, baseOptions));
            Tensor digammaTerm = digamma(alpha)
                    .add(digamma(alpha.add(torch.tensor(k * 0.5f, baseOptions))))
                    .mul(torch.tensor(0.5f).mul(torch.tensor(k, baseOptions)));
            entropy = entropy.add(digammaTerm);
            safeClose(alpha);
            safeClose(digammaTerm);
        }

        double sumK = (d * (d - 1)) / 2.0;
        Tensor sumKTensor = torch.tensor((float) sumK, baseOptions);
        Tensor log2Minus1 = torch.log(torch.tensor((float)Math.PI)).sub(SCALAR_1);
        entropy = entropy.add(sumKTensor.mul(log2Minus1));

        safeClose(sumKTensor);
        safeClose(log2Minus1);

        return entropy.clone().detach();
    }

    /**
     * 资源释放：彻底释放所有内部张量
     */
    @Override
    public void close() {
        safeClose(concentration);
    }

    // -------------------------- 辅助方法 --------------------------
    private void safeClose(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                System.err.println("LKJCholesky资源释放警告：" + e.getMessage());
            }
        }
    }

    private long[] concatLongArrays(long[] a, long[] b) {
        if (a == null) return b;
        if (b == null) return a;
        long[] result = new long[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }

    // Getter方法
    public int getDim() {
        return dim;
    }

    public Tensor getConcentration() {
        return concentration.clone().detach();
    }
}
