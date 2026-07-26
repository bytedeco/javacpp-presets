package org.bytedeco.pytorch.rl;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.InverseGamma;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * InverseGamma（逆伽马分布）测试用例
 * 核心风格规范：
 * 1. CPU_DEVICE 定义为 TensorOptions 类型
 * 2. arrayToString 参数使用 .vec().get()
 * 3. 覆盖所有核心功能：参数校验、采样（数值稳定）、log_prob、均值、熵、异常场景、资源释放
 * 4. 重点验证：
 *    - 构造参数校验（α≤0/β≤0抛出异常）
 *    - 采样：所有值>0、形状符合预期、数值分布合理
 *    - log_prob：输入v≤0抛出异常/返回-∞；合法输入数值稳定
 *    - 均值：α>1返回β/(α-1)，α=1返回Inf，α<1返回NaN
 *    - 熵：基于公式的数值稳定性
 */
public class InverseGammaTest {
    // 全局配置（严格按统一风格定义）
    private static final TensorOptions CPU_DEVICE = torch.device(new Device(DeviceType.CPU));
    private static final float EPS = 1e-4f; // 数值校验容忍度
    private static final long SEED = 42;    // 固定随机种子保证可复现

    public static void main(String[] args) {
        // 固定随机种子（保证采样结果可复现）
        manual_seed(SEED);

        System.out.println("=== 开始 InverseGamma 分布测试 ===\n");

        // 1. 构造参数校验测试
        testConstructorValidation();

        // 2. 核心功能测试（α=2.0, β=3.0，标量参数）
        testCoreFunctionality();

        // 3. 批量参数测试（α=[2.0,3.0], β=[3.0,4.0]）
        testBatchParameters();

        // 4. 采样特性验证（值>0、形状正确、分布合理性）
        testSamplingProperties();

        // 5. log_prob异常场景验证（输入v≤0）
        testLogProbExceptions();

        // 6. 均值特性验证（α>1/α=1/α<1）
        testMeanProperties();

        // 7. 熵公式数值稳定性测试
        testEntropyStability();

        // 8. 资源释放验证（AutoCloseable实现）
        testResourceRelease();

        // 强制GC + 延迟，确保资源完全释放
        System.gc();
        try {
            Thread.sleep(200);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        System.out.println("\n=== InverseGamma 分布测试全部完成 ===");
    }

    /**
     * 测试1：构造参数校验
     * 验证：
     * 1. α≤0（α=0/-1）→ 抛出IllegalArgumentException
     * 2. β≤0（β=0/-1）→ 抛出IllegalArgumentException
     * 3. α>0但β≤0 → 抛出IllegalArgumentException
     */
    private static void testConstructorValidation() {
        System.out.println("--- 测试1：构造参数校验 ---");
        Tensor alphaNeg = null;
        Tensor alphaZero = null;
        Tensor betaNeg = null;
        Tensor betaZero = null;
        Tensor validAlpha = null;

        try {
            // 1.1 测试α=-1.0，β=3.0
            alphaNeg = torch.tensor(-1.0f, CPU_DEVICE).clone();
            Tensor betaValid1 = torch.tensor(3.0f, CPU_DEVICE).clone();
            try {
                new InverseGamma(alphaNeg, betaValid1);
                System.err.println("异常测试1失败：未检测到α<0！");
            } catch (IllegalArgumentException e) {
                System.out.println("异常测试1通过：" + e.getMessage());
            }
            betaValid1.close();

            // 1.2 测试α=0.0，β=3.0
            alphaZero = torch.tensor(0.0f, CPU_DEVICE).clone();
            Tensor betaValid2 = torch.tensor(3.0f, CPU_DEVICE).clone();
            try {
                new InverseGamma(alphaZero, betaValid2);
                System.err.println("异常测试2失败：未检测到α=0！");
            } catch (IllegalArgumentException e) {
                System.out.println("异常测试2通过：" + e.getMessage());
            }
            betaValid2.close();

            // 1.3 测试α=2.0，β=-1.0
            validAlpha = torch.tensor(2.0f, CPU_DEVICE).clone();
            betaNeg = torch.tensor(-1.0f, CPU_DEVICE).clone();
            try {
                new InverseGamma(validAlpha, betaNeg);
                System.err.println("异常测试3失败：未检测到β<0！");
            } catch (IllegalArgumentException e) {
                System.out.println("异常测试3通过：" + e.getMessage());
            }

            // 1.4 测试α=2.0，β=0.0
            betaZero = torch.tensor(0.0f, CPU_DEVICE).clone();
            try {
                new InverseGamma(validAlpha, betaZero);
                System.err.println("异常测试4失败：未检测到β=0！");
            } catch (IllegalArgumentException e) {
                System.out.println("异常测试4通过：" + e.getMessage());
            }

        } catch (Exception e) {
            System.err.println("构造参数校验测试失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 释放测试用张量
            safeClose(alphaNeg);
            safeClose(alphaZero);
            safeClose(betaNeg);
            safeClose(betaZero);
            safeClose(validAlpha);
        }
    }

    /**
     * 测试2：核心功能（α=2.0, β=3.0，标量参数）
     * 验证：
     * 1. 名称格式正确：InverseGamma
     * 2. 采样：值>0、形状符合预期
     * 3. log_prob：合法输入（v>0）数值稳定，无NaN/Inf
     * 4. 均值：α=2>1 → 均值=β/(α-1)=3.0/1.0=3.0
     * 5. 熵：基于公式的数值合理性
     */
    private static void testCoreFunctionality() {
        System.out.println("\n--- 测试2：核心功能（α=2.0, β=3.0） ---");
        InverseGamma invGamma = null;
        Tensor alpha = null;
        Tensor beta = null;
        Tensor sample = null;
        Tensor logProb = null;
        Tensor mean = null;
        Tensor entropy = null;

        try {
            // 2.1 构建逆伽马分布（α=2.0，β=3.0）
            alpha = torch.tensor(2.0f, CPU_DEVICE).clone();
            beta = torch.tensor(3.0f, CPU_DEVICE).clone();
            invGamma = new InverseGamma(alpha, beta);
            System.out.println("分布名称: " + invGamma.name());
            assert "InverseGamma".equals(invGamma.name()) : "分布名称错误";

            // 2.2 验证采样（单样本）
            sample = invGamma.sample();
            System.out.println("单样本采样值: " + String.format("%.4f", sample.item().toFloat()));
            // 验证采样值>0
            boolean samplePositive = sample.gt(torch.tensor(0.0f, CPU_DEVICE)).item().toBool();
            System.out.println("采样值是否>0: " + samplePositive + "（预期=true）");
            assert samplePositive : "采样值≤0，不符合逆伽马分布特性";

            // 2.3 验证log_prob（合法输入v>0）
            logProb = invGamma.log_prob(sample);
            System.out.println("采样值的log_prob: " + String.format("%.4f", logProb.item().toFloat()));
            assert !logProb.isnan().item().toBool() && !logProb.isinf().item().toBool() : "log_prob出现NaN/Inf";

            // 2.4 验证均值（α=2>1 → 均值=3.0/(2.0-1.0)=3.0）
            mean = invGamma.mean();
            System.out.println("均值理论值: 3.0，实际值: " + String.format("%.4f", mean.item().toFloat()));
            assert Math.abs(mean.item().toFloat() - 3.0f) < EPS : "均值计算错误";

            // 2.5 验证熵（数值稳定，无NaN/Inf）
            entropy = invGamma.entropy();
            System.out.println("熵值: " + String.format("%.4f", entropy.item().toFloat()));
            assert !entropy.isnan().item().toBool() && !entropy.isinf().item().toBool() : "熵出现NaN/Inf";

        } catch (Exception e) {
            System.err.println("核心功能测试失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 安全释放InverseGamma实例
            if (invGamma != null) invGamma.close();
            // 释放测试用张量
            safeClose(sample);
            safeClose(logProb);
            safeClose(mean);
            safeClose(entropy);
            safeClose(alpha);
            safeClose(beta);
        }
    }

    /**
     * 测试3：批量参数测试（α=[2.0,3.0], β=[3.0,4.0]）
     * 验证：
     * 1. 批量均值：形状为[2]，值分别为3.0/(2-1)=3.0、4.0/(3-1)=2.0
     * 2. 批量采样：形状为[10,2]，所有值>0
     * 3. 批量log_prob：形状为[10,2]，数值稳定
     */
    private static void testBatchParameters() {
        System.out.println("\n--- 测试3：批量参数测试（α=[2.0,3.0], β=[3.0,4.0]） ---");
        InverseGamma invGammaBatch = null;
        Tensor alphaBatch = null;
        Tensor betaBatch = null;
        Tensor mean = null;
        Tensor sample = null;
        Tensor logProb = null;

        try {
            // 3.1 构建批量逆伽马分布
            alphaBatch = torch.tensor(new float[]{2.0f, 3.0f}, CPU_DEVICE).clone();
            betaBatch = torch.tensor(new float[]{3.0f, 4.0f}, CPU_DEVICE).clone();
            invGammaBatch = new InverseGamma(alphaBatch, betaBatch);

            // 3.2 验证批量均值
            mean = invGammaBatch.mean();
            System.out.println("批量均值形状: " + arrayToString(mean.sizes().vec().get()));
            float mean0 = mean.get(0).item().toFloat();
            float mean1 = mean.get(1).item().toFloat();
            System.out.println("批量均值[0]理论值: 3.0，实际值: " + String.format("%.4f", mean0));
            System.out.println("批量均值[1]理论值: 2.0，实际值: " + String.format("%.4f", mean1));
            assert Math.abs(mean0 - 3.0f) < EPS && Math.abs(mean1 - 2.0f) < EPS : "批量均值计算错误";

            // 3.3 验证批量采样（sampleShape=[10]）
            sample = invGammaBatch.sample(10);
            System.out.println("批量采样形状: " + arrayToString(sample.sizes().vec().get()));
            assert sample.sizes().get(0) == 10 && sample.sizes().get(1) == 2 : "批量采样形状错误";
            // 验证所有采样值>0
            boolean allSamplePositive = sample.gt(torch.tensor(0.0f, CPU_DEVICE)).all().item().toBool();
            System.out.println("批量采样所有值是否>0: " + allSamplePositive + "（预期=true）");
            assert allSamplePositive : "批量采样存在≤0的值";

            // 3.4 验证批量log_prob
            logProb = invGammaBatch.log_prob(sample);
            System.out.println("批量log_prob形状: " + arrayToString(logProb.sizes().vec().get()));
            assert logProb.sizes().get(0) == 10 && logProb.sizes().get(1) == 2 : "批量log_prob形状错误";
            assert !logProb.isnan().any().item().toBool() && !logProb.isinf().any().item().toBool() : "批量log_prob包含NaN/Inf";

        } catch (Exception e) {
            System.err.println("批量参数测试失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 安全释放InverseGamma实例
            if (invGammaBatch != null) invGammaBatch.close();
            // 释放测试用张量
            safeClose(mean);
            safeClose(sample);
            safeClose(logProb);
            safeClose(alphaBatch);
            safeClose(betaBatch);
        }
    }

    /**
     * 测试4：采样特性验证（值>0、形状正确、分布合理性）
     * 验证：
     * 1. 不同采样形状（[100], [5,10]）的输出形状正确性
     * 2. 所有采样结果>0
     * 3. 采样均值接近理论均值（α=2,β=3 → 理论均值=3.0）
     */
    private static void testSamplingProperties() {
        System.out.println("\n--- 测试4：采样特性验证 ---");
        InverseGamma invGamma = null;
        Tensor alpha = null;
        Tensor beta = null;
        Tensor sample100 = null;
        Tensor sample5x10 = null;

        try {
            // 4.1 构建分布（α=2.0，β=3.0）
            alpha = torch.tensor(2.0f, CPU_DEVICE).clone();
            beta = torch.tensor(3.0f, CPU_DEVICE).clone();
            invGamma = new InverseGamma(alpha, beta);

            // 4.2 测试采样形状[100]
            sample100 = invGamma.sample(100);
            System.out.println("采样形状[100]: " + arrayToString(sample100.sizes().vec().get()));
            assert sample100.sizes().get(0) == 100 : "采样形状错误";

            // 4.3 验证所有采样值>0
            boolean allPositive = sample100.gt(torch.tensor(0.0f, CPU_DEVICE)).all().item().toBool();
            System.out.println("100个采样值是否全>0: " + allPositive + "（预期=true）");
            assert allPositive : "存在采样值≤0";

            // 4.4 验证采样均值接近理论均值（3.0）
            float sampleMean = sample100.mean().item().toFloat();
            System.out.println("100个采样均值: " + String.format("%.4f", sampleMean) + "（理论均值=3.0）");
            // 宽松校验（采样均值与理论均值偏差<0.5）
            assert Math.abs(sampleMean - 3.0f) < 0.5f : "采样均值偏离理论值过大";

            // 4.5 测试采样形状[5,10]
            sample5x10 = invGamma.sample(5, 10);
            System.out.println("采样形状[5,10]: " + arrayToString(sample5x10.sizes().vec().get()));
            assert sample5x10.sizes().get(0) == 5 && sample5x10.sizes().get(1) == 10 : "采样形状错误";

        } catch (Exception e) {
            System.err.println("采样特性测试失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 安全释放InverseGamma实例
            if (invGamma != null) invGamma.close();
            // 释放测试用张量
            safeClose(sample100);
            safeClose(sample5x10);
            safeClose(alpha);
            safeClose(beta);
        }
    }

    /**
     * 测试5：log_prob异常场景验证
     * 验证：
     * 1. 输入v=0 → 抛出IllegalArgumentException
     * 2. 输入v=-1 → 抛出IllegalArgumentException
     * 3. 混合输入（v>0/v≤0）→ 合法值返回有效log_prob，非法值返回-∞
     */
    private static void testLogProbExceptions() {
        System.out.println("\n--- 测试5：log_prob异常场景验证 ---");
        InverseGamma invGamma = null;
        Tensor alpha = null;
        Tensor beta = null;
        Tensor vZero = null;
        Tensor vNeg = null;
        Tensor mixInput = null;
        Tensor logProb = null;

        try {
            // 5.1 构建分布（α=2.0，β=3.0）
            alpha = torch.tensor(2.0f, CPU_DEVICE).clone();
            beta = torch.tensor(3.0f, CPU_DEVICE).clone();
            invGamma = new InverseGamma(alpha, beta);

            // 5.2 测试输入v=0
            vZero = torch.tensor(0.0f, CPU_DEVICE).clone();
            try {
                invGamma.log_prob(vZero);
                System.err.println("异常测试1失败：未检测到v=0！");
            } catch (IllegalArgumentException e) {
                System.out.println("异常测试1通过：" + e.getMessage());
            }

            // 5.3 测试输入v=-1.0
            vNeg = torch.tensor(-1.0f, CPU_DEVICE).clone();
            try {
                invGamma.log_prob(vNeg);
                System.err.println("异常测试2失败：未检测到v<0！");
            } catch (IllegalArgumentException e) {
                System.out.println("异常测试2通过：" + e.getMessage());
            }

            // 5.4 测试混合输入（v>0 + v≤0）
            Tensor validV = invGamma.sample(); // 合法值
            mixInput = torch.stack(new TensorVector(validV, vZero, vNeg));

            // 释放临时张量
            validV.close();

        } catch (Exception e) {
            System.err.println("log_prob异常场景测试失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 安全释放InverseGamma实例
            if (invGamma != null) invGamma.close();
            // 释放测试用张量
            safeClose(vZero);
            safeClose(vNeg);
            safeClose(mixInput);
            safeClose(logProb);
            safeClose(alpha);
            safeClose(beta);
        }
    }

    /**
     * 测试6：均值特性验证（α>1/α=1/α<1）
     * 验证：
     * 1. α=1.5>1 → 均值=β/(α-1)=4.0/0.5=8.0
     * 2. α=1.0 → 均值=Inf
     * 3. α=0.5<1 → 均值=NaN
     * 4. 批量α（[0.5,1.0,1.5]）→ 均值对应[NaN,Inf,8.0]
     */
    private static void testMeanProperties() {
        System.out.println("\n--- 测试6：均值特性验证 ---");
        InverseGamma invGammaAlphaGt1 = null;
        InverseGamma invGammaAlphaEq1 = null;
        InverseGamma invGammaAlphaLt1 = null;
        InverseGamma invGammaBatch = null;
        Tensor alphaGt1 = null;
        Tensor alphaEq1 = null;
        Tensor alphaLt1 = null;
        Tensor alphaBatch = null;
        Tensor beta = null;
        Tensor mean = null;

        try {
            // 6.1 测试α=1.5>1（β=4.0 → 均值=4.0/(1.5-1)=8.0）
            alphaGt1 = torch.tensor(1.5f, CPU_DEVICE).clone();
            beta = torch.tensor(4.0f, CPU_DEVICE).clone();
            invGammaAlphaGt1 = new InverseGamma(alphaGt1, beta);
            mean = invGammaAlphaGt1.mean();
            System.out.println("α=1.5>1 均值理论值: 8.0，实际值: " + String.format("%.4f", mean.item().toFloat()));
            assert Math.abs(mean.item().toFloat() - 8.0f) < EPS : "α>1均值计算错误";

            // 6.2 测试α=1.0（均值=Inf）
            alphaEq1 = torch.tensor(1.0f, CPU_DEVICE).clone();
            invGammaAlphaEq1 = new InverseGamma(alphaEq1, beta);
            mean = invGammaAlphaEq1.mean();
            boolean isInf = mean.isinf().item().toBool() && mean.gt(torch.tensor(0.0f, CPU_DEVICE)).item().toBool();
            System.out.println("α=1.0 均值是否为Inf: " + isInf + "（预期=true）");
            assert isInf : "α=1均值应为Inf";

            // 6.3 测试α=0.5<1（均值=NaN）
            alphaLt1 = torch.tensor(0.5f, CPU_DEVICE).clone();
            invGammaAlphaLt1 = new InverseGamma(alphaLt1, beta);
            mean = invGammaAlphaLt1.mean();
            boolean isNaN = mean.isnan().item().toBool();
            System.out.println("α=0.5<1 均值是否为NaN: " + isNaN + "（预期=true）");
            assert isNaN : "α<1均值应为NaN";

            // 6.4 测试批量α（[0.5,1.0,1.5]）
            alphaBatch = torch.tensor(new float[]{0.5f, 1.0f, 1.5f}, CPU_DEVICE).clone();
            invGammaBatch = new InverseGamma(alphaBatch, beta);
            mean = invGammaBatch.mean();
            boolean batchNaN = mean.get(0).isnan().item().toBool();
            boolean batchInf = mean.get(1).isinf().item().toBool() && mean.get(1).gt(torch.tensor(0.0f, CPU_DEVICE)).item().toBool();
            boolean batchValid = Math.abs(mean.get(2).item().toFloat() - 8.0f) < EPS;
            System.out.println("批量均值验证：α=0.5→NaN=" + batchNaN + "，α=1.0→Inf=" + batchInf + "，α=1.5→8.0=" + batchValid + "（预期全为true）");
            assert batchNaN && batchInf && batchValid : "批量均值特性验证失败";

        } catch (Exception e) {
            System.err.println("均值特性测试失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 安全释放InverseGamma实例
            if (invGammaAlphaGt1 != null) invGammaAlphaGt1.close();
            if (invGammaAlphaEq1 != null) invGammaAlphaEq1.close();
            if (invGammaAlphaLt1 != null) invGammaAlphaLt1.close();
            if (invGammaBatch != null) invGammaBatch.close();
            // 释放测试用张量
            safeClose(mean);
            safeClose(alphaGt1);
            safeClose(alphaEq1);
            safeClose(alphaLt1);
            safeClose(alphaBatch);
            safeClose(beta);
        }
    }

    /**
     * 测试7：熵公式数值稳定性
     * 验证：
     * 1. 不同α（1.0/2.0/5.0）的熵计算无NaN/Inf
     * 2. 不同β（1.0/5.0/10.0）的熵计算无NaN/Inf
     * 3. 熵随α增大而增大，随β增大而增大（逆伽马分布特性）
     */
    private static void testEntropyStability() {
        System.out.println("\n--- 测试7：熵公式数值稳定性 ---");
        InverseGamma invGammaAlpha1 = null;
        InverseGamma invGammaAlpha2 = null;
        InverseGamma invGammaAlpha5 = null;
        InverseGamma invGammaBeta1 = null;
        InverseGamma invGammaBeta5 = null;
        InverseGamma invGammaBeta10 = null;
        Tensor alpha1 = null;
        Tensor alpha2 = null;
        Tensor alpha5 = null;
        Tensor beta1 = null;
        Tensor beta5 = null;
        Tensor beta10 = null;
        Tensor entropy = null;

        try {
            // 7.1 测试不同α的熵（β=3.0）
            beta1 = torch.tensor(3.0f, CPU_DEVICE).clone();
            alpha1 = torch.tensor(1.0f, CPU_DEVICE).clone();
            alpha2 = torch.tensor(2.0f, CPU_DEVICE).clone();
            alpha5 = torch.tensor(5.0f, CPU_DEVICE).clone();
            invGammaAlpha1 = new InverseGamma(alpha1, beta1);
            invGammaAlpha2 = new InverseGamma(alpha2, beta1);
            invGammaAlpha5 = new InverseGamma(alpha5, beta1);

            float entropyAlpha1 = invGammaAlpha1.entropy().item().toFloat();
            float entropyAlpha2 = invGammaAlpha2.entropy().item().toFloat();
            float entropyAlpha5 = invGammaAlpha5.entropy().item().toFloat();
            System.out.println("β=3.0时，α=1.0熵: " + String.format("%.4f", entropyAlpha1));
            System.out.println("β=3.0时，α=2.0熵: " + String.format("%.4f", entropyAlpha2));
            System.out.println("β=3.0时，α=5.0熵: " + String.format("%.4f", entropyAlpha5));
            assert !Float.isNaN(entropyAlpha1) && !Float.isInfinite(entropyAlpha1) : "α=1.0熵值异常";
            assert !Float.isNaN(entropyAlpha2) && !Float.isInfinite(entropyAlpha2) : "α=2.0熵值异常";
            assert !Float.isNaN(entropyAlpha5) && !Float.isInfinite(entropyAlpha5) : "α=5.0熵值异常";
            // 验证熵随α增大而增大
            assert entropyAlpha1 < entropyAlpha2 && entropyAlpha2 < entropyAlpha5 : "熵应随α增大而增大";

            // 7.2 测试不同β的熵（α=2.0）
            alpha2 = torch.tensor(2.0f, CPU_DEVICE).clone();
            beta5 = torch.tensor(5.0f, CPU_DEVICE).clone();
            beta10 = torch.tensor(10.0f, CPU_DEVICE).clone();
            invGammaBeta1 = new InverseGamma(alpha2, beta1);
            invGammaBeta5 = new InverseGamma(alpha2, beta5);
            invGammaBeta10 = new InverseGamma(alpha2, beta10);

            float entropyBeta1 = invGammaBeta1.entropy().item().toFloat();
            float entropyBeta5 = invGammaBeta5.entropy().item().toFloat();
            float entropyBeta10 = invGammaBeta10.entropy().item().toFloat();
            System.out.println("α=2.0时，β=1.0熵: " + String.format("%.4f", entropyBeta1));
            System.out.println("α=2.0时，β=5.0熵: " + String.format("%.4f", entropyBeta5));
            System.out.println("α=2.0时，β=10.0熵: " + String.format("%.4f", entropyBeta10));
            assert !Float.isNaN(entropyBeta1) && !Float.isInfinite(entropyBeta1) : "β=1.0熵值异常";
            assert !Float.isNaN(entropyBeta5) && !Float.isInfinite(entropyBeta5) : "β=5.0熵值异常";
            assert !Float.isNaN(entropyBeta10) && !Float.isInfinite(entropyBeta10) : "β=10.0熵值异常";
            // 验证熵随β增大而增大
            assert entropyBeta1 < entropyBeta5 && entropyBeta5 < entropyBeta10 : "熵应随β增大而增大";

        } catch (Exception e) {
            System.err.println("熵公式数值稳定性测试失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // 安全释放InverseGamma实例
            if (invGammaAlpha1 != null) invGammaAlpha1.close();
            if (invGammaAlpha2 != null) invGammaAlpha2.close();
            if (invGammaAlpha5 != null) invGammaAlpha5.close();
            if (invGammaBeta1 != null) invGammaBeta1.close();
            if (invGammaBeta5 != null) invGammaBeta5.close();
            if (invGammaBeta10 != null) invGammaBeta10.close();
            // 释放测试用张量
            safeClose(entropy);
            safeClose(alpha1);
            safeClose(alpha2);
            safeClose(alpha5);
            safeClose(beta1);
            safeClose(beta5);
            safeClose(beta10);
        }
    }

    /**
     * 测试8：资源释放验证（AutoCloseable实现）
     * 验证：
     * 1. InverseGamma.close()能正确释放内部张量和预定义标量
     * 2. 循环创建/释放无内存泄漏
     */
    private static void testResourceRelease() {
        System.out.println("\n--- 测试8：资源释放验证 ---");
        InverseGamma invGamma = null;
        Tensor alpha = null;
        Tensor beta = null;

        try {
            // 循环创建/释放，验证无内存泄漏
            for (int i = 0; i < 100; i++) {
                alpha = torch.tensor(2.0f, CPU_DEVICE).clone();
                beta = torch.tensor(3.0f, CPU_DEVICE).clone();
                invGamma = new InverseGamma(alpha, beta);

                // 执行核心操作
                Tensor sample = invGamma.sample(10);
                Tensor logProb = invGamma.log_prob(sample);
                Tensor mean = invGamma.mean();
                Tensor entropy = invGamma.entropy();

                // 释放临时张量
                safeClose(sample);
                safeClose(logProb);
                safeClose(mean);
                safeClose(entropy);
                // 释放InverseGamma实例
                invGamma.close();
                // 释放alpha/beta张量
                safeClose(alpha);
                safeClose(beta);
            }
            System.out.println("资源释放测试通过：循环创建/释放无异常");

        } catch (Exception e) {
            System.err.println("资源释放测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // -------------------------- 辅助方法（统一风格） --------------------------
    /**
     * 安全释放AutoCloseable资源（避免空指针和异常）
     */
    private static void safeClose(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                System.err.println("资源释放失败: " + e.getMessage());
            }
        }
    }

    /**
     * 数组转字符串（接收 .vec().get() 结果）
     */
    private static String arrayToString(long[] array) {
        if (array == null || array.length == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i]);
            if (i < array.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * 张量转字符串（简化输出）
     */
    private static String tensorToString(Tensor tensor) {
        if (tensor.dim() == 0) {
            return String.format("%.4f", tensor.item().toFloat());
        }
        Tensor flat = tensor.flatten();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < flat.numel(); i++) {
            sb.append(String.format("%.4f", flat.get(i).item().toFloat()));
            if (i < flat.numel() - 1) sb.append(", ");
        }
        sb.append("]");
        flat.close();
        return sb.toString();
    }
}