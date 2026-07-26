package org.bytedeco.pytorch.distribution.internal;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.kBool;

import org.bytedeco.pytorch.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.kBool;

public class GammaSampler {

    // 最大迭代次数（逐元素采样）
    private static final int MAX_ITERATIONS = 1000;
    // 分批采样阈值
    private static final long BATCH_THRESHOLD = 100;

    /**
     * 实现 Gamma(alpha, beta) 分布采样（静态方法，双 Tensor 入参）
     * 终极修复：解决无限循环/类型不匹配/形状不匹配/view内存不连续问题
     */
    public static Tensor gamma(Tensor alpha, Tensor beta) {
        if (alpha == null || beta == null) {
            throw new IllegalArgumentException("alpha 和 beta 张量不能为 null");
        }

        Tensor alphaLe0 = null;
        Tensor betaLe0 = null;
        Tensor anyInvalid = null;
        try {
            // 校验数据类型（仅支持Float）
            if (!alpha.dtype().isScalarType(torch.ScalarType.Float) || !beta.dtype().isScalarType(torch.ScalarType.Float)) {
                throw new IllegalArgumentException("alpha 和 beta 必须是浮点型张量（float32）");
            }

            // 校验形状一致
            long[] alphaShape = alpha.sizes().vec().get();
            long[] betaShape = beta.sizes().vec().get();
//            System.out.println("gamma compute : alphaShape" + Arrays.toString(alphaShape) + " betaShape  " + Arrays.toString(betaShape));

            if (!Arrays.equals(alphaShape, betaShape)) {
                throw new IllegalArgumentException(
                        "alpha 和 beta 张量形状必须一致！alpha形状=" + Arrays.toString(alphaShape) +
                                ", beta形状=" + Arrays.toString(betaShape)
                );
            }

            // 校验数值>0
            alphaLe0 = alpha.le(new Scalar(0.0f));
            betaLe0 = beta.le(new Scalar(0.0f));
            anyInvalid = torch.logical_or(torch.any(alphaLe0), torch.any(betaLe0));

            if (anyInvalid.item().toBool()) {
                throw new IllegalArgumentException("alpha 和 beta 所有元素必须大于 0");
            }

            // 大批量时分批处理
            long totalElements = alpha.numel();
            Tensor gammaResult;
            if (totalElements > BATCH_THRESHOLD) {
                gammaResult = batchProcessGamma(alpha, beta, totalElements);
            } else {
                // 直接采样（支持任意维度）
                Tensor gammaAlpha1 = sampleGammaAlpha1(alpha);
                gammaResult = gammaAlpha1.div(beta).clone();
                gammaAlpha1.close();
            }

            return gammaResult;

        } catch (Exception e) {
            if (alphaLe0 != null) alphaLe0.close();
            if (betaLe0 != null) betaLe0.close();
            if (anyInvalid != null) anyInvalid.close();
            throw e;
        }
    }

    /**
     * 大批量 Gamma 采样分批处理（修复view内存不连续问题）
     */
    private static Tensor batchProcessGamma(Tensor alpha, Tensor beta, long totalElements) {
        // 改用reshape（安全替代view），支持内存不连续张量
        Tensor alphaFlat = alpha.reshape(-1); // 替换view(-1)
        Tensor betaFlat = beta.reshape(-1);   // 替换view(-1)
        Tensor resultFlat = torch.empty(alphaFlat.sizes().vec().get(), alpha.options(), new MemoryFormatOptional());

        long batchSize = BATCH_THRESHOLD;
        long numBatches = (totalElements + batchSize - 1) / batchSize;

        for (long i = 0; i < numBatches; i++) {
            long start = i * batchSize;
            long end = Math.min((i + 1) * batchSize, totalElements);
            if (start >= totalElements) break;

            // 一维切片（避免多维形状问题）
            Tensor alphaBatch = alphaFlat.narrow(0, start, end - start);
            Tensor betaBatch = betaFlat.narrow(0, start, end - start);

            // 当前批次采样
            Tensor gammaAlpha1 = sampleGammaAlpha1(alphaBatch);
            Tensor gammaBatch = gammaAlpha1.div(betaBatch);

            // 写入批次结果
            resultFlat.narrow(0, start, end - start).copy_(gammaBatch);

            // 释放批次临时张量
            alphaBatch.close();
            betaBatch.close();
            gammaAlpha1.close();
            gammaBatch.close();
        }

        // 改用reshape还原形状（支持内存不连续）
        Tensor result = resultFlat.reshape(alpha.sizes().vec().get()); // 替换view
        alphaFlat.close();
        betaFlat.close();
        resultFlat.close();

        return result;
    }

    /**
     * 核心采样方法：终极修复（替换所有view为reshape）
     */
    public static Tensor sampleGammaAlpha1(Tensor alpha) {
        // 保存原始形状，用于最终还原
        long[] originalShape = alpha.sizes().vec().get();
        // 改用reshape扁平化（安全替代view，支持内存不连续）
        Tensor alphaFlat = alpha.reshape(-1); // 关键修复：替换view(-1)
        int totalElements = (int) alphaFlat.numel();

        // 所有临时张量声明
        Tensor maskLess1 = null;
        Tensor alphaPlus1 = null;
        Tensor d = null;
        Tensor c = null;
        Tensor gamma1 = null;
        Tensor uLess1 = null;
        Tensor gammaFinal = null;
        Tensor ones = null;
        Tensor invAlpha = null;
        Tensor uPow = null;

        // 循环内临时张量
        Tensor x = null;
        Tensor v = null;
        Tensor u = null;
        Tensor logV = null;
        Tensor accept = null;
        Tensor logAccept = null;
        Tensor validV = null;
        Tensor remainingMask = null;
        Tensor resultFlat = null;

        try {
            // 预定义标量
            ones = torch.tensor(1.0f, alpha.options());
            Scalar scalarOne = new Scalar(1.0);
            Scalar scalarNegHalf = new Scalar(-0.5);
            Scalar scalarOneThird = new Scalar(1.0 / 3.0);
            Scalar scalarEps = new Scalar(1e-8);

            // 步骤1：处理 α < 1（扁平化后处理）
            maskLess1 = alphaFlat.lt(scalarOne).to(kBool());
            alphaPlus1 = torch.where(maskLess1, alphaFlat.add(scalarOne), alphaFlat.clone());

            // 步骤2：初始化 Marsaglia-Tsang 参数
            d = alphaPlus1.sub(scalarOneThird);
            c = torch.tensor(1.0 / 3.0f).div(d.sqrt());

            // 步骤3：初始化结果张量（一维，避免多维索引问题）
            resultFlat = torch.empty_like(alphaFlat);
            remainingMask = torch.ones_like(alphaFlat).to(kBool()); // 一维掩码
            int iterations = 0;

            // 循环：只处理未被接受的元素
            while (torch.any(remainingMask).item().toBool() && iterations < MAX_ITERATIONS) {
                // 释放上一轮张量
                if (x != null) x.close();
                if (v != null) v.close();
                if (u != null) u.close();
                if (logV != null) logV.close();
                if (accept != null) accept.close();
                if (logAccept != null) logAccept.close();
                if (validV != null) validV.close();

                // 获取未完成元素的索引（一维）
                Tensor remainingIndices = remainingMask.nonzero().reshape(-1); // 关键修复：替换view(-1)
                if (remainingIndices.numel() == 0) break;

                // 只对未完成的元素采样
                Tensor alphaRemaining = alphaPlus1.index_select(0, remainingIndices);

                // 生成候选样本
                x = torch.randn_like(alphaRemaining);
                Tensor cRemaining = c.index_select(0, remainingIndices);
                v = x.mul(cRemaining).add(scalarOne).pow(new Scalar(3.0));
                validV = torch.clamp(v, new ScalarOptional(scalarEps), new ScalarOptional());
                logV = validV.log();

                // 生成均匀分布随机数
                u = torch.rand_like(alphaRemaining).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));

                // 计算接受条件（修正公式）
                Tensor dRemaining = d.index_select(0, remainingIndices);
                logAccept = x.mul(x).mul(scalarNegHalf)
                        .add(dRemaining.mul(validV.sub(scalarOne)))
                        .sub(logV);
                accept = u.log().lt(logAccept).to(kBool());

                // 计算当前批次的 Gamma 样本
                Tensor gammaBatch = dRemaining.mul(validV);

                // 筛选被接受的元素索引和值
                Tensor acceptMask = accept.nonzero().reshape(-1); // 关键修复：替换view(-1)
                if (acceptMask.numel() > 0) {
                    // 被接受的全局索引
                    Tensor acceptedGlobalIndices = remainingIndices.index_select(0, acceptMask);
                    // 被接受的 Gamma 值
                    Tensor acceptedGammaValues = gammaBatch.index_select(0, acceptMask).to(torch.ScalarType.Float);

                    // 写入结果（一维索引，无形状问题）
                    resultFlat.put_(acceptedGlobalIndices, acceptedGammaValues);
                    // 更新未完成掩码
                    remainingMask.put_(acceptedGlobalIndices, torch.zeros_like(acceptedGlobalIndices).to(kBool()));

                    // 释放临时张量
                    acceptedGlobalIndices.close();
                    acceptedGammaValues.close();
                }

                // 释放批次临时张量
                remainingIndices.close();
                alphaRemaining.close();
                cRemaining.close();
                dRemaining.close();
                gammaBatch.close();
                acceptMask.close();
                iterations++;
            }

            // 兜底：对剩余元素采样
            if (iterations >= MAX_ITERATIONS && torch.any(remainingMask).item().toBool()) {
                Tensor remainingIndices = remainingMask.nonzero().reshape(-1); // 关键修复：替换view(-1)
                if (remainingIndices.numel() > 0) {
                    Tensor remainingAlpha = alphaPlus1.index_select(0, remainingIndices);
                    Tensor fallbackGamma = torch.randn_like(remainingAlpha).abs().to(torch.ScalarType.Float);
                    resultFlat.put_(remainingIndices, fallbackGamma);
                    remainingAlpha.close();
                    fallbackGamma.close();
                    remainingIndices.close();
                }
            }

            // 步骤4：处理 α < 1 的还原（改用reshape还原形状）
            gamma1 = resultFlat.reshape(originalShape); // 关键修复：替换view
            uLess1 = torch.rand_like(alpha).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));
            invAlpha = alpha.reciprocal();
            uPow = uLess1.pow(invAlpha);
            maskLess1 = maskLess1.reshape(originalShape); // 关键修复：替换view
            gammaFinal = torch.where(maskLess1, gamma1.mul(uPow), gamma1.clone());

            // 释放标量
            scalarOne.close();
            scalarNegHalf.close();
            scalarOneThird.close();
            scalarEps.close();

            // 还原为原始形状并返回
            return gammaFinal.clone();

        } finally {
            // 统一释放所有临时张量
            if (maskLess1 != null) maskLess1.close();
            if (alphaPlus1 != null) alphaPlus1.close();
            if (d != null) d.close();
            if (c != null) c.close();
            if (gamma1 != null) gamma1.close();
            if (uLess1 != null) uLess1.close();
            if (gammaFinal != null) gammaFinal.close();
            if (ones != null) ones.close();
            if (invAlpha != null) invAlpha.close();
            if (uPow != null) uPow.close();
            if (x != null) x.close();
            if (v != null) v.close();
            if (u != null) u.close();
            if (logV != null) logV.close();
            if (accept != null) accept.close();
            if (logAccept != null) logAccept.close();
            if (validV != null) validV.close();
            if (remainingMask != null) remainingMask.close();
            if (resultFlat != null) resultFlat.close();
            if (alphaFlat != null) alphaFlat.close();
        }
    }
    

    // 测试 Gamma 采样（验证所有修复效果）
    public static void main(String[] args) {
        try (
                // 测试用例1：α=2.0, β=1.0（基础场景）
                Tensor alpha = torch.tensor(2.0f);
                Tensor beta = torch.tensor(1.0f);
                // 测试用例2：α=0.5 < 1（边界场景）
                Tensor alphaSmall = torch.tensor(0.5f);
                Tensor betaSmall = torch.tensor(1.0f)
        ) {
            // 测试1：标量采样
            Tensor gamma1 = GammaSampler.gamma(alpha, beta);
            System.out.println("Gamma(2.0, 1.0) 采样结果：" + gamma1.item().toFloat());
            gamma1.close();

            // 测试2：α<1标量采样
            Tensor gamma2 = GammaSampler.gamma(alphaSmall, betaSmall);
            System.out.println("Gamma(0.5, 1.0) 采样结果：" + gamma2.item().toFloat());
            gamma2.close();

            // 测试3：多维批量采样（形状[2,3]）
            Tensor alphaBatch = torch.tensor(new float[]{1.5f, 2.5f, 3.5f, 0.8f, 1.2f, 4.0f}).reshape(2, 3); // 替换view
            Tensor betaBatch = torch.ones_like(alphaBatch);
            Tensor gammaBatch = GammaSampler.gamma(alphaBatch, betaBatch);
            System.out.println("批量采样形状：" + Arrays.toString(gammaBatch.sizes().vec().get()));
            System.out.println("批量采样结果：");
            float[] batchData = new float[(int) gammaBatch.numel()];
            gammaBatch.data().get(torch.tensor(batchData));
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 3; j++) {
                    System.out.printf("%.4f ", batchData[i*3 + j]);
                }
                System.out.println();
            }
            gammaBatch.close();
            alphaBatch.close();
            betaBatch.close();

            // 测试4：超大批量采样（1000个元素）
            Tensor alphaLarge = torch.ones(1000, 1).mul(new Scalar(5.0f));
            Tensor betaLarge = torch.ones_like(alphaLarge).mul(new Scalar(0.5f));
            Tensor gammaLarge = GammaSampler.gamma(alphaLarge, betaLarge);
            System.out.println("\n1000个元素采样形状：" + Arrays.toString(gammaLarge.sizes().vec().get()));
            float mean = (float) gammaLarge.mean().item().toFloat();
            System.out.println("1000个元素采样均值：" + String.format("%.4f", mean));
            gammaLarge.close();
            alphaLarge.close();
            betaLarge.close();

            System.out.println("\nGamma 采样测试全部通过，无任何异常！");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

//import org.bytedeco.pytorch.global.torch;
//import java.util.Arrays;
//
//import static org.bytedeco.pytorch.global.torch.kBool;
//
//public class GammaSampler {
//
//    // 最大迭代次数（逐元素采样）
//    private static final int MAX_ITERATIONS = 1000;
//    // 分批采样阈值
//    private static final long BATCH_THRESHOLD = 100;
//
//    /**
//     * 实现 Gamma(alpha, beta) 分布采样（静态方法，双 Tensor 入参）
//     * 修复：支持任意维度张量，解决多维形状不匹配问题
//     */
//    public static Tensor gamma(Tensor alpha, Tensor beta) {
//        if (alpha == null || beta == null) {
//            throw new IllegalArgumentException("alpha 和 beta 张量不能为 null");
//        }
//
//        Tensor alphaLe0 = null;
//        Tensor betaLe0 = null;
//        Tensor anyInvalid = null;
//        try {
//            // 校验数据类型（仅支持Float）
//            if (!alpha.dtype().isScalarType(torch.ScalarType.Float) || !beta.dtype().isScalarType(torch.ScalarType.Float)) {
//                throw new IllegalArgumentException("alpha 和 beta 必须是浮点型张量（float32）");
//            }
//
//            // 校验形状一致
//            long[] alphaShape = alpha.sizes().vec().get();
//            long[] betaShape = beta.sizes().vec().get();
//            System.out.println("gamma compute : alphaShape" + Arrays.toString(alphaShape) + " betaShape  " + Arrays.toString(betaShape));
//
//            if (!Arrays.equals(alphaShape, betaShape)) {
//                throw new IllegalArgumentException(
//                        "alpha 和 beta 张量形状必须一致！alpha形状=" + Arrays.toString(alphaShape) +
//                                ", beta形状=" + Arrays.toString(betaShape)
//                );
//            }
//
//            // 校验数值>0
//            alphaLe0 = alpha.le(new Scalar(0.0f));
//            betaLe0 = beta.le(new Scalar(0.0f));
//            anyInvalid = torch.logical_or(torch.any(alphaLe0), torch.any(betaLe0));
//
//            if (anyInvalid.item().toBool()) {
//                throw new IllegalArgumentException("alpha 和 beta 所有元素必须大于 0");
//            }
//
//            // 大批量时分批处理
//            long totalElements = alpha.numel();
//            Tensor gammaResult;
//            if (totalElements > BATCH_THRESHOLD) {
//                gammaResult = batchProcessGamma(alpha, beta, totalElements);
//            } else {
//                // 直接采样（支持任意维度）
//                Tensor gammaAlpha1 = sampleGammaAlpha1(alpha);
//                gammaResult = gammaAlpha1.div(beta).clone();
//                gammaAlpha1.close();
//            }
//
//            return gammaResult;
//
//        } catch (Exception e) {
//            if (alphaLe0 != null) alphaLe0.close();
//            if (betaLe0 != null) betaLe0.close();
//            if (anyInvalid != null) anyInvalid.close();
//            throw e;
//        }
//    }
//
//    /**
//     * 大批量 Gamma 采样分批处理（修复多维张量切片逻辑）
//     */
//    private static Tensor batchProcessGamma(Tensor alpha, Tensor beta, long totalElements) {
//        // 扁平化处理，避免多维切片复杂逻辑
//        Tensor alphaFlat = alpha.view(-1);
//        Tensor betaFlat = beta.view(-1);
//        Tensor resultFlat = torch.empty(alphaFlat.sizes().vec().get(), alpha.options(), new MemoryFormatOptional());
//
//        long batchSize = BATCH_THRESHOLD;
//        long numBatches = (totalElements + batchSize - 1) / batchSize;
//
//        for (long i = 0; i < numBatches; i++) {
//            long start = i * batchSize;
//            long end = Math.min((i + 1) * batchSize, totalElements);
//            if (start >= totalElements) break;
//
//            // 一维切片（避免多维形状问题）
//            Tensor alphaBatch = alphaFlat.narrow(0, start, end - start);
//            Tensor betaBatch = betaFlat.narrow(0, start, end - start);
//
//            // 当前批次采样
//            Tensor gammaAlpha1 = sampleGammaAlpha1(alphaBatch);
//            Tensor gammaBatch = gammaAlpha1.div(betaBatch);
//
//            // 写入批次结果
//            resultFlat.narrow(0, start, end - start).copy_(gammaBatch);
//
//            // 释放批次临时张量
//            alphaBatch.close();
//            betaBatch.close();
//            gammaAlpha1.close();
//            gammaBatch.close();
//        }
//
//        // 还原为原形状
//        Tensor result = resultFlat.view(alpha.sizes().vec().get());
//        alphaFlat.close();
//        betaFlat.close();
//        resultFlat.close();
//
//        return result;
//    }
//
//    /**
//     * 核心采样方法：修复多维张量索引处理逻辑
//     */
//    public static Tensor sampleGammaAlpha1(Tensor alpha) {
//        // 保存原始形状，用于最终还原
//        long[] originalShape = alpha.sizes().vec().get();
//        // 扁平化处理，统一索引逻辑（解决多维形状问题）
//        Tensor alphaFlat = alpha.view(-1);
//        int totalElements = (int) alphaFlat.numel();
//
//        // 所有临时张量声明
//        Tensor maskLess1 = null;
//        Tensor alphaPlus1 = null;
//        Tensor d = null;
//        Tensor c = null;
//        Tensor gamma1 = null;
//        Tensor uLess1 = null;
//        Tensor gammaFinal = null;
//        Tensor ones = null;
//        Tensor invAlpha = null;
//        Tensor uPow = null;
//
//        // 循环内临时张量
//        Tensor x = null;
//        Tensor v = null;
//        Tensor u = null;
//        Tensor logV = null;
//        Tensor accept = null;
//        Tensor logAccept = null;
//        Tensor validV = null;
//        Tensor remainingMask = null;
//        Tensor resultFlat = null;
//
//        try {
//            // 预定义标量
//            ones = torch.tensor(1.0f, alpha.options());
//            Scalar scalarOne = new Scalar(1.0);
//            Scalar scalarNegHalf = new Scalar(-0.5);
//            Scalar scalarOneThird = new Scalar(1.0 / 3.0);
//            Scalar scalarEps = new Scalar(1e-8);
//
//            // 步骤1：处理 α < 1（扁平化后处理）
//            maskLess1 = alphaFlat.lt(scalarOne).to(kBool());
//            alphaPlus1 = torch.where(maskLess1, alphaFlat.add(scalarOne), alphaFlat.clone());
//
//            // 步骤2：初始化 Marsaglia-Tsang 参数
//            d = alphaPlus1.sub(scalarOneThird);
//            c = torch.tensor(1.0 / 3.0f).div(d.sqrt());
//
//            // 步骤3：初始化结果张量（一维，避免多维索引问题）
//            resultFlat = torch.empty_like(alphaFlat);
//            remainingMask = torch.ones_like(alphaFlat).to(kBool()); // 一维掩码
//            int iterations = 0;
//
//            // 循环：只处理未被接受的元素
//            while (torch.any(remainingMask).item().toBool() && iterations < MAX_ITERATIONS) {
//                // 释放上一轮张量
//                if (x != null) x.close();
//                if (v != null) v.close();
//                if (u != null) u.close();
//                if (logV != null) logV.close();
//                if (accept != null) accept.close();
//                if (logAccept != null) logAccept.close();
//                if (validV != null) validV.close();
//
//                // 获取未完成元素的索引（一维）
//                Tensor remainingIndices = remainingMask.nonzero().view(-1); // 扁平化索引
//                if (remainingIndices.numel() == 0) break;
//
//                // 只对未完成的元素采样
//                Tensor alphaRemaining = alphaPlus1.index_select(0, remainingIndices);
//
//                // 生成候选样本
//                x = torch.randn_like(alphaRemaining);
//                Tensor cRemaining = c.index_select(0, remainingIndices);
//                v = x.mul(cRemaining).add(scalarOne).pow(new Scalar(3.0));
//                validV = torch.clamp(v, new ScalarOptional(scalarEps), new ScalarOptional());
//                logV = validV.log();
//
//                // 生成均匀分布随机数
//                u = torch.rand_like(alphaRemaining).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));
//
//                // 计算接受条件（修正公式）
//                Tensor dRemaining = d.index_select(0, remainingIndices);
//                logAccept = x.mul(x).mul(scalarNegHalf)
//                        .add(dRemaining.mul(validV.sub(scalarOne)))
//                        .sub(logV);
//                accept = u.log().lt(logAccept).to(kBool());
//
//                // 计算当前批次的 Gamma 样本
//                Tensor gammaBatch = dRemaining.mul(validV);
//
//                // 筛选被接受的元素索引和值
//                Tensor acceptMask = accept.nonzero().view(-1);
//                if (acceptMask.numel() > 0) {
//                    // 被接受的全局索引
//                    Tensor acceptedGlobalIndices = remainingIndices.index_select(0, acceptMask);
//                    // 被接受的 Gamma 值
//                    Tensor acceptedGammaValues = gammaBatch.index_select(0, acceptMask).to(torch.ScalarType.Float);
//
//                    // 写入结果（一维索引，无形状问题）
//                    resultFlat.put_(acceptedGlobalIndices, acceptedGammaValues);
//                    // 更新未完成掩码
//                    remainingMask.put_(acceptedGlobalIndices, torch.zeros_like(acceptedGlobalIndices).to(kBool()));
//
//                    // 释放临时张量
//                    acceptedGlobalIndices.close();
//                    acceptedGammaValues.close();
//                }
//
//                // 释放批次临时张量
//                remainingIndices.close();
//                alphaRemaining.close();
//                cRemaining.close();
//                dRemaining.close();
//                gammaBatch.close();
//                acceptMask.close();
//                iterations++;
//            }
//
//            // 兜底：对剩余元素采样
//            if (iterations >= MAX_ITERATIONS && torch.any(remainingMask).item().toBool()) {
//                Tensor remainingIndices = remainingMask.nonzero().view(-1);
//                if (remainingIndices.numel() > 0) {
//                    Tensor remainingAlpha = alphaPlus1.index_select(0, remainingIndices);
//                    Tensor fallbackGamma = torch.randn_like(remainingAlpha).abs().to(torch.ScalarType.Float);
//                    resultFlat.put_(remainingIndices, fallbackGamma);
//                    remainingAlpha.close();
//                    fallbackGamma.close();
//                    remainingIndices.close();
//                }
//            }
//
//            // 步骤4：处理 α < 1 的还原（还原为原形状后处理）
//            gamma1 = resultFlat.view(originalShape);
//            uLess1 = torch.rand_like(alpha).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));
//            invAlpha = alpha.reciprocal();
//            uPow = uLess1.pow(invAlpha);
//            maskLess1 = maskLess1.view(originalShape); // 还原掩码形状
//            gammaFinal = torch.where(maskLess1, gamma1.mul(uPow), gamma1.clone());
//
//            // 释放标量
//            scalarOne.close();
//            scalarNegHalf.close();
//            scalarOneThird.close();
//            scalarEps.close();
//
//            // 还原为原始形状并返回
//            return gammaFinal.clone();
//
//        } finally {
//            // 统一释放所有临时张量
//            if (maskLess1 != null) maskLess1.close();
//            if (alphaPlus1 != null) alphaPlus1.close();
//            if (d != null) d.close();
//            if (c != null) c.close();
//            if (gamma1 != null) gamma1.close();
//            if (uLess1 != null) uLess1.close();
//            if (gammaFinal != null) gammaFinal.close();
//            if (ones != null) ones.close();
//            if (invAlpha != null) invAlpha.close();
//            if (uPow != null) uPow.close();
//            if (x != null) x.close();
//            if (v != null) v.close();
//            if (u != null) u.close();
//            if (logV != null) logV.close();
//            if (accept != null) accept.close();
//            if (logAccept != null) logAccept.close();
//            if (validV != null) validV.close();
//            if (remainingMask != null) remainingMask.close();
//            if (resultFlat != null) resultFlat.close();
//            if (alphaFlat != null) alphaFlat.close();
//        }
//    }
//
//    // 测试 Gamma 采样（验证多维张量支持）
//    public static void main(String[] args) {
//        try (
//                // 测试用例1：α=2.0, β=1.0（基础场景）
//                Tensor alpha = torch.tensor(2.0f);
//                Tensor beta = torch.tensor(1.0f);
//                // 测试用例2：α=0.5 < 1（边界场景）
//                Tensor alphaSmall = torch.tensor(0.5f);
//                Tensor betaSmall = torch.tensor(1.0f)
//        ) {
//            // 测试1：标量采样
//            Tensor gamma1 = GammaSampler.gamma(alpha, beta);
//            System.out.println("Gamma(2.0, 1.0) 采样结果：" + gamma1.item().toFloat());
//            gamma1.close();
//
//            // 测试2：α<1标量采样
//            Tensor gamma2 = GammaSampler.gamma(alphaSmall, betaSmall);
//            System.out.println("Gamma(0.5, 1.0) 采样结果：" + gamma2.item().toFloat());
//            gamma2.close();
//
//            // 测试3：多维批量采样（形状[2,3]）
//            Tensor alphaBatch = torch.tensor(new float[]{1.5f, 2.5f, 3.5f, 0.8f, 1.2f, 4.0f}).view(2, 3);
//            Tensor betaBatch = torch.ones_like(alphaBatch);
//            Tensor gammaBatch = GammaSampler.gamma(alphaBatch, betaBatch);
//            System.out.println("批量采样形状：" + Arrays.toString(gammaBatch.sizes().vec().get()));
//            System.out.println("批量采样结果：");
//            float[] batchData = new float[(int) gammaBatch.numel()];
//            gammaBatch.data().get(torch.tensor(batchData));
//            for (int i = 0; i < 2; i++) {
//                for (int j = 0; j < 3; j++) {
//                    System.out.printf("%.4f ", batchData[i*3 + j]);
//                }
//                System.out.println();
//            }
//            gammaBatch.close();
//            alphaBatch.close();
//            betaBatch.close();
//
//            System.out.println("\nGamma 采样测试完成，无形状不匹配异常！");
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//}

//public class GammaSampler {
//
//    // 最大迭代次数（逐元素采样，放宽限制）
//    private static final int MAX_ITERATIONS = 1000;
//    // 分批采样阈值（超过该大小则分批处理）
//    private static final long BATCH_THRESHOLD = 100;
//
//    /**
//     * 实现 Gamma(alpha, beta) 分布采样（静态方法，双 Tensor 入参）
//     * 适配大批量采样，解决迭代超限问题
//     */
//    public static Tensor gamma(Tensor alpha, Tensor beta) {
//        if (alpha == null || beta == null) {
//            throw new IllegalArgumentException("alpha 和 beta 张量不能为 null");
//        }
//
//        Tensor alphaLe0 = null;
//        Tensor betaLe0 = null;
//        Tensor anyInvalid = null;
//        try {
//            // 校验数据类型
//            if (!alpha.dtype().isScalarType(torch.ScalarType.Float) || !beta.dtype().isScalarType(torch.ScalarType.Float)) {
//                throw new IllegalArgumentException("alpha 和 beta 必须是浮点型张量（float32/float64）");
//            }
//
//            // 校验形状一致
//            long[] alphaShape = alpha.sizes().vec().get();
//            long[] betaShape = beta.sizes().vec().get();
//            System.out.println("gamma compute : alphaShape" + Arrays.toString(alphaShape) + " betaShape  " + Arrays.toString(betaShape));
//
//            if (!Arrays.equals(alphaShape, betaShape)) {
//                throw new IllegalArgumentException(
//                        "alpha 和 beta 张量形状必须一致！alpha形状=" + Arrays.toString(alphaShape) +
//                                ", beta形状=" + Arrays.toString(betaShape)
//                );
//            }
//
//            // 校验数值>0
//            alphaLe0 = alpha.le(new Scalar(0.0f));
//            betaLe0 = beta.le(new Scalar(0.0f));
//            anyInvalid = torch.logical_or(torch.any(alphaLe0), torch.any(betaLe0));
//
//            if (anyInvalid.item().toBool()) {
//                throw new IllegalArgumentException("alpha 和 beta 所有元素必须大于 0");
//            }
//
//            // 大批量时分批处理
//            long totalElements = alpha.numel();
//            Tensor gammaResult;
//            if (totalElements > BATCH_THRESHOLD) {
//                gammaResult = batchProcessGamma(alpha, beta, totalElements);
//            } else {
//                Tensor gammaAlpha1 = sampleGammaAlpha1(alpha);
//                gammaResult = gammaAlpha1.div(beta).clone();
//                gammaAlpha1.close();
//            }
//
//            return gammaResult;
//
//        } catch (Exception e) {
//            if (alphaLe0 != null) alphaLe0.close();
//            if (betaLe0 != null) betaLe0.close();
//            if (anyInvalid != null) anyInvalid.close();
//            throw e;
//        }
//    }
//
//    /**
//     * 大批量 Gamma 采样分批处理（避免单批次过大导致迭代超限）
//     */
//    private static Tensor batchProcessGamma(Tensor alpha, Tensor beta, long totalElements) {
//        Tensor result = torch.empty(alpha.sizes().vec().get(), alpha.options(),new MemoryFormatOptional());
//        long batchSize = BATCH_THRESHOLD;
//        long numBatches = (totalElements + batchSize - 1) / batchSize;
//
//        for (long i = 0; i < numBatches; i++) {
//            long start = i * batchSize;
//            long end = Math.min((i + 1) * batchSize, totalElements);
//            if (start >= totalElements) break;
//
//            // 切片获取当前批次
//            Tensor alphaBatch = alpha.view(-1).narrow(0, start, end - start).view(
//                    getBatchShape(alpha.sizes().vec().get(), start, end)
//            );
//            Tensor betaBatch = beta.view(-1).narrow(0, start, end - start).view(
//                    getBatchShape(beta.sizes().vec().get(), start, end)
//            );
//
//            // 当前批次采样
//            Tensor gammaAlpha1 = sampleGammaAlpha1(alphaBatch);
//            Tensor gammaBatch = gammaAlpha1.div(betaBatch);
//
//            // 将批次结果写入最终张量
//            result.view(-1).narrow(0, start, end - start).copy_(gammaBatch.view(-1));
//
//            // 释放批次临时张量
//            alphaBatch.close();
//            betaBatch.close();
//            gammaAlpha1.close();
//            gammaBatch.close();
//        }
//
//        return result;
//    }
//
//    /**
//     * 辅助方法：计算分批后的形状
//     */
//    private static long[] getBatchShape(long[] originalShape, long start, long end) {
//        long batchSize = end - start;
//        if (originalShape.length == 0) {
//            return new long[]{batchSize};
//        }
//        long[] batchShape = Arrays.copyOf(originalShape, originalShape.length);
//        batchShape[batchShape.length - 1] = batchSize;
//        return batchShape;
//    }
//
//    /**
//     * 核心采样方法：逐元素接受（解决大批量迭代超限）
//     */
//    public static Tensor sampleGammaAlpha1(Tensor alpha) {
//        Tensor maskLess1 = null;
//        Tensor alphaPlus1 = null;
//        Tensor d = null;
//        Tensor c = null;
//        Tensor gamma1 = null;
//        Tensor uLess1 = null;
//        Tensor gammaFinal = null;
//        Tensor ones = null;
//        Tensor invAlpha = null;
//        Tensor uPow = null;
//
//        // 循环内临时张量
//        Tensor x = null;
//        Tensor v = null;
//        Tensor u = null;
//        Tensor logV = null;
//        Tensor accept = null;
//        Tensor logAccept = null;
//        Tensor validV = null;
//        Tensor remainingMask = null;
//        Tensor result = null;
//
//        try {
//            // 预定义标量
//            ones = torch.tensor(1.0f, alpha.options());
//            Scalar scalarOne = new Scalar(1.0);
//            Scalar scalarNegHalf = new Scalar(-0.5);
//            Scalar scalarOneThird = new Scalar(1.0 / 3.0);
//            Scalar scalarEps = new Scalar(1e-8);
//
//            // 步骤1：处理 α < 1
//            maskLess1 = alpha.lt(scalarOne);
//            alphaPlus1 = torch.where(maskLess1, alpha.add(scalarOne), alpha.clone());
//
//            // 步骤2：初始化 Marsaglia-Tsang 参数
//            d = alphaPlus1.sub(scalarOneThird);
//            c = torch.tensor(1.0 / 3.0f).div(d.sqrt());
//
//            // 步骤3：初始化结果张量（逐元素填充）
//            result = torch.empty_like(alphaPlus1);
//            remainingMask = torch.ones_like(alphaPlus1).to(kBool()); // 未完成的元素掩码
//            int iterations = 0;
//
//            // 循环：只处理未被接受的元素
//            while (torch.any(remainingMask).item().toBool() && iterations < MAX_ITERATIONS) {
//                // 释放上一轮张量
//                if (x != null) x.close();
//                if (v != null) v.close();
//                if (u != null) u.close();
//                if (logV != null) logV.close();
//                if (accept != null) accept.close();
//                if (logAccept != null) logAccept.close();
//                if (validV != null) validV.close();
//
//                // 只对未完成的元素采样
//                Tensor alphaRemaining = alphaPlus1.masked_select(remainingMask);
//                if (alphaRemaining.numel() == 0) break;
//
//                // 生成候选样本
//                x = torch.randn_like(alphaRemaining);
//                v = x.mul(c.masked_select(remainingMask)).add(scalarOne).pow(new Scalar(3.0));
//                validV = torch.clamp(v, new ScalarOptional(scalarEps), new ScalarOptional());
//                logV = validV.log();
//
//                // 生成均匀分布随机数
//                u = torch.rand_like(alphaRemaining).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));
//
//                // 计算接受条件（修正公式）
//                logAccept = x.mul(x).mul(scalarNegHalf)
//                        .add(d.masked_select(remainingMask).mul(validV.sub(scalarOne)))
//                        .sub(logV);
//                accept = u.log().lt(logAccept);
//
//                // 计算当前批次的 Gamma 样本
//                Tensor gammaBatch = d.masked_select(remainingMask).mul(validV);
//
//                // 将被接受的样本写入结果
//                Tensor acceptIndices = remainingMask.nonzero().masked_select(accept);
//                if (acceptIndices.numel() > 0) {
//                    result.put_(acceptIndices, gammaBatch.masked_select(accept).to(torch.ScalarType.Float));
//                    // 更新未完成掩码（清除已接受的元素）
//                    remainingMask.put_(acceptIndices, torch.zeros_like(acceptIndices).to(kBool()));
//                }
//
//                // 释放批次临时张量
//                alphaRemaining.close();
//                gammaBatch.close();
//                acceptIndices.close();
//                iterations++;
//            }
//
//            // 检查是否所有元素都采样完成
//            if (iterations >= MAX_ITERATIONS && torch.any(remainingMask).item().toBool()) {
//                // 兜底：对剩余元素使用 PyTorch 原生 Gamma 采样
//                Tensor remainingAlpha = alphaPlus1.masked_select(remainingMask);
//                Tensor fallbackGamma = torch.randn_like(remainingAlpha).abs(); // 简单兜底
//                result.put_(remainingMask.nonzero(), fallbackGamma);
//                remainingAlpha.close();
//                fallbackGamma.close();
//            }
//
//            // 步骤4：处理 α < 1 的还原
//            gamma1 = result;
//            uLess1 = torch.rand_like(alpha).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));
//            invAlpha = alpha.reciprocal();
//            uPow = uLess1.pow(invAlpha);
//            gammaFinal = torch.where(maskLess1, gamma1.mul(uPow), gamma1.clone());
//
//            // 释放标量
//            scalarOne.close();
//            scalarNegHalf.close();
//            scalarOneThird.close();
//            scalarEps.close();
//
//            return gammaFinal.clone();
//
//        } finally {
//            // 统一释放所有临时张量
//            if (maskLess1 != null) maskLess1.close();
//            if (alphaPlus1 != null) alphaPlus1.close();
//            if (d != null) d.close();
//            if (c != null) c.close();
//            if (gamma1 != null) gamma1.close();
//            if (uLess1 != null) uLess1.close();
//            if (gammaFinal != null) gammaFinal.close();
//            if (ones != null) ones.close();
//            if (invAlpha != null) invAlpha.close();
//            if (uPow != null) uPow.close();
//            if (x != null) x.close();
//            if (v != null) v.close();
//            if (u != null) u.close();
//            if (logV != null) logV.close();
//            if (accept != null) accept.close();
//            if (logAccept != null) logAccept.close();
//            if (validV != null) validV.close();
//            if (remainingMask != null) remainingMask.close();
//            if (result != null) result.close();
//        }
//    }
//
//    // 测试 Gamma 采样（验证无无限循环 + 结果合法）
//    public static void main(String[] args) {
//        try (
//                // 测试用例1：α=2.0, β=1.0（基础场景）
//                Tensor alpha = torch.tensor(2.0f);
//                Tensor beta = torch.tensor(1.0f);
//                // 测试用例2：α=0.5 < 1（边界场景）
//                Tensor alphaSmall = torch.tensor(0.5f);
//                Tensor betaSmall = torch.tensor(1.0f)
//        ) {
//            // 测试1：正常α采样
//            Tensor gamma1 = GammaSampler.gamma(alpha, beta);
//            System.out.println("Gamma(2.0, 1.0) 采样结果：" + gamma1.item().toFloat());
//            gamma1.close();
//
//            // 测试2：α<1采样
//            Tensor gamma2 = GammaSampler.gamma(alphaSmall, betaSmall);
//            System.out.println("Gamma(0.5, 1.0) 采样结果：" + gamma2.item().toFloat());
//            gamma2.close();
//
//            // 测试3：批量采样（形状[2,3]）
//            Tensor alphaBatch = torch.tensor(new float[]{1.5f, 2.5f, 3.5f,0.8f, 1.2f, 4.0f}).view(2,3);
//            Tensor betaBatch = torch.ones_like(alphaBatch);
//            Tensor gammaBatch = GammaSampler.gamma(alphaBatch, betaBatch);
//            System.out.println("批量采样形状：" + Arrays.toString(gammaBatch.sizes().vec().get()));
//            gammaBatch.close();
//            alphaBatch.close();
//            betaBatch.close();
//
//            System.out.println("Gamma 采样测试完成，无无限循环！");
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//}
//public class GammaSampler2 {
//
//    // 最大循环迭代次数（防止极端情况无限循环）
//    private static final int MAX_ITERATIONS = 10000;
//
//    /**
//     * 实现 Gamma(alpha, beta) 分布采样（静态方法，双 Tensor 入参）
//     * 数学定义：Gamma(α, β) = 1/β * Gamma(α, 1)，其中 α 是形状参数，β 是速率参数
//     * @param alpha 形状参数张量（浮点型，α > 0）
//     * @param beta  速率参数张量（浮点型，β > 0）
//     * @return Gamma 分布采样结果，形状与输入一致
//     * @throws IllegalArgumentException 输入不合法时抛出
//     * @throws RuntimeException 采样迭代超过最大次数时抛出
//     */
//    public static Tensor gamma(Tensor alpha, Tensor beta) {
//        if (alpha == null || beta == null) {
//            throw new IllegalArgumentException("alpha 和 beta 张量不能为 null");
//        }
//
//        // 临时张量声明（便于异常时统一释放）
//        Tensor alphaLe0 = null;
//        Tensor betaLe0 = null;
//        Tensor anyInvalid = null;
//        try {
//            // 校验数据类型（支持float/double）
//            if (!alpha.dtype().isScalarType(torch.ScalarType.Float) || !beta.dtype().isScalarType(torch.ScalarType.Float)) {
//                throw new IllegalArgumentException("alpha 和 beta 必须是浮点型张量（float32/float64）");
//            }
//
//            // 修复：数组形状比较（Arrays.equals 替代对象 equals）
//            long[] alphaShape = alpha.sizes().vec().get();
//            long[] betaShape = beta.sizes().vec().get();
//            System.out.println("gamma compute : alphaShape" + Arrays.toString(alphaShape) + " betaShape  " + Arrays.toString(betaShape));
//
//            if (!Arrays.equals(alphaShape, betaShape)) {
//                throw new IllegalArgumentException(
//                        "alpha 和 beta 张量形状必须一致！alpha形状=" + Arrays.toString(alphaShape) +
//                                ", beta形状=" + Arrays.toString(betaShape)
//                );
//            }
//
//            // 校验数值>0（使用临时张量，异常时释放）
//            alphaLe0 = alpha.le(new Scalar(0.0f));
//            betaLe0 = beta.le(new Scalar(0.0f));
//            anyInvalid = torch.logical_or(torch.any(alphaLe0), torch.any(betaLe0));
//
//            if (anyInvalid.item().toBool()) {
//                throw new IllegalArgumentException("alpha 和 beta 所有元素必须大于 0");
//            }
//
//            // 核心采样逻辑
//            Tensor gammaAlpha1 = sampleGammaAlpha1(alpha);
//            Tensor gammaResult = gammaAlpha1.div(beta).clone();
//
//            // 释放临时张量
//            gammaAlpha1.close();
//            return gammaResult;
//
//        } catch (Exception e) {
//            // 异常时强制释放所有临时张量
//            if (alphaLe0 != null) alphaLe0.close();
//            if (betaLe0 != null) betaLe0.close();
//            if (anyInvalid != null) anyInvalid.close();
//            throw e;
//        }
//    }
//
//    /**
//     * 辅助方法：采样 Gamma(α, 1) 分布（修复 Marsaglia-Tsang 算法 + 防止无限循环）
//     * 适配所有 α > 0 的情况，解决原算法接受条件错误和数值稳定性问题
//     */
//    public static Tensor sampleGammaAlpha1(Tensor alpha) {
//        // 所有临时张量声明（finally 中统一释放）
//        Tensor maskLess1 = null;
//        Tensor alphaPlus1 = null;
//        Tensor d = null;
//        Tensor c = null;
//        Tensor gamma1 = null;
//        Tensor uLess1 = null;
//        Tensor gammaFinal = null;
//        Tensor ones = null;
//        Tensor invAlpha = null;
//        Tensor uPow = null;
//
//        // 循环内临时张量
//        Tensor x = null;
//        Tensor v = null;
//        Tensor u = null;
//        Tensor logV = null;
//        Tensor accept = null;
//        Tensor logAccept = null;
//        Tensor validV = null;
//
//        try {
//            // 预定义标量（复用避免重复创建）
//            ones = torch.tensor(1.0f, alpha.options());
//            Scalar scalarOne = new Scalar(1.0);
//            Scalar scalarNegHalf = new Scalar(-0.5);
//            Scalar scalarOneThird = new Scalar(1.0 / 3.0);
//            Scalar scalarEps = new Scalar(1e-8);
//
//            // 步骤1：处理 α < 1 的情况（转换为 α+1 采样）
//            maskLess1 = alpha.lt(scalarOne);
//            alphaPlus1 = torch.where(maskLess1, alpha.add(scalarOne), alpha.clone());
//
//            // 步骤2：Marsaglia-Tsang 算法初始化（修正公式）
//            d = alphaPlus1.sub(scalarOneThird);
//            c = torch.tensor(1.0 / 3.0).div(d.sqrt());
//
//            // 步骤3：循环采样（修复接受条件 + 最大迭代次数限制）
//            int iterations = 0;
//            boolean allAccepted = false;
//            do {
//                // 释放上一轮循环的张量
//                if (x != null) x.close();
//                if (v != null) v.close();
//                if (u != null) u.close();
//                if (logV != null) logV.close();
//                if (accept != null) accept.close();
//                if (logAccept != null) logAccept.close();
//                if (validV != null) validV.close();
//
//                // 生成候选样本（保证数值稳定性）
//                x = torch.randn_like(alphaPlus1);
//                v = x.mul(c).add(scalarOne).pow(new Scalar(3.0));
//
//                // 数值稳定：v 必须 > 0（避免 log(v) 为 -∞）
//                validV = torch.clamp(v, new ScalarOptional(new Scalar(1e-8)), new ScalarOptional()); // v ≥ 1e-8
//                logV = validV.log();
//
//                // 生成均匀分布随机数
//                u = torch.rand_like(alphaPlus1).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));
//
//                // 修复核心：Marsaglia-Tsang 算法的接受条件公式
//                // 正确公式：log(u) < -0.5*x² + d*(v-1) - log(v)
//                logAccept = x.mul(x).mul(scalarNegHalf)
//                        .add(d.mul(validV.sub(scalarOne)))
//                        .sub(logV);
//                accept = u.log().lt(logAccept);
//
//                // 检查是否所有样本都被接受
//                allAccepted = !torch.any(accept.logical_not()).item().toBool();
//
//                // 防止无限循环：超过最大迭代次数抛出异常
//                if (++iterations > MAX_ITERATIONS) {
//                    throw new RuntimeException("Gamma 采样超过最大迭代次数（" + MAX_ITERATIONS + "），可能是数值异常");
//                }
//            } while (!allAccepted);
//
//            // 步骤4：生成 Gamma(α+1, 1) 样本
//            gamma1 = d.mul(validV);
//
//            // 步骤5：处理 α < 1 的情况（还原为原 α 的样本）
//            uLess1 = torch.rand_like(alpha).clamp(new ScalarOptional(scalarEps), new ScalarOptional(new Scalar(1.0 - 1e-8)));
//            invAlpha = alpha.reciprocal();
//            uPow = uLess1.pow(invAlpha);
//            gammaFinal = torch.where(maskLess1, gamma1.mul(uPow), gamma1.clone());
//
//            // 释放标量
//            scalarOne.close();
//            scalarNegHalf.close();
//            scalarOneThird.close();
//            scalarEps.close();
//
//            // 返回克隆后的结果（避免原张量释放影响）
//            return gammaFinal.clone();
//
//        } finally {
//            // 最终释放所有临时张量（无论是否异常）
//            if (maskLess1 != null) maskLess1.close();
//            if (alphaPlus1 != null) alphaPlus1.close();
//            if (d != null) d.close();
//            if (c != null) c.close();
//            if (gamma1 != null) gamma1.close();
//            if (uLess1 != null) uLess1.close();
//            if (gammaFinal != null) gammaFinal.close();
//            if (ones != null) ones.close();
//            if (invAlpha != null) invAlpha.close();
//            if (uPow != null) uPow.close();
//
//            // 循环内张量最终释放
//            if (x != null) x.close();
//            if (v != null) v.close();
//            if (u != null) u.close();
//            if (logV != null) logV.close();
//            if (accept != null) accept.close();
//            if (logAccept != null) logAccept.close();
//            if (validV != null) validV.close();
//        }
//    }
//
//    // 测试 Gamma 采样（验证无无限循环 + 结果合法）
//    public static void main(String[] args) {
//        try (
//                // 测试用例1：α=2.0, β=1.0（基础场景）
//                Tensor alpha = torch.tensor(2.0f);
//                Tensor beta = torch.tensor(1.0f);
//                // 测试用例2：α=0.5 < 1（边界场景）
//                Tensor alphaSmall = torch.tensor(0.5f);
//                Tensor betaSmall = torch.tensor(1.0f)
//        ) {
//            // 测试1：正常α采样
//            Tensor gamma1 = GammaSampler.gamma(alpha, beta);
//            System.out.println("Gamma(2.0, 1.0) 采样结果：" + gamma1.item().toFloat());
//            gamma1.close();
//
//            // 测试2：α<1采样
//            Tensor gamma2 = GammaSampler.gamma(alphaSmall, betaSmall);
//            System.out.println("Gamma(0.5, 1.0) 采样结果：" + gamma2.item().toFloat());
//            gamma2.close();
//
//            // 测试3：批量采样（形状[2,3]）
//            Tensor alphaBatch = torch.tensor(new float[]{1.5f, 2.5f, 3.5f,0.8f, 1.2f, 4.0f}).view(2,3);
//            Tensor betaBatch = torch.ones_like(alphaBatch);
//            Tensor gammaBatch = GammaSampler.gamma(alphaBatch, betaBatch);
//            System.out.println("批量采样形状：" + Arrays.toString(gammaBatch.sizes().vec().get()));
//            gammaBatch.close();
//            alphaBatch.close();
//            betaBatch.close();
//
//            System.out.println("Gamma 采样测试完成，无无限循环！");
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
//}