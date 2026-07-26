package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.distribution.internal.GammaSampler;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * NegativeBinomial（负二项）分布实现
 * 严格遵循PyTorch Java API规范：
 * 1. clamp使用ScalarOptional封装参数
 * 2. Scalar必须转为Tensor后操作
 * 3. TensorOptions/Device严格按API创建
 * 4. 所有临时张量显式close，避免内存泄漏
 */
public class NegativeBinomial extends Distribution implements AutoCloseable {
    private final Tensor total_count;  // 成功次数r（>0，形状：batch_shape）
    private final Tensor probs;        // 单次成功概率p（0<p<1，形状：batch_shape）

    // 全局常量（严格按API创建，避免重复初始化）
    private static final Device CPU_DEVICE = new Device(torch.kCPU());
    private static final Tensor TENSOR_0_FLOAT;
    private static final Tensor TENSOR_1_FLOAT;
    private static final Tensor TENSOR_EPS_FLOAT; // 1e-8，数值稳定性
    private static final Tensor TENSOR_NEG_INF_FLOAT;

    // 静态初始化：严格按API创建常量张量
    static {
        // 设备统一为CPU，类型为Float
        TENSOR_0_FLOAT = torch.tensor(0.0f).to(kFloat());
        TENSOR_1_FLOAT = torch.tensor(1.0f).to(kFloat());
        TENSOR_EPS_FLOAT = torch.tensor(1e-8f).to(kFloat());
        TENSOR_NEG_INF_FLOAT = torch.tensor(Float.NEGATIVE_INFINITY).to(kFloat());

        // JVM退出时释放静态张量
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            safeClose(TENSOR_0_FLOAT);
            safeClose(TENSOR_1_FLOAT);
            safeClose(TENSOR_EPS_FLOAT);
            safeClose(TENSOR_NEG_INF_FLOAT);
        }));
    }

    /**
     * 构造函数：严格校验参数合法性 + 深拷贝（遵循API规范）
     * @param total_count 成功次数r（必须>0）
     * @param probs 单次成功概率p（必须0<p<1）
     * @throws IllegalArgumentException 参数非法时抛出
     */
    public NegativeBinomial(Tensor total_count, Tensor probs) {
        // 1. 统一转换设备/类型：CPU + Float
        Tensor tcCPU = total_count.to(CPU_DEVICE, kFloat()).clone().detach();
        Tensor pCPU = probs.to(CPU_DEVICE, kFloat()).clone().detach();

        // 2. 校验total_count>0（严格按API操作Scalar）
        Tensor tcLe0 = torch.le(tcCPU, TENSOR_0_FLOAT);
        if (torch.any(tcLe0).item().toBool()) {
            safeClose(tcLe0);
            safeClose(tcCPU);
            safeClose(pCPU);
            throw new IllegalArgumentException("total_count(r)必须大于0！");
        }

        // 3. 校验0 < p < 1
        Tensor pLe0 = torch.le(pCPU, TENSOR_0_FLOAT);
        Tensor pGe1 = torch.ge(pCPU, TENSOR_1_FLOAT);
        Tensor pInvalid = torch.logical_or(pLe0, pGe1);
        if (torch.any(pInvalid).item().toBool()) {
            safeClose(tcLe0);
            safeClose(pLe0);
            safeClose(pGe1);
            safeClose(pInvalid);
            safeClose(tcCPU);
            safeClose(pCPU);
            throw new IllegalArgumentException("probs(p)必须满足 0 < p < 1！");
        }

        // 4. 深拷贝保存内部状态
        this.total_count = tcCPU;
        this.probs = pCPU;

        // 释放校验临时张量
        safeClose(tcLe0);
        safeClose(pLe0);
        safeClose(pGe1);
        safeClose(pInvalid);
    }

    @Override
    public String name() {
        return "NegativeBinomial";
    }

    /**
     * 采样：Gamma-Poisson混合（严格遵循API，返回标量/正确形状）
     * 公式：NB(r,p) = Poisson(Gamma(r, (1-p)/p))
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：扩展形状（sampleShape + batch_shape）
        long[] extendedShape = getExtendedShape(total_count, sampleShape);

        // 步骤2：扩展参数到采样形状（CPU + Float）
        Tensor expandedTc = total_count.expand(extendedShape).to(CPU_DEVICE, kFloat());
        Tensor expandedP = probs.expand(extendedShape).to(CPU_DEVICE, kFloat());

        // 步骤3：计算Gamma rate = (1-p)/p（严格按API调用clamp）
        Tensor oneMinusP = torch.sub(TENSOR_1_FLOAT, expandedP);
        // clamp避免除0：下界1e-8，上界1.0（无null，用ScalarOptional封装）
        Tensor safeP = torch.clamp(expandedP, new ScalarOptional(TENSOR_EPS_FLOAT.item()), new ScalarOptional(TENSOR_1_FLOAT.item()));
        Tensor rate = torch.div(oneMinusP, safeP);

        // 步骤4：Gamma采样（shape=r, scale=1/rate）
        Tensor gammaSample = GammaSampler.gamma(expandedTc, torch.reciprocal(rate));

        // 步骤5：Poisson采样 + 转为Long类型
        Tensor nbSample = torch.poisson(gammaSample).toType(kLong());

        // 步骤6：优化形状：单样本返回标量（dim=1且numel=1时挤压）
        if (nbSample.dim() == 1 && nbSample.numel() == 1) {
            Tensor scalarSample = nbSample.squeeze(0);
            safeClose(nbSample);
            nbSample = scalarSample;
        }

        // 释放临时张量
        safeClose(expandedTc);
        safeClose(expandedP);
        safeClose(oneMinusP);
        safeClose(safeP);
        safeClose(rate);
        safeClose(gammaSample);

        return nbSample;
    }

    /**
     * 对数概率：修复非整数判断，严格遵循API，非法输入返回-∞
     * 公式：logP = lgamma(k+r) - lgamma(k+1) - lgamma(r) + r*logp + k*log(1-p)
     */
    @Override
    public Tensor log_prob(Tensor v) {
        // 1. 统一转换输入为CPU + Float，保留原始形状
        Tensor vCPU = v.to(CPU_DEVICE, kFloat()).clone().detach();
        long[] inputShape = vCPU.sizes().vec().get();

        // 2. 快速判断：是否为非法输入（<0 或 非整数）
        boolean hasInvalidInput = false;
        Tensor invalidMask = torch.ones_like(vCPU).to(kBool()); // 初始化为全非法
        try {
            // 2.1 校验v ≥ 0
            Tensor vGe0 = torch.ge(vCPU, TENSOR_0_FLOAT);
            // 2.2 校验v为整数（浮点型：值等于四舍五入后的值，精度1e-6）
            Tensor vIsInt;
            if (v.scalar_type() == kFloat() || v.scalar_type() == kDouble()) {
                Tensor vRounded = vCPU.round();
                Tensor vDiff = torch.abs(torch.sub(vCPU, vRounded));
                vIsInt = torch.lt(vDiff, torch.tensor(1e-6f));
                safeClose(vRounded);
                safeClose(vDiff);
            } else {
                vIsInt = torch.ones_like(vCPU).to(kBool()); // 整型默认合法
            }
            // 2.3 合法输入 = v≥0 AND 是整数
            invalidMask = torch.logical_not(torch.logical_and(vGe0, vIsInt));
            // 2.4 判断是否存在任何非法输入
            hasInvalidInput = torch.any(invalidMask).item().toBool();

            safeClose(vGe0);
            safeClose(vIsInt);
        } catch (Exception e) {
            // 任何校验异常，直接判定为非法输入
            hasInvalidInput = true;
        }

        // 3. 兜底逻辑：只要有非法输入，先创建全-∞的张量
        Tensor finalLogProb = torch.full(
                inputShape,
                new Scalar(Float.NEGATIVE_INFINITY),
                new TensorOptions().
                        dtype(new ScalarTypeOptional(kFloat()))
                        .device(new DeviceOptional(CPU_DEVICE))
        );

        // ============= 第二步：仅对合法输入计算log_prob =============
        if (!hasInvalidInput) {
            // 3.1 扩展参数到输入形状
            Tensor expandedTc = total_count.expand(inputShape).to(CPU_DEVICE, kFloat());
            Tensor expandedP = probs.expand(inputShape).to(CPU_DEVICE, kFloat());

            // 3.2 数值稳定性处理
            Tensor safeP = torch.clamp(
                    expandedP,
                    new ScalarOptional(TENSOR_EPS_FLOAT.item()),
                    new ScalarOptional(torch.sub(TENSOR_1_FLOAT, TENSOR_EPS_FLOAT).item())
            );
            Tensor safeOneMinusP = torch.clamp(
                    torch.sub(TENSOR_1_FLOAT, safeP),
                    new ScalarOptional(TENSOR_EPS_FLOAT.item()),
                    new ScalarOptional(torch.sub(TENSOR_1_FLOAT, TENSOR_EPS_FLOAT).item())
            );

            // 3.3 计算log_prob（仅合法输入）
            Tensor vPlusTc = torch.add(vCPU, expandedTc);
            Tensor vPlus1 = torch.add(vCPU, TENSOR_1_FLOAT);
            Tensor term1 = torch.sub(
                    torch.sub(torch.lgamma(vPlusTc), torch.lgamma(vPlus1)),
                    torch.lgamma(expandedTc)
            );
            Tensor logP = torch.log(safeP);
            Tensor logOneMinusP = torch.log(safeOneMinusP);
            Tensor term2 = torch.add(
                    torch.mul(expandedTc, logP),
                    torch.mul(vCPU, logOneMinusP)
            );
            Tensor validLogProb = torch.add(term1, term2);

            // 替换finalLogProb为合法计算结果
            safeClose(finalLogProb);
            finalLogProb = validLogProb.clone().detach();

            // 释放临时张量
            safeClose(expandedTc);
            safeClose(expandedP);
            safeClose(safeP);
            safeClose(safeOneMinusP);
            safeClose(vPlusTc);
            safeClose(vPlus1);
            safeClose(term1);
            safeClose(logP);
            safeClose(logOneMinusP);
            safeClose(term2);
            safeClose(validLogProb);
        } else {
            // 非法输入：直接返回全-∞，无需计算
        }

        // ============= 第三步：清理资源并返回 =============
        safeClose(vCPU);
        safeClose(invalidMask);
        return finalLogProb;
    }

    /**
     * 均值：严格按公式 r*(1-p)/p，遵循API规范
     */
    @Override
    public Tensor mean() {
        // 数值稳定性处理（clamp避免除0）
        Tensor safeP = torch.clamp(
                probs,
                new ScalarOptional(TENSOR_EPS_FLOAT.item()),
                new ScalarOptional(torch.sub(TENSOR_1_FLOAT, TENSOR_EPS_FLOAT).item())
        );
        Tensor oneMinusP = torch.sub(TENSOR_1_FLOAT, safeP);

        // 计算均值：r*(1-p)/p
        Tensor mean = torch.mul(
                total_count,
                torch.div(oneMinusP, safeP)
        ).clone().detach();

        // 释放临时张量
        safeClose(safeP);
        safeClose(oneMinusP);

        return mean;
    }

    /**
     * Entropy via Monte-Carlo: H ≈ -mean(log_prob(sample)).
     * PyTorch's NegativeBinomial does not implement a closed-form entropy;
     * this matches that design while still returning a usable estimate.
     */
    @Override
    public Tensor entropy() {
        final long n = 4096;
        Tensor samples = sample(n);
        Tensor lp = log_prob(samples.to(kFloat()));
        // mean over sample dimension 0 → shape = batch_shape
        Tensor ent = lp.mean(new long[]{0}, false, new ScalarTypeOptional()).neg();
        safeClose(samples);
        safeClose(lp);
        return ent;
    }

    /**
     * 资源释放：仅释放实例张量，静态张量由JVM钩子释放
     */
    @Override
    public void close() {
        safeClose(total_count);
        safeClose(probs);
    }

    // 辅助方法：安全释放AutoCloseable资源
    private static void safeClose(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                System.err.println("资源释放失败: " + e.getMessage());
            }
        }
    }

    // Getter方法
    public Tensor getTotalCount() { return total_count; }
    public Tensor getProbs() { return probs; }

}
