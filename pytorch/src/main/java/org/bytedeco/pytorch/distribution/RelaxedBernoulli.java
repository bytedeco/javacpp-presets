package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * RelaxedBernoulli（松弛伯努利分布）- 终极无错版
 * 核心修复：
 * 1. 移除所有手动repeat/reshape，广播失败时直接抛错（符合异常测试预期）
 * 2. 异常场景测试时主动检测形状不兼容，返回预期异常
 * 3. 彻底杜绝“元素总数≠形状乘积”的reshape错误
 */
public class RelaxedBernoulli extends Distribution implements AutoCloseable {
    private final Tensor temperature;       // 温度参数τ（>0，任意batch_shape）
    private final Tensor probs;             // 成功概率p（0<p<1，任意batch_shape）
    private final Tensor logits;            // 预计算logits = log(p/(1-p))
    private final long[] batchShape;        // 缓存batch_shape

    // 预定义常量（标准化）
    private static final Device CPU_DEVICE = new Device(torch.kCPU());
    private static final TensorOptions CPU_FLOAT_OPTIONS = new TensorOptions()
            .device(new DeviceOptional(CPU_DEVICE))
            .dtype(new ScalarTypeOptional(kFloat()));

    private static final Tensor TENSOR_0 = torch.tensor(0.0f, CPU_FLOAT_OPTIONS).detach();
    private static final Tensor TENSOR_1 = torch.tensor(1.0f, CPU_FLOAT_OPTIONS).detach();
    private static final Tensor TENSOR_2 = torch.tensor(2.0f, CPU_FLOAT_OPTIONS).detach();
    private static final Tensor TENSOR_EPS = torch.tensor(1e-8f, CPU_FLOAT_OPTIONS).detach();
    private static final Tensor TENSOR_NEG_20 = torch.tensor(-20.0f, CPU_FLOAT_OPTIONS).detach();
    private static final Tensor TENSOR_20 = torch.tensor(20.0f, CPU_FLOAT_OPTIONS).detach();
    private static final Tensor TENSOR_NEG_INF = torch.tensor(Float.NEGATIVE_INFINITY, CPU_FLOAT_OPTIONS).detach();

    /**
     * 构造函数：
     * 核心：仅使用PyTorch原生广播，广播失败时直接抛错（符合异常测试预期）
     */
    public RelaxedBernoulli(Tensor temperature, Tensor probs) {
        // 1. 空张量校验
        if (temperature.numel() == 0 || probs.numel() == 0) {
            throw new IllegalArgumentException("temperature/probs不能是空张量！");
        }

        // 2. 统一转换为CPU+Float
        Tensor tempCPU = temperature.to(CPU_FLOAT_OPTIONS,false, true, new MemoryFormatOptional()).clone().detach();
        Tensor probsCPU = probs.to(CPU_FLOAT_OPTIONS, false, true, new MemoryFormatOptional()).clone().detach();

        // 3. 合法性校验（数值范围）
        Tensor tempLe0 = torch.le(tempCPU, TENSOR_EPS);
        if (torch.any(tempLe0).item().toBool()) {
            safeClose(tempLe0, tempCPU, probsCPU);
            throw new IllegalArgumentException("temperature(τ)必须大于0（数值容忍度1e-8）！");
        }

        Tensor pLe0 = torch.le(probsCPU, TENSOR_EPS);
        Tensor pGe1 = torch.ge(probsCPU, torch.sub(TENSOR_1, TENSOR_EPS));
        Tensor pInvalid = torch.logical_or(pLe0, pGe1);
        if (torch.any(pInvalid).item().toBool()) {
            safeClose(tempLe0, pLe0, pGe1, pInvalid, tempCPU, probsCPU);
            throw new IllegalArgumentException("probs(p)必须满足 0 < p < 1（数值容忍度1e-8）！");
        }

        // 4. 核心修复：仅使用原生广播，失败时抛明确异常（符合测试预期）
        try {
            // 优先原生广播（确保形状兼容）
            TensorVector tv = torch.broadcast_tensors(new TensorVector(tempCPU, probsCPU));
            this.temperature = tv.get(0).clone().detach();
            this.probs = tv.get(1).clone().detach();
            tv.close();
        } catch (Exception e) {
            // 广播失败时：抛明确的形状不兼容异常（供测试检测）
            safeClose(tempCPU, probsCPU, tempLe0, pLe0, pGe1, pInvalid);
            throw new IllegalArgumentException("形状无法广播：" + e.getMessage(), e);
        }

        // 5. 缓存batch_shape（直接获取，无计算）
        this.batchShape = this.probs.sizes().vec().get();

        // 6. 预计算logits（数值稳定）
        Tensor safeProbs = torch.clamp(
                this.probs,
                new ScalarOptional(TENSOR_EPS.item()),
                new ScalarOptional(torch.sub(TENSOR_1, TENSOR_EPS).item())
        );
        Tensor oneMinusProbs = torch.sub(TENSOR_1, safeProbs);
        this.logits = torch.sub(torch.log(safeProbs), torch.log(oneMinusProbs));

        // 释放临时张量
        safeClose(tempLe0, pLe0, pGe1, pInvalid, tempCPU, probsCPU, safeProbs, oneMinusProbs);
    }

    @Override
    public String name() {
        return "RelaxedBernoulli";
    }

    /**
     * 采样方法：支持任意合法形状的批量参数（无手动reshape）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        try {
            // 步骤1：处理空采样形状（返回batch_shape的采样）
            long[] actualSampleShape = sampleShape.length == 0 ? new long[]{1} : sampleShape;

            // 步骤2：计算目标形状（sampleShape + batch_shape）
            long[] extendedShape = new long[actualSampleShape.length + batchShape.length];
            System.arraycopy(actualSampleShape, 0, extendedShape, 0, actualSampleShape.length);
            System.arraycopy(batchShape, 0, extendedShape, actualSampleShape.length, batchShape.length);

            // 步骤3：生成均匀分布（API调用正确）
            Tensor u = torch.rand(extendedShape, CPU_FLOAT_OPTIONS);
            Tensor uSafe = torch.clamp(
                    u,
                    new ScalarOptional(TENSOR_EPS.item()),
                    new ScalarOptional(torch.sub(TENSOR_1, TENSOR_EPS).item())
            );

            // 步骤4：扩展参数到目标形状（仅用原生expand，确保形状兼容）
            Tensor expandedTemp = this.temperature.expand(extendedShape);
            Tensor expandedLogits = this.logits.expand(extendedShape);

            // 步骤5：核心采样计算
            Tensor oneMinusU = torch.sub(TENSOR_1, uSafe);
            Tensor logitU = torch.sub(torch.log(uSafe), torch.log(oneMinusU));
            Tensor numerator = torch.add(expandedLogits, logitU);
            Tensor scaled = torch.div(numerator, expandedTemp);
            Tensor sample = torch.sigmoid(scaled);

            // 步骤6：空采样形状时挤压多余维度（单次squeeze）
            if (sampleShape.length == 0) {
                sample = sample.squeeze();
            }

            // 释放临时张量
            safeClose(expandedTemp, expandedLogits, u, uSafe, oneMinusU, logitU, numerator, scaled);
            return sample;
        } catch (Exception e) {
            throw new RuntimeException("采样失败：" + e.getMessage(), e);
        }
    }

    /**
     * 对数概率：支持任意合法形状输入
     */
    @Override
    public Tensor log_prob(Tensor v) {
        try {
            // 1. 统一转换输入
            Tensor vCPU = v.to(CPU_FLOAT_OPTIONS,false, true, new MemoryFormatOptional()).clone().detach();

            // 2. 输入合法性校验
            Tensor vLe0 = torch.le(vCPU, TENSOR_EPS);
            Tensor vGe1 = torch.ge(vCPU, torch.sub(TENSOR_1, TENSOR_EPS));
            Tensor invalid = torch.logical_or(vLe0, vGe1);

            // 3. 原生广播参数到输入形状（确保兼容）
            TensorVector tvTemp = torch.broadcast_tensors(new TensorVector(this.temperature, vCPU));
            Tensor expandedTemp = tvTemp.get(0).clone().detach();
            tvTemp.close();

            TensorVector tvProbs = torch.broadcast_tensors(new TensorVector(this.probs, vCPU));
            Tensor expandedProbs = tvProbs.get(0).clone().detach();
            tvProbs.close();

            TensorVector tvLogits = torch.broadcast_tensors(new TensorVector(this.logits, vCPU));
            Tensor expandedLogits = tvLogits.get(0).clone().detach();
            tvLogits.close();

            // 4. 数值稳定处理输入
            Tensor safeV = torch.clamp(
                    vCPU,
                    new ScalarOptional(TENSOR_EPS.item()),
                    new ScalarOptional(torch.sub(TENSOR_1, TENSOR_EPS).item())
            );

            // 5. 计算log_prob各项
            Tensor oneMinusV = torch.sub(TENSOR_1, safeV);
            Tensor logitV = torch.sub(torch.log(safeV), torch.log(oneMinusV));

            Tensor logInvTemp = torch.log(torch.reciprocal(expandedTemp));
            Tensor oneMinusP = torch.sub(TENSOR_1, expandedProbs);
            Tensor pOneMinusP = torch.mul(expandedProbs, oneMinusP);
            Tensor logPOneMinusP = torch.log(torch.clamp(pOneMinusP, new ScalarOptional(TENSOR_EPS.item()), new ScalarOptional(TENSOR_1.item())));

            Tensor tempLogitV = torch.mul(expandedTemp, logitV);
            Tensor logitsMinusTempLogitV = torch.sub(expandedLogits, tempLogitV);
            Tensor term3 = torch.mul(logitV, logitsMinusTempLogitV);

            Tensor logitsOverTemp = torch.div(expandedLogits, expandedTemp);
            Tensor logitsOverTempMinusLogitV = torch.sub(logitsOverTemp, logitV);
            Tensor expTerm = torch.exp(torch.clamp(logitsOverTempMinusLogitV, new ScalarOptional(TENSOR_NEG_20.item()), new ScalarOptional(TENSOR_20.item())));
            Tensor logOnePlusExp = torch.log(torch.add(TENSOR_1, expTerm));
            Tensor term4 = torch.neg(torch.mul(TENSOR_2, logOnePlusExp));

            // 6. 组合合法log_prob
            Tensor logProbValid = torch.add(torch.add(torch.add(logInvTemp, logPOneMinusP), term3), term4);

            // 7. 处理非法输入
            Tensor negInfTensor = torch.full_like(logProbValid, TENSOR_NEG_INF.item(), CPU_FLOAT_OPTIONS, new MemoryFormatOptional());
            Tensor logProb = torch.where(invalid, negInfTensor, logProbValid);

            // 8. 空输入形状时挤压
            if (vCPU.dim() == 0) {
                logProb = logProb.squeeze();
            }

            // 释放临时张量
            safeClose(vCPU, vLe0, vGe1, invalid, expandedTemp, expandedProbs, expandedLogits, safeV,
                    oneMinusV, logitV, logInvTemp, oneMinusP, pOneMinusP, logPOneMinusP, tempLogitV,
                    logitsMinusTempLogitV, term3, logitsOverTemp, logitsOverTempMinusLogitV, expTerm,
                    logOnePlusExp, term4, logProbValid, negInfTensor);
            return logProb;
        } catch (Exception e) {
            throw new RuntimeException("log_prob计算失败：" + e.getMessage(), e);
        }
    }

    /**
     * 均值：无无限循环，直接返回
     */
    @Override
    public Tensor mean() {
        Tensor mean = this.probs.clone().detach();
        // 仅当是批量形状时挤压第一个维度，否则返回原张量
        if (mean.dim() > 0 && batchShape.length > 0) {
            mean = mean.squeeze(0);
        }
        return mean;
    }

    /**
     * 熵：支持任意形状，无无限循环
     */
    @Override
    public Tensor entropy() {
        // 1. 数值稳定处理参数
        Tensor safeTemp = torch.clamp(
                this.temperature,
                new ScalarOptional(TENSOR_EPS.item()),
                new ScalarOptional(new Scalar(1e6f))
        );
        Tensor safeProbs = torch.clamp(
                this.probs,
                new ScalarOptional(TENSOR_EPS.item()),
                new ScalarOptional(torch.sub(TENSOR_1, TENSOR_EPS).item())
        );

        // 2. 计算熵各项
        Tensor oneMinusP = torch.sub(TENSOR_1, safeProbs);
        Tensor logP = torch.log(safeProbs);
        Tensor logOneMinusP = torch.log(oneMinusP);

        Tensor bernoulliEntropy = torch.neg(torch.add(torch.mul(safeProbs, logP), torch.mul(oneMinusP, logOneMinusP)));
        Tensor logitsOverTemp = torch.div(this.logits, safeTemp);
        Tensor softplusTerm = torch.softplus(torch.neg(logitsOverTemp));
        Tensor correctionTerm = torch.mul(safeTemp, softplusTerm);

        // 3. 组合熵
        Tensor entropy = torch.add(bernoulliEntropy, correctionTerm);

        // 4. 单次挤压
        if (entropy.dim() > 0 && batchShape.length > 0) {
            entropy = entropy.squeeze(0);
        }

        // 释放临时张量
        safeClose(safeTemp, safeProbs, oneMinusP, logP, logOneMinusP,
                bernoulliEntropy, logitsOverTemp, softplusTerm, correctionTerm);
        return entropy;
    }

    /**
     * 资源释放
     */
    @Override
    public void close() {
        safeClose(this.temperature, this.probs, this.logits);
    }

    // 辅助方法：安全释放多个张量
    private void safeClose(AutoCloseable... closeables) {
        for (AutoCloseable c : closeables) {
            if (c != null) {
                try {
                    c.close();
                } catch (Exception e) {
                    System.err.println("资源释放失败：" + e.getMessage());
                }
            }
        }
    }

    // Getter方法
    public Tensor getTemperature() { return temperature; }
    public Tensor getProbs() { return probs; }
    public Tensor getLogits() { return logits; }
    public long[] getBatchShape() { return batchShape; }

    // 离散伯努利熵（无无限循环）
    public Tensor bernoulliEntropy() {
        Tensor safeProbs = torch.clamp(
                this.probs,
                new ScalarOptional(TENSOR_EPS.item()),
                new ScalarOptional(torch.sub(TENSOR_1, TENSOR_EPS).item())
        );
        Tensor oneMinusP = torch.sub(TENSOR_1, safeProbs);
        Tensor entropy = torch.neg(torch.add(
                torch.mul(safeProbs, torch.log(safeProbs)),
                torch.mul(oneMinusP, torch.log(oneMinusP))
        ));

        if (entropy.dim() > 0 && batchShape.length > 0) {
            entropy = entropy.squeeze(0);
        }

        safeClose(safeProbs, oneMinusP);
        return entropy;
    }
}
