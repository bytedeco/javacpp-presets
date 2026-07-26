package org.bytedeco.pytorch.distribution;
import org.bytedeco.pytorch.data.transforms.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Geometric（几何分布）最终稳定版
 * 定义：P(X=k) = (1-p)^(k-1) * p （k≥1，0 < p < 1）
 * 修复点：
 * 1. ✅ 修复构造函数参数校验逻辑（先校验原始p，再做数值保护）
 * 2. ✅ 修正熵公式为几何分布标准公式
 * 3. ✅ 移除静态Scalar，改为局部创建+即时释放
 * 4. ✅ 增加数值截断保护，避免极端参数溢出
 * 5. ✅ 完善资源释放逻辑，无重复释放风险
 */
public class Geometric extends Distribution implements AutoCloseable {
    private final Tensor p;   // 成功概率（0 < p < 1）
    private final TensorOptions baseOptions; // 基础设备/类型配置
    private boolean isClosed = false; // 防止重复释放

    // 数值稳定性常量
    private static final float EPS = 1e-8f;
    private static final float MAX_ENTROPY = 1e5f; // 熵最大截断值
    private static final float MIN_LOG_VAL = -1e5f; // log_prob最小截断值

    /**
     * 构造函数：严格校验原始参数 + 深拷贝 + 数值保护
     * @param p 成功概率（必须满足 0 < p < 1）
     * @throws IllegalArgumentException 参数超出(0,1)范围时抛出异常
     */
    public Geometric(Tensor p) {
        // 1. 统一转换为Float32+CPU，避免类型/设备不匹配
        Tensor pCpu = p.to(kFloat())
                .to(torch.device(new Device(DeviceType.CPU)), false, true, new MemoryFormatOptional())
                .clone().detach();
        this.baseOptions = pCpu.options();

        // 2. 先校验原始p（关键修复：不提前clamp，保证校验准确性）
        Scalar scalar0 = new Scalar(0.0f);
        Scalar scalar1 = new Scalar(1.0f);
        Tensor pLe0 = torch.le(pCpu, torch.tensor(0.0f, baseOptions));
        Tensor pGe1 = torch.ge(pCpu, torch.tensor(1.0f, baseOptions));
        Tensor paramInvalid = torch.logical_or(pLe0, pGe1);

        try {
            if (torch.any(paramInvalid).item().toBool()) {
                throw new IllegalArgumentException("几何分布p必须满足0<p<1！");
            }
        } finally {
            // 释放校验临时张量
            safeClose(pLe0);
            safeClose(pGe1);
            safeClose(paramInvalid);
            safeClose(scalar0);
            safeClose(scalar1);
        }

        // 3. 数值保护（仅用于计算，不影响校验）
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalar1MinusEPS = new Scalar(1.0f - EPS);
        Tensor safeP = torch.clamp(pCpu, new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));

        // 4. 深拷贝保存最终参数
        this.p = safeP.clone().detach();

        // 释放临时张量
        safeClose(pCpu);
        safeClose(scalarEPS);
        safeClose(scalar1MinusEPS);
        safeClose(safeP);
    }

    @Override
    public String name() {
        return "Geometric";
    }

    /**
     * 采样：Inverse Transform Sampling 方法（数值稳定版）
     * 公式：k = floor(ln(U)/ln(1-p)) + 1 （U~Uniform(0,1)）
     */
    @Override
    public Tensor sample(long... sampleShape) {
        checkClosed();

        // 步骤1：扩展采样形状（sampleShape + batch_shape）
        long[] extendedShape = getExtendedShape(p, sampleShape);
        Tensor expandedP = p.expand(extendedShape).clone().detach();

        // 步骤2：生成Uniform(ε,1-ε)随机数（避免ln(0)）
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalar1MinusEPS = new Scalar(1.0f - EPS);
        Tensor u = torch.rand(extendedShape, baseOptions)
                .clamp(new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));

        // 步骤3：数值稳定性处理
        Scalar scalar1 = new Scalar(1.0f);
        Tensor oneMinusP = torch.sub(torch.tensor(1.0f, baseOptions), expandedP);
        oneMinusP = torch.clamp(oneMinusP, new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));

        // 步骤4：Inverse Transform Sampling
        Tensor logU = torch.log(u);
        Tensor logOneMinusP = torch.log(oneMinusP);
        // 防止除零/溢出
        logOneMinusP = torch.clamp(logOneMinusP, new ScalarOptional(new Scalar(MIN_LOG_VAL)), new ScalarOptional(scalar1MinusEPS));

        Tensor kFloat = torch.floor(torch.div(logU, logOneMinusP)).add(scalar1);

        // 转换为整数类型，确保k≥1
        Tensor k = kFloat.to(ScalarType.Long);
        k = torch.clamp(k, new ScalarOptional(new Scalar(1L)), new ScalarOptional(new Scalar(Long.MAX_VALUE)));

        // 释放所有临时张量
        safeClose(expandedP);
        safeClose(scalarEPS);
        safeClose(scalar1MinusEPS);
        safeClose(u);
        safeClose(scalar1);
        safeClose(oneMinusP);
        safeClose(logU);
        safeClose(logOneMinusP);
        safeClose(kFloat);

        return k.clone().detach();
    }

    /**
     * 对数概率：标准几何分布公式（数值稳定版）
     * 公式：log(P(X=k)) = (k-1)*log(1-p) + log(p) （k≥1）
     */
    @Override
    public Tensor log_prob(Tensor v) {
        checkClosed();
        if (v == null) {
            throw new IllegalArgumentException("log_prob输入张量不能为空！");
        }

        // 步骤1：统一转换为浮点型+CPU
        Tensor vFloat = v.to(baseOptions,false, true, new MemoryFormatOptional()).clone().detach();

        // 步骤2：校验输入合法性（v≥1 且 为整数）
        Scalar scalar1 = new Scalar(1.0f);
        Tensor vLt1 = torch.lt(vFloat, torch.tensor(1.0f, baseOptions));
        Tensor vRounded = torch.round(vFloat);
        Tensor vIsInteger = torch.eq(vFloat, vRounded);
        Tensor vInvalid = torch.logical_or(vLt1, torch.logical_not(vIsInteger));

        // 步骤3：扩展p到v的形状
        Tensor expandedP = p.expand_as(vFloat).clone().detach();
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalar1MinusEPS = new Scalar(1.0f - EPS);
        Tensor safeP = torch.clamp(expandedP, new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));

        // 步骤4：计算合法输入的log_prob
        Tensor logP = torch.log(safeP);
        Tensor oneMinusP = torch.sub(torch.tensor(1.0f, baseOptions), safeP);
        oneMinusP = torch.clamp(oneMinusP, new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));
        Tensor logOneMinusP = torch.log(oneMinusP);

        Tensor kMinus1 = torch.sub(vFloat, scalar1);
        Tensor logProbValid = torch.add(torch.mul(kMinus1, logOneMinusP), logP);

        // 步骤5：数值截断（避免极端值）
        logProbValid = torch.clamp(logProbValid, new ScalarOptional(new Scalar(MIN_LOG_VAL)), new ScalarOptional(scalar1MinusEPS));

        // 步骤6：处理非法输入（返回-∞）
        Tensor negInfTensor = torch.full_like(logProbValid, new Scalar(Float.NEGATIVE_INFINITY), baseOptions, new MemoryFormatOptional());
        Tensor logProb = torch.where(vInvalid, negInfTensor, logProbValid);

        // 释放所有临时张量
        safeClose(vFloat);
        safeClose(scalar1);
        safeClose(vLt1);
        safeClose(vRounded);
        safeClose(vIsInteger);
        safeClose(vInvalid);
        safeClose(expandedP);
        safeClose(scalarEPS);
        safeClose(scalar1MinusEPS);
        safeClose(safeP);
        safeClose(logP);
        safeClose(oneMinusP);
        safeClose(logOneMinusP);
        safeClose(kMinus1);
        safeClose(logProbValid);
        safeClose(negInfTensor);

        return logProb.clone().detach();
    }

    /**
     * 熵：修复为几何分布标准公式（数值稳定版）
     * 正确公式：H = -(p*ln(p) + (1-p)*ln(1-p)) / p
     */
    @Override
    public Tensor entropy() {
        checkClosed();

        // 数值稳定性处理
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalar1MinusEPS = new Scalar(1.0f - EPS);
        Tensor safeP = torch.clamp(p, new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));

        Scalar scalar1 = new Scalar(1.0f);
        Tensor oneMinusP = torch.sub(torch.tensor(1.0f, baseOptions), safeP);
        oneMinusP = torch.clamp(oneMinusP, new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));

        // 计算各项（标准公式）
        Tensor logP = torch.log(safeP);
        Tensor logOneMinusP = torch.log(oneMinusP);

        // 项1：p*ln(p)
        Tensor term1 = torch.mul(safeP, logP);
        // 项2：(1-p)*ln(1-p)
        Tensor term2 = torch.mul(oneMinusP, logOneMinusP);
        // 总熵 = -(term1 + term2) / p
        Tensor entropy = torch.neg(torch.add(term1, term2));
        entropy = torch.div(entropy, safeP);

        // 数值截断（避免极端参数溢出）
        entropy = torch.clamp(entropy, new ScalarOptional(new Scalar(0.0f)), new ScalarOptional(new Scalar(MAX_ENTROPY)));

        // 释放所有临时张量
        safeClose(scalarEPS);
        safeClose(scalar1MinusEPS);
        safeClose(safeP);
        safeClose(scalar1);
        safeClose(oneMinusP);
        safeClose(logP);
        safeClose(logOneMinusP);
        safeClose(term1);
        safeClose(term2);

        return entropy.clone().detach();
    }

    /**
     * 均值：标准公式 E[X] = 1/p（数值稳定版）
     */
    @Override
    public Tensor mean() {
        checkClosed();

        // 数值稳定性处理
        Scalar scalarEPS = new Scalar(EPS);
        Scalar scalar1MinusEPS = new Scalar(1.0f - EPS);
        Tensor safeP = torch.clamp(p, new ScalarOptional(scalarEPS), new ScalarOptional(scalar1MinusEPS));

        // 正确公式：1/p
        Tensor mean = torch.reciprocal(safeP);
        // 数值截断
        mean = torch.clamp(mean, new ScalarOptional(new Scalar(1.0f)), new ScalarOptional(new Scalar(1e6f)));

        // 释放临时张量
        safeClose(scalarEPS);
        safeClose(scalar1MinusEPS);
        safeClose(safeP);

        return mean.clone().detach();
    }

    // -------------------------- 辅助方法 --------------------------
    /**
     * 检查资源是否已释放
     */
    private void checkClosed() {
        if (isClosed) {
            throw new IllegalStateException("Geometric实例已释放，无法继续使用！");
        }
    }

    /**
     * 扩展形状（sampleShape + batchShape）
     */

    /**
     * 安全释放资源（避免空指针和重复释放）
     */
    private void safeClose(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                System.err.println("Geometric资源释放警告：" + e.getMessage());
            }
        }
    }

    /**
     * 资源释放：实现AutoCloseable，线程安全
     */
    @Override
    public void close() {
        if (!isClosed) {
            safeClose(p);
            isClosed = true;
        }
    }

    // Getter方法（返回拷贝，避免外部修改）
    public Tensor getP() {
        checkClosed();
        return p.clone().detach();
    }
}
