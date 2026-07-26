package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * OneHotCategorical（独热类别）分布实现
 * 最终修复版本：
 * 1. 从原始概率张量计算类别数（彻底解决one_hot越界）
 * 2. 修复维度索引API调用错误（ScalarTypeOptional误用）
 * 3. 增加索引范围校验，防止采样索引超出类别数
 * 4. 兼容torch.distribution包的Distribution父类API
 */
public class OneHotCategorical extends Distribution implements AutoCloseable {
    private final Categorical categorical;  // 底层类别分布
    private final int numCategories;        // 类别数k（核心修复：从原始probs获取）
    private final Tensor probs;             // 归一化后的类别概率（备份）
    private final TensorOptions baseOptions; // 基础配置（复用）

    // 预定义常量（静态常量不释放！仅作为参数使用）
    private static final Scalar SCALAR_0 = new Scalar(0.0);
    private static final Scalar SCALAR_1 = new Scalar(1.0);
    private static final Scalar SCALAR_EPS = new Scalar(1e-8);
    private static final LongOptional DIM_NEG_1 = new LongOptional(-1);
    private static final float NEG_INF = Float.NEGATIVE_INFINITY;

    /**
     * 构造函数：校验参数合法性 + 初始化底层Categorical分布
     * @param probs 类别概率（形状：batch_shape + [k]，非负，自动归一化）
     * @throws IllegalArgumentException 参数非法时抛出
     */
    public OneHotCategorical(Tensor probs) {
        // 1. 空张量校验
        if (probs == null || probs.numel() == 0) {
            throw new IllegalArgumentException("probs不能为空张量！");
        }

        // 2. 校验probs非负
        Tensor probsNeg = null;
        try {
            probsNeg = torch.lt(probs, torch.tensor(0.0f, probs.options()));
            if (torch.any(probsNeg).item().toBool()) {
                throw new IllegalArgumentException("probs所有元素必须非负！");
            }
        } finally {
            safeClose(probsNeg); // 确保临时张量释放
        }

        // 3. 初始化底层Categorical分布（自动归一化概率）
        this.categorical = new Categorical(probs);

        // 修复1：TensorOptions创建错误（移除错误的ScalarTypeOptional包装）
        this.baseOptions = probs.options().dtype(new ScalarTypeOptional(kFloat())).device(new DeviceOptional(new Device(DeviceType.CPU)));

        // 修复2：核心错误 - 从原始probs计算类别数（而非错误的mean张量）
        if (probs.dim() == 0) {
            // 标量概率 → 1类
            this.numCategories = 1;
        } else {
            // 多维概率 → 最后一维为类别数（安全获取）
            this.numCategories = (int) probs.size(probs.dim() - 1);
        }

        // 5. 备份归一化后的概率（确保至少是1维）
        Tensor rawProbs = this.categorical.getProbs();
        if (rawProbs.dim() == 0) {
            this.probs = rawProbs.unsqueeze(0); // 标量转为1维
        } else {
            this.probs = rawProbs.clone();
        }
    }

    @Override
    public String name() {
        return "OneHotCategorical";
    }

    /**
     * 采样：生成独热编码的采样结果，支持任意批量采样形状
     * 修复：增加索引范围校验 + 维度调整
     */
    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：调用底层Categorical采样索引
        Tensor indices = categorical.sample(sampleShape);

        // 修复3：索引范围校验（防止one_hot越界）
        Tensor indicesClamped = torch.clamp(indices,new ScalarOptional(new Scalar( 0)), new ScalarOptional(new Scalar(numCategories - 1)));

        // 步骤2：转换为Long类型（one_hot要求Long输入）
        Tensor indicesLong = indicesClamped.to(kLong());

        // 步骤3：生成独热编码（保证类别数匹配）
        Tensor oneHot = one_hot(indicesLong, (long) numCategories);

        // 步骤4：调整维度（保证输出形状正确）
        long[] targetShape = getExtendedShape(probs, sampleShape);
        Tensor sample = oneHot.reshape(targetShape);

        // 释放临时张量
        safeClose(indices);
        safeClose(indicesClamped);
        safeClose(indicesLong);
        safeClose(oneHot);

        return sample;
    }

    /**
     * 对数概率：校验独热输入合法性后，转回索引计算对数概率
     * 修复：维度索引API调用错误 + 索引范围校验
     */
    @Override
    public Tensor log_prob(Tensor v) {
        // 步骤1：输入合法性基础校验
        if (v == null || v.numel() == 0) {
            return torch.tensor(NEG_INF, baseOptions);
        }

        // 临时张量声明（集中管理）
        Tensor vSum = null;
        Tensor vIsBinary = null;
        Tensor vIsBinaryAll = null;
        Tensor isValid = null;
        Tensor indices = null;
        Tensor logProbValid = null;
        Tensor logProb = null;

        try {
            // 1.1 校验最后一维为类别数k（安全获取最后一维）
            long lastDimSize = v.dim() == 0 ? 0 : v.size(v.dim() - 1);
            if (lastDimSize != numCategories) {
                throw new IllegalArgumentException(
                        "输入最后一维必须为类别数" + numCategories + "，实际为" + lastDimSize
                );
            }

            // 修复4：sum的维度参数错误（移除ScalarTypeOptional包装）
            vSum = torch.sum(v, -1);
            vIsBinary = torch.logical_or(torch.eq(v, SCALAR_0), torch.eq(v, SCALAR_1));
            // 修复5：all的维度参数统一使用LongOptional
            vIsBinaryAll = torch.all(vIsBinary, -1);
            isValid = torch.logical_and(torch.eq(vSum, SCALAR_1), vIsBinaryAll);

            // 步骤2：计算合法输入的对数概率
            // 2.1 argmax转回索引 + 范围校验（安全处理维度）
            indices = v.argmax(DIM_NEG_1, false);
            indices = torch.clamp(indices,new ScalarOptional(new Scalar( 0)), new ScalarOptional(new Scalar(numCategories - 1))).to(kLong());
            // 2.2 调用底层Categorical的log_prob
            logProbValid = categorical.log_prob(indices);

            // 步骤3：处理非法输入（返回-∞）
            // 修复6：full_like的Scalar参数简化（移除冗余包装）
            logProb = torch.where(
                    isValid,
                    logProbValid,
                    torch.full_like(logProbValid, new Scalar(NEG_INF), baseOptions,new MemoryFormatOptional())
            );

            return logProb;

        } finally {
            // 释放所有临时张量
            safeClose(vSum);
            safeClose(vIsBinary);
            safeClose(vIsBinaryAll);
            safeClose(isValid);
            safeClose(indices);
            safeClose(logProbValid);
            // logProb作为返回值，不释放
        }
    }

    /**
     * 均值：与底层Categorical分布一致（类别概率）
     */
    @Override
    public Tensor mean() {
        Tensor mean = categorical.mean();
        // 确保返回至少1维张量
        if (mean.dim() == 0) {
            Tensor result = mean.unsqueeze(0);
            mean.close();
            return result;
        }
        return mean.clone();
    }

    /**
     * 熵：与底层Categorical分布一致
     */
    @Override
    public Tensor entropy() {
        return categorical.entropy().clone();
    }

    /**
     * 安全释放资源（核心：静态常量不释放！）
     */
    @Override
    public void close() {
        safeClose(categorical);
        safeClose(probs);
        // 静态常量不能释放！否则后续使用会崩溃
    }

    /**
     * 辅助方法：安全释放AutoCloseable资源
     */
    private void safeClose(AutoCloseable closeable) {
        if (closeable != null) {
            try {
                closeable.close();
            } catch (Exception e) {
                System.err.println("资源释放警告：" + e.getMessage());
            }
        }
    }

    // Getter方法（符合torch.distribution API规范）
    public Tensor getProbs() { return probs.clone(); } // 返回拷贝保护内部状态
    public int getNumCategories() { return numCategories; }
    public Categorical getCategorical() { return categorical; }
}
