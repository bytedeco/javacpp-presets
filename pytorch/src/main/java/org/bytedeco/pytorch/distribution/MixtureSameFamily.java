package org.bytedeco.pytorch.distribution;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * MixtureSameFamily（同分布族混合分布）实现
 * mixture_dist：类别分布（Categorical），提供混合权重π
 * component_dist：同分布族的组件分布（如Normal/Uniform），需支持批量组件参数
 * 支持任意批量维度，严格遵循混合分布的数学定义，具备完整的维度对齐和数值稳定性
 */
public class MixtureSameFamily extends Distribution implements AutoCloseable {
    private final Categorical mixtureDist;          // 类别分布（混合权重π）
    private final Distribution componentDist;       // 组件分布族（如Normal/K个组件）
    private final int numComponents;                // 组件数量K
    private Tensor mixtureLogProbs;                 // 预计算log(π)，提升效率

    // 预定义常量（改为实例变量，避免静态释放导致空指针）
    private final Scalar SCALAR_EPS;
    private final Scalar SCALAR_NEG_INF;
    private final LongOptional DIM_LAST;
    private final boolean KEEP_DIMS = false;

    /**
     * 构造函数：校验参数合法性 + 预计算关键值
     * @param mixture 类别分布（必须输出组件索引，权重和为1）
     * @param component 组件分布（需支持K个组件的批量参数）
     * @throws IllegalArgumentException 参数非法/维度不匹配抛出异常
     */
    public MixtureSameFamily(Categorical mixture, Distribution component) {
        // 1. 空值校验
        if (mixture == null || component == null) {
            throw new IllegalArgumentException("mixture（类别分布）和component（组件分布）不能为空！");
        }

        // 2. 获取组件数量并校验
        this.mixtureDist = mixture;
        this.componentDist = component;
        this.numComponents = mixture.numCategories(); // 从Categorical获取组件数K
        if (numComponents <= 1) {
            throw new IllegalArgumentException("混合分布组件数量必须>1（当前K=" + numComponents + "）！");
        }

        // 初始化常量（实例化而非静态，避免提前释放）
        this.SCALAR_EPS = new Scalar(1e-8);
        this.SCALAR_NEG_INF = new Scalar(Double.NEGATIVE_INFINITY);
        this.DIM_LAST = new LongOptional(-1);

        // 3. 预计算log(π)（数值稳定处理，避免log(0)）
        Tensor mixtureProbs = mixture.getProbs().clone(); // 克隆避免原张量被释放
        this.mixtureLogProbs = log(torch.clamp(mixtureProbs, new ScalarOptional(SCALAR_EPS), new ScalarOptional(new Scalar(1.0 - 1e-8))));

        // 4. 校验组件分布维度适配性（需支持K个组件）
        Tensor componentMean = component.mean();
        long[] componentBatchShape = componentMean.sizes().vec().get();
        componentMean.close();

        if (componentBatchShape.length == 0 || componentBatchShape[componentBatchShape.length - 1] != numComponents) {
            throw new IllegalArgumentException(
                    String.format("组件分布最后一维必须为组件数K=%d（当前：%d）",
                            numComponents,
                            componentBatchShape.length > 0 ? componentBatchShape[componentBatchShape.length - 1] : 0)
            );
        }

        // 释放临时张量
        mixtureProbs.close();
    }

    @Override
    public String name() {
        return String.format("MixtureSameFamily(K=%d, %s)", numComponents, componentDist.name());
    }

    /**
     * 采样：严格遵循混合分布采样逻辑
     * 步骤1：从类别分布采样组件索引k（形状：sample_shape + batch_shape）
     * 步骤2：从对应组件分布P_k采样x（形状：sample_shape + batch_shape + K + event_shape）
     * 步骤3：根据索引gather对应组件的样本
     * @param sampleShape 批量采样形状
     * @return 混合分布采样结果（形状：sample_shape + batch_shape + event_shape）
     */

    @Override
    public Tensor sample(long... sampleShape) {
        // 步骤1：采样组件索引（形状：sample_shape + batch_shape）
        Tensor componentIndices = mixtureDist.sample(sampleShape);
        long[] indicesShape = componentIndices.sizes().vec().get();

        // 步骤2：从组件分布采样所有组件的样本（形状：sample_shape + batch_shape + K + event_shape）
        Tensor allComponentSamples = componentDist.sample(sampleShape);
        long[] sampleSizes = allComponentSamples.sizes().vec().get();

        // 步骤3：扩展索引维度以匹配组件样本
        // 索引形状：sample_shape + batch_shape → 扩展为 sample_shape + batch_shape + 1
        Tensor expandedIndices = componentIndices.unsqueeze(-1);

        // 修复：手动截取前 N 个维度（去掉 event 维度）
        int eventDim = getEventDim(componentDist);
        int targetLength = sampleSizes.length - eventDim;
        long[] expandShape = new long[targetLength];
        System.arraycopy(sampleSizes, 0, expandShape, 0, targetLength);
        expandedIndices = expandedIndices.expand(expandShape);

        // 步骤4：根据索引gather对应组件的样本
        Tensor samples = allComponentSamples.gather(-(eventDim + 1), expandedIndices);
        // 移除组件维度
        samples = samples.squeeze(-(eventDim + 1));

        // 释放临时张量（保留返回值）
        componentIndices.close();
        allComponentSamples.close();
        expandedIndices.close();

        return samples;
    }

    /**
     * 均值：混合分布的精确均值 = Σπ_k·E[X_k]
     * @return 均值张量（形状：batch_shape + event_shape）
     */
    @Override
    public Tensor mean() {
        // 步骤1：获取组件均值（形状：batch_shape + K + event_shape）
        Tensor componentMeans = componentDist.mean().clone();
        int eventDim = getEventDim(componentDist);
        int componentDim = (int)componentMeans.dim() - eventDim - 1;

        // 步骤2：扩展混合权重到组件均值形状
        Tensor mixtureProbs = mixtureDist.getProbs().clone();
        long[] componentMeansSizes = componentMeans.sizes().vec().get();

        // 修复：手动截取前 N 个维度（去掉 event 维度）
        int targetLength = componentMeansSizes.length - eventDim;
        long[] expandShape = new long[targetLength];
        System.arraycopy(componentMeansSizes, 0, expandShape, 0, targetLength);

        Tensor expandedMixtureProbs = mixtureProbs.expand(expandShape);

        // 扩展event维度
        for (int i = 0; i < eventDim; i++) {
            expandedMixtureProbs = expandedMixtureProbs.unsqueeze(-1);
        }
        expandedMixtureProbs = expandedMixtureProbs.expand(componentMeans.sizes());

        // 步骤3：计算π_k·E[X_k]
        Tensor weightedMeans = componentMeans.mul(expandedMixtureProbs);

        // 步骤4：沿组件维度求和
        Tensor mixtureMean = weightedMeans.sum(new long[]{componentDim}, KEEP_DIMS, new ScalarTypeOptional());

        // 释放临时张量
        componentMeans.close();
        mixtureProbs.close();
        expandedMixtureProbs.close();
        weightedMeans.close();

        return mixtureMean;
    }
    /**
     * 工具方法：截取数组前 N 个元素
     */
    private long[] takeHead(long[] array, int n) {
        if (array == null || n <= 0) {
            return new long[0];
        }
        n = Math.min(n, array.length); // 防止越界
        long[] result = new long[n];
        System.arraycopy(array, 0, result, 0, n);
        return result;
    }
    
    /**
     * 对数概率：实现混合分布精确对数概率公式
     * 公式：logP(x) = logsumexp(logπ_k + logP_k(x))（沿组件维度求和）
     * @param v 输入张量（形状：batch_shape + event_shape）
     * @return 对数概率张量（形状：batch_shape）
     */
    @Override
    public Tensor log_prob(Tensor value) {
        // 步骤1：扩展value到组件维度（添加K维度）
        int eventDim = getEventDim(componentDist);
        int componentDim = (int)value.dim() - eventDim;

        // 插入组件维度（K）
        Tensor expandedValue = value.unsqueeze(componentDim);

        // 步骤2：计算所有组件的log_prob（形状：batch_shape + K + event_shape）
        Tensor componentLogProbs = componentDist.log_prob(expandedValue);

        // 步骤3：获取混合权重（形状：batch_shape + K）
        Tensor mixtureProbs = mixtureDist.getProbs().clone();

        // 扩展权重到log_prob形状（匹配event维度）
        long[] logProbsShape = componentLogProbs.sizes().vec().get();
        Tensor expandedMixtureProbs = mixtureProbs.expand(logProbsShape);

        // 步骤4：计算 log(Σπ_k * exp(log_p_k)) = log_sum_exp(log_p_k + logπ_k)
        Tensor logMixtureProbs = torch.log(expandedMixtureProbs);
        Tensor logWeightedProbs = componentLogProbs.add(logMixtureProbs);

        // 沿组件维度计算log_sum_exp
        Tensor logProb = torch.logsumexp(logWeightedProbs, new long[]{componentDim}, false);

        // 释放临时张量
        expandedValue.close();
        componentLogProbs.close();
        mixtureProbs.close();
        expandedMixtureProbs.close();
        logMixtureProbs.close();
        logWeightedProbs.close();

        return logProb;
    }

    /**
     * 均值：混合分布的精确均值 = Σπ_k·E[X_k]
     * @return 均值张量（形状：batch_shape + event_shape）
     */

    /**
     * 熵：混合分布的精确熵 = 类别分布熵 + Σπ_k·组件熵
     * 公式：H = H_cat(π) + Σπ_k·H_k
     * @return 熵张量（形状：batch_shape）
     */
    @Override
    public Tensor entropy() {
        // 步骤1：类别分布熵 H_cat(π)（形状：batch_shape）
        Tensor mixtureEntropy = mixtureDist.entropy().clone();

        // 步骤2：组件熵 H_k（形状：batch_shape + K + event_shape）
        Tensor componentEntropies = componentDist.entropy().clone();
        int eventDim = getEventDim(componentDist);
        int componentDim = (int)componentEntropies.dim() - eventDim - 1;

        // 步骤3：求和event维度
        if (eventDim > 0) {
            long[] reduceDims = new long[eventDim];
            for (int i = 0; i < eventDim; i++) {
                reduceDims[i] = componentEntropies.dim() - 1 - i;
            }
            componentEntropies = componentEntropies.sum(reduceDims, KEEP_DIMS, new ScalarTypeOptional());
        }

        // 步骤4：扩展混合权重到组件熵形状
        Tensor mixtureProbs = mixtureDist.getProbs().clone();
        Tensor expandedMixtureProbs = mixtureProbs.expand(componentEntropies.sizes());

        // 步骤5：计算Σπ_k·H_k
        Tensor weightedComponentEntropies = componentEntropies.mul(expandedMixtureProbs);
        Tensor meanComponentEntropy = weightedComponentEntropies.sum(new long[]{componentDim}, KEEP_DIMS, new ScalarTypeOptional());

        // 步骤6：扩展类别分布熵到结果形状
        Tensor expandedMixtureEntropy = mixtureEntropy.expand(meanComponentEntropy.sizes());

        // 步骤7：总熵 = 类别熵 + 加权组件熵
        Tensor totalEntropy = expandedMixtureEntropy.add(meanComponentEntropy);

        // 释放临时张量
        mixtureEntropy.close();
        componentEntropies.close();
        mixtureProbs.close();
        expandedMixtureProbs.close();
        weightedComponentEntropies.close();
        meanComponentEntropy.close();
        expandedMixtureEntropy.close();

        return totalEntropy;
    }

    // ------------------------------ 辅助方法 ------------------------------
    /**
     * 获取组件分布的事件维度（event_dim）
     * 例如：标量分布event_dim=0，向量分布event_dim=1，矩阵分布event_dim=2
     */
    private int getEventDim(Distribution dist) {
        // 安全实现：通过分布类型推断（Normal是标量分布，event_dim=0）
        if (dist instanceof Normal) {
            return 0;
        }
        // 通用实现：通过均值形状推断
        Tensor mean = dist.mean();
        long batchDim = mixtureLogProbs.dim();
        long meanDim = mean.dim();
        mean.close();
        return Math.max((int)(meanDim - batchDim), 0);
    }

    /**
     * 资源释放：实现AutoCloseable，避免native内存泄漏
     */
    @Override
    public void close() {
        // 释放预计算的logπ
        if (mixtureLogProbs != null) {
            mixtureLogProbs.close();
        }

        // 释放实例常量

        // 注意：不释放mixtureDist和componentDist！
        // 因为这些分布可能被外部复用，应由创建者负责释放
    }

    // Getter方法
    public Categorical getMixtureDist() { return mixtureDist; }
    public Distribution getComponentDist() { return componentDist; }
    public int getNumComponents() { return numComponents; }
    public Tensor getMixtureLogProbs() { return mixtureLogProbs; }
}
