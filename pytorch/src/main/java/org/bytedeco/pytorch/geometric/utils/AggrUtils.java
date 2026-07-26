package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

/**
 * 聚合操作底层工具箱
 */
public class AggrUtils {

    // 基础 org.bytedeco.pytorch.geometric.utils.Scatter 操作
    public static Tensor scatter(Tensor src, Tensor index, long dimSize, String reduce) {
        // 构造输出形状: [dimSize, F...]
        long[] srcShape = src.shape();
        long[] outShape = new long[srcShape.length];
        outShape[0] = dimSize;
        System.arraycopy(srcShape, 1, outShape, 1, srcShape.length - 1);

        Tensor out = torch.zeros(outShape, src.options());
        if ("prod".equals(reduce) || "mul".equals(reduce)) {
            // 初始化为 1.0
            Tensor out2 = torch.ones(outShape, src.options());
            return out2.index_reduce_(0, index, src, "prod", false);
        }
        if ("add".equals(reduce) || "sum".equals(reduce)) {
            return out.index_add_(0, index, src);
        } else if ("mean".equals(reduce)) {
            Tensor sum = out.index_add_(0, index, src);
            Tensor count = torch.zeros(new long[]{dimSize}, src.options());
            Tensor ones = torch.ones(new long[]{src.size(0)}, src.options());
            count.index_add_(0, index, ones);
            count = count.clamp_min(new Scalar(1.0));

            // 广播 count
            for (int i = 1; i < outShape.length; i++) count = count.unsqueeze(i);
            return sum.div(count);
        } else if ("max".equals(reduce)) {
            // 初始化为极小值
            out.fill_(new Scalar(-1.0e38));
            return out.index_reduce_(0, index, src, "amax", false);
        } else if ("min".equals(reduce)) {
            // 初始化为极大值
            out.fill_(new Scalar(1.0e38));
            return out.index_reduce_(0, index, src, "amin", false);
        }
        throw new UnsupportedOperationException("Unknown reduce: " + reduce);
    }

    // 实现 org.bytedeco.pytorch.geometric.utils.Scatter Softmax: exp(x_i) / sum(exp(x_j))
    // 用于 Attention 和 org.bytedeco.pytorch.geometric.aggr.SoftmaxAggregation
    public static Tensor scatter_softmax(Tensor src, Tensor index, long dimSize) {
        // 1. 数值稳定性: x - max(x)
        Tensor maxVal = scatter(src, index, dimSize, "max"); // [N, F]

        // 将 maxVal 映射回边维度 [E, F]
        Tensor maxExpanded = maxVal.index_select(0, index);

        // 2. 计算 exp
        Tensor num = src.sub(maxExpanded).exp();

        // 3. 计算分母 sum
        Tensor den = scatter(num, index, dimSize, "sum"); // [N, F]
        Tensor denExpanded = den.index_select(0, index);

        // 4. 除法 (加 eps 防止除0)
        return num.div(denExpanded.add(new Scalar(1e-12)));
    }
    

    // 计算节点的度 (Degree)
    public static Tensor compute_degree(Tensor index, long dimSize) {
        Tensor ones = torch.ones(new long[]{index.size(0)}, index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
        Tensor out = torch.zeros(new long[]{dimSize}, index.options().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
        return out.index_add_(0, index, ones);
    }

    // 请添加到 org.gnn.framework.utils.org.bytedeco.pytorch.geometric.utils.AggrUtils 类中

    /**
     * 将稀疏的邻居特征转换为稠密 Batch 格式，便于进行 Median/Quantile/LSTM 计算。
     *
     * @param x       特征 [E, F]
     * @param index   索引 [E]
     * @param dimSize 目标节点数 N
     * @param fillValue 填充值 (通常是 0 或 NaN)
     * @return Tensor[] {dense_x, mask, lengths}
     * dense_x: [N, MaxDeg, F]
     * mask:    [N, MaxDeg] (Boolean, True表示有效数据)
     * lengths: [N] (每个节点的度)
     */
    public static Tensor[] to_dense_batch(Tensor x, Tensor index, long dimSize, float fillValue) {
        long numEdges = x.size(0);
        long numFeatures = x.size(1);

        // 1. 计算度数 (Lengths)
        Tensor lengths = compute_degree(index, dimSize).to(torch.ScalarType.Long); // [N]
        long maxDeg = lengths.max().item().toLong();

        // 2. 排序 Index 以确保分组连续
        // 这一步对于生成 inner_index 至关重要
        T_TensorTensor_T sortRet = torch.sort(index);
        Tensor perm = sortRet.get1(); //indices
        Tensor sortedIndex = sortRet.get0(); //values
        Tensor sortedX = x.index_select(0, perm);

        // 3. 生成 Inner Index (组内索引 0, 1, 2...)
        // 算法：arange(E) - cumsum(counts)[sorted_index] + counts[sorted_index] - 这里的逻辑比较复杂
        // 我们采用更稳健的 searchsorted 方法:
        // inner_idx = arange(E) - starts[sorted_index]

        // 3.1 找到每组的起始位置
        // Cumulative sum of lengths gives end positions. Shift to get start.
        Tensor endPos = torch.cumsum(lengths, 0);
        Tensor startPos = torch.cat(new TensorVector(torch.zeros(new long[]{1}, lengths.options()), endPos.slice(0, new LongOptional(0), new LongOptional(dimSize - 1), 1l)), 0);

        // 3.2 扩展 Start Position 到每条边
        Tensor edgeStartPos = startPos.index_select(0, sortedIndex);

        // 3.3 计算组内偏移量
        Tensor range = torch.arange(new Scalar(numEdges), index.options());
        Tensor innerIdx = range.sub(edgeStartPos); // [E]

        // 4. org.bytedeco.pytorch.geometric.utils.Scatter 填充 Dense Tensor
        // 目标: dense[sortedIndex, innerIdx, :] = sortedX
        // 由于 LibTorch 没有 scatter_nd (或者是高级索引比较麻烦)，我们展平前两维进行 scatter

        // Init with fillValue
        Tensor dense = torch.full(new long[]{dimSize, maxDeg, numFeatures}, new Scalar(fillValue), x.options());

        // 计算 Flatten Index: idx * maxDeg + inner
        Tensor flatIdx = sortedIndex.mul(new Scalar(maxDeg)).add(innerIdx); // [E]

        // Flatten dense to [N*MaxDeg, F]
        Tensor denseFlat = dense.view(dimSize * maxDeg, numFeatures);

        // org.bytedeco.pytorch.geometric.utils.Scatter: denseFlat[flatIdx] = sortedX
        // index_copy_ 或 index_add_ (如果初始为0)
        // 这里用 index_copy_ 确保覆盖
        denseFlat.index_copy_(0, flatIdx, sortedX);

        // Reshape back
        dense = denseFlat.view(dimSize, maxDeg, numFeatures);

        // 5. 生成 Mask
        // mask[i, j] = j < lengths[i]
        Tensor degRange = torch.arange(new Scalar(maxDeg), lengths.options()).unsqueeze(0); // [1, MaxDeg]
        Tensor mask = degRange.lt(lengths.unsqueeze(1)); // [N, MaxDeg]

        return new Tensor[]{dense, mask, lengths};
    }
}

