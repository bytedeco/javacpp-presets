package org.bytedeco.pytorch.nn;

import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.data.transforms.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;

public class Functional {

    /**
     * BRO: Batch Representation Orthogonality penalty
     * 迫使特征维度之间去相关 (Disentanglement)。
     * Loss = || M^T M - I || (Frobenius Norm)
     * M: Normalized Batch Features [Batch, Dim]
     */
    public static Tensor bro_penalty(Tensor x) {
        long N = x.size(0);
        long D = x.size(1);

        // 1. Normalize columns (features) to have unit norm
        // dim=0
        NormalizeFuncOptions opt = new NormalizeFuncOptions();
        opt.p().put(2);
        opt.dim().put(0);
        Tensor xNorm = torch.normalize(x, opt);

        // 2. Compute Correlation Matrix: M^T M -> [D, D]
        Tensor corr = xNorm.t().matmul(xNorm);

        // 3. Identity Matrix
        Tensor eye = torch.eye(D, x.options());

        // 4. Frobenius Norm
        return corr.sub(eye).norm();
    }

    /**
     * Gini Coefficient
     * 衡量稀疏性 (0 = complete equality/dense, 1 = complete inequality/sparse)
     * 公式: G = (2 * sum(i * x_sorted_i)) / (n * sum(x_i)) - (n + 1)/n
     */
    public static Tensor gini(Tensor x) {
        // 1. 必须展平为一维向量，因为 Gini 是衡量一组数值的分布不均
        Tensor xFlat = x.abs().view(-1).add(new Scalar(1e-6));;
        long n = xFlat.size(0); // 这里 n 会是 128 * 64 = 8192

        // 2. 排序 (从小到大)
        Tensor xSorted = torch.sort(xFlat).get0();

        // 3. 创建索引 [1, 2, ..., n]
        // 确保 index 的形状和 xSorted 一致，都是 [n]
        Tensor index = torch.arange(new Scalar(1),new Scalar( n + 1), x.options());

        // 4. 计算公式: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1)/n
        // 分子: 2 * sum(index * xSorted)
        Tensor num = index.mul(xSorted).sum().mul(new Scalar(2.0));

        // 分母: n * sum(xSorted)
        Tensor den = xSorted.sum().mul(new Scalar((double)n));

        // 5. 结果
        return num.div(den).sub(new Scalar((double)(n + 1) / n));
    }
    public static Tensor gini2(Tensor x) {
        // x: [N] (Flattened positive values)
        // Ensure x is positive
        Tensor xAbs = x.abs().add(new Scalar(1e-6)); // avoid div 0
        long n = x.size(0);

        // 1. Sort
        Tensor xSorted = torch.sort(xAbs).get0();  //get value  ????

        // 2. Indices: 1, 2, ..., n
        Tensor index = torch.arange(new Scalar(1),new Scalar( n + 1), x.options());

        // 3. Numerator: 2 * sum(i * x_i)
        Tensor num = index.mul(xSorted).sum().mul(new Scalar(2.0));

        // 4. Denominator: n * sum(x_i)
        Tensor den = xSorted.sum().mul(new Scalar((double)n));

        // 5. Result
        return num.div(den).sub(new Scalar((double)(n + 1) / n));
    }
}