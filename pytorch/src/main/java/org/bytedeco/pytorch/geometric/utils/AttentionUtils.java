package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

public class AttentionUtils {

    /**
     * Polynormer / Linear Transformer 常用的核函数
     * phi(x) = elu(x) + 1 (保证非负)
     */
    public static Tensor kernel_elu(Tensor x) {
        return torch.elu(x).add(new Scalar(1.0));
    }

    /**
     * Performer 的正交随机特征 (ORF) 生成器
     * 生成正交的高斯随机矩阵 Omega
     */
    public static Tensor create_projection_matrix(long numFeatures, long dim, boolean orthogonal) {
        if (orthogonal) {
            // 生成正交矩阵块
            long numBlocks = (long) Math.ceil((double) numFeatures / dim);
            TensorVector blocks = new TensorVector();
            for (int i = 0; i < numBlocks; i++) {
                Tensor mat = torch.randn(new long[]{dim, dim});
                // QR 分解获取正交矩阵 Q
                T_TensorTensor_T qr = torch.linalg_qr(mat);
                blocks.put(qr.get0()); // Q
            }
            Tensor full = torch.cat(blocks, 0);
            // 截取所需行数
            return full.slice(0, new LongOptional(0), new LongOptional(numFeatures), 1l).t(); // [Dim, M]
        } else {
            return torch.randn(new long[]{dim, numFeatures}).t(); // [Dim, M]
        }
    }

    /**
     * Performer 核函数
     * phi(x) = C * exp(x @ w - |x|^2 / 2)
     */
    public static Tensor kernel_performer(Tensor data, Tensor projectionMatrix, boolean isQuery) {
        // data: [N, D]
        // projection: [D, M]

        // 1. 计算数据模长的平方 / 2
        // sum(dim=-1, keepdim=true)
        Tensor dataSq = data.pow(new Scalar(2.0)).sum(new long[]{-1}, true, new ScalarTypeOptional()).div(new Scalar(2.0));

        // 2. 投影 data @ projection -> [N, M]
        // 缩放系数: 1 / sqrt(sqrt(M)) -> M^(-0.25)
        double scale = Math.pow(projectionMatrix.size(1), -0.25);
        Tensor proj = data.matmul(projectionMatrix).mul(new Scalar(scale));

        // 3. 组合: exp(proj - dataSq)
        // 注意广播机制
        return proj.sub(dataSq).exp();
    }
}