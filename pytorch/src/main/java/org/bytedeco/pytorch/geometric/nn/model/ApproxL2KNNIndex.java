package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.geometric.nn.Parameter;
/**
 * Approx L2 KNN (via Random Projection)
 * 将高维数据投影到低维，然后在低维空间做 Exact Search。
 */
public class ApproxL2KNNIndex extends L2KNNIndex {
    private long projectionDim;

    public ApproxL2KNNIndex(long k, long projectionDim) {
        super(k);
        this.projectionDim = projectionDim;
    }

    @Override
    public Tensor[] search(Tensor x, Tensor y, Tensor batchX, Tensor batchY) {
        long dim = x.size(1);
        // 生成随机投影矩阵 [D, D_proj]
        Tensor projMat = torch.randn(new long[]{dim, projectionDim}).to(x.device(),torch.ScalarType.Float);

        // 投影
        Tensor xProj = x.matmul(projMat);
        Tensor yProj = (y == null) ? xProj : y.matmul(projMat);

        // 在投影空间搜索
        return super.search(xProj, yProj, batchX, batchY);
    }
}

