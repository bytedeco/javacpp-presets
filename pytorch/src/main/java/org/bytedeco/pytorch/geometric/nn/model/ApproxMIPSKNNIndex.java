package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * Approx MIPS KNN
 * 同理，使用随机投影近似。
 */
public class ApproxMIPSKNNIndex extends MIPSKNNIndex {
    private long projectionDim;

    public ApproxMIPSKNNIndex(long k, long projectionDim) {
        super(k);
        this.projectionDim = projectionDim;
    }

    @Override
    public Tensor[] search(Tensor x, Tensor y, Tensor batchX, Tensor batchY) {
        long dim = x.size(1);
        Tensor projMat = torch.randn(new long[]{dim, projectionDim});
        projMat.to(x.device(), torch.ScalarType.Float);

        Tensor xProj = x.matmul(projMat);
        Tensor yProj = (y == null) ? xProj : y.matmul(projMat);

        return super.search(xProj, yProj, batchX, batchY);
    }
}