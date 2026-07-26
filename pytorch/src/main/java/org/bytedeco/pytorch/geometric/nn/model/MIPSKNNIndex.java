package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

/**
 * Exact MIPS KNN (Maximum Inner Product Search)
 * 使用 Dot Product
 */
public class MIPSKNNIndex extends KNNIndex {
    public MIPSKNNIndex(long k) { super(k); }

    @Override
    public Tensor[] search(Tensor x, Tensor y, Tensor batchX, Tensor batchY) {
        Tensor target = (y == null) ? x : y;

        // Dot Product: X @ Y^T
        Tensor scores = x.matmul(target.t());

        // Batch Masking
        if (batchX != null && batchY != null) {
            Tensor mask = batchX.unsqueeze(1).ne(batchY.unsqueeze(0));
            // MIPS 找最大值，所以 mask 设为负无穷
            scores.masked_fill_(mask, new Scalar(Float.NEGATIVE_INFINITY));
        }

        // TopK (largest scores)
        T_TensorTensor_T ret = torch.topk(scores, k, 1, true, true);
        return new Tensor[]{ret.get0(), ret.get1()};
    }
}