package org.bytedeco.pytorch.geometric.transforms;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.data.GraphData;

import static org.bytedeco.pytorch.global.torch.kFloat;
import static org.bytedeco.pytorch.global.torch.linalg_eigh;

/**
 * NormalizeRotation: center point cloud and rotate onto principal axes (PCA).
 *
 * <p>Uses symmetric eigendecomposition {@code linalg_eigh} of the covariance
 * (real eigenvectors). Divides by {@code N-1} via {@link Scalar}, not
 * {@code new Tensor(long)} which yields an undefined tensor.
 */
public class NormalizeRotation implements BaseTransform {
    @Override
    public GraphData apply(GraphData data) {
        Tensor pos = TransformUtils.requirePos(data);
        long n = pos.size(0);
        if (n < 2) {
            // nothing to normalize
            return data;
        }

        // 1. Center
        pos = pos.sub(pos.mean(new long[]{0}, true, new ScalarTypeOptional(kFloat())));

        // 2. Covariance C = X^T X / (N-1)
        Tensor cov = pos.t().mm(pos).div(new Scalar(n - 1));

        // 3. Symmetric eigendecomposition — eigenvalues ascending, V columns = eigenvectors
        // linalg_eigh is appropriate for real symmetric cov (avoids complex from linalg_eig).
        T_TensorTensor_T eig = linalg_eigh(cov);
        Tensor v = eig.get1().to(kFloat()); // [D, D]

        // 4. Rotate into eigenbasis (largest variance along last axis if ascending)
        data.pos = pos.mm(v);
        return data;
    }
}
