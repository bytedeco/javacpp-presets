package org.bytedeco.pytorch.geometric.utils;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

/**
 * Shared kernels for linear / Performer-style attention.
 */
public final class AttentionUtils {

    private AttentionUtils() {}

    /** Polynormer / linear-attention kernel: {@code φ(x) = elu(x) + 1} (non-negative). */
    public static Tensor kernel_elu(Tensor x) {
        return torch.elu(x).add(new Scalar(1.0));
    }

    /**
     * Build a random feature projection matrix of shape {@code [dim, numFeatures]} (D × M).
     * When {@code orthogonal=true}, uses block-wise QR (FAVOR+ style ORF).
     */
    public static Tensor create_projection_matrix(long numFeatures, long dim, boolean orthogonal) {
        if (numFeatures <= 0 || dim <= 0) {
            throw new IllegalArgumentException("numFeatures and dim must be > 0");
        }
        if (orthogonal) {
            long numBlocks = (long) Math.ceil((double) numFeatures / dim);
            TensorVector blocks = new TensorVector();
            for (long i = 0; i < numBlocks; i++) {
                Tensor mat = torch.randn(new long[]{dim, dim});
                T_TensorTensor_T qr = torch.linalg_qr(mat);
                // push_back grows the vector; put(Tensor) alone does not reliably append
                blocks.push_back(qr.get0()); // Q [dim, dim]
            }
            // Stack blocks along rows → [numBlocks*dim, dim], take first numFeatures rows,
            // transpose → [dim, M]
            Tensor full = torch.cat(blocks, 0); // [numBlocks*dim, dim]
            Tensor rows = full.slice(0, new LongOptional(0), new LongOptional(numFeatures), 1L); // [M, dim]
            return rows.t().contiguous(); // [dim, M]
        }
        // Non-orthogonal: plain Gaussian [dim, M]
        return torch.randn(new long[]{dim, numFeatures}).contiguous();
    }

    /**
     * Performer positive feature map.
     *
     * @param data              [N, D]
     * @param projectionMatrix  [D, M]
     * @param isQuery           unused flag kept for API parity with FAVOR+ variants
     * @return [N, M]
     */
    public static Tensor kernel_performer(Tensor data, Tensor projectionMatrix, boolean isQuery) {
        if (data.dim() != 2) {
            throw new IllegalArgumentException("data must be [N, D]");
        }
        if (projectionMatrix.dim() != 2) {
            throw new IllegalArgumentException("projectionMatrix must be [D, M]");
        }
        // Accept either [D,M] or [M,D] and normalize to [D,M]
        Tensor projMat = projectionMatrix;
        if (projMat.size(0) != data.size(1) && projMat.size(1) == data.size(1)) {
            projMat = projMat.t().contiguous();
        }
        if (projMat.size(0) != data.size(1)) {
            throw new IllegalArgumentException(
                    "projectionMatrix rows must equal data.size(1) (got "
                            + projMat.size(0) + " vs " + data.size(1) + ")");
        }
        long M = projMat.size(1);

        // ‖x‖² / 2  → [N, 1]
        Tensor dataSq = data.pow(new Scalar(2.0))
                .sum(new long[]{-1}, true, new ScalarTypeOptional())
                .div(new Scalar(2.0));

        // scale = M^{-1/4}
        double scale = Math.pow(M, -0.25);
        Tensor proj = data.matmul(projMat).mul(new Scalar(scale)); // [N, M]
        return proj.sub(dataSq).exp();
    }
}
