package org.bytedeco.pytorch.geometric.attention;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AttentionUtils;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Performer attention (FAVOR+, Choromanski et al.) — linear-complexity attention
 * via positive orthogonal random features.
 *
 * <pre>
 *   φ(x) = exp(x Ω − ‖x‖²/2)           // random feature map
 *   y    = φ(Q) (φ(K)ᵀ V) / (φ(Q) (φ(K)ᵀ 1))
 * </pre>
 * Complexity O(N · D · M) with M = {@code numFeatures}.
 */
public class PerformerAttention extends Module {

    private final LinearImpl linQ;
    private final LinearImpl linK;
    private final LinearImpl linV;
    private final LinearImpl linOut;
    private final Tensor projectionMatrix; // [D, M] buffer
    private final long numHeads;
    private final long headDim;
    private final long numFeatures;
    private final long inChannels;

    public PerformerAttention(long inChannels, long numHeads, long numFeatures) {
        super();
        if (inChannels <= 0 || numHeads <= 0 || numFeatures <= 0) {
            throw new IllegalArgumentException("inChannels/numHeads/numFeatures must be > 0");
        }
        if (inChannels % numHeads != 0) {
            throw new IllegalArgumentException("inChannels must be divisible by numHeads");
        }
        this.inChannels = inChannels;
        this.numHeads = numHeads;
        this.headDim = inChannels / numHeads;
        this.numFeatures = numFeatures;

        this.linQ = register_module("linQ", new LinearImpl(inChannels, inChannels));
        this.linK = register_module("linK", new LinearImpl(inChannels, inChannels));
        this.linV = register_module("linV", new LinearImpl(inChannels, inChannels));
        this.linOut = register_module("linOut", new LinearImpl(inChannels, inChannels));

        // Orthogonal random features [headDim, M] as non-trainable buffer
        Tensor proj = AttentionUtils.create_projection_matrix(numFeatures, headDim, true).contiguous();
        this.projectionMatrix = proj;
        register_buffer("projectionMatrix", this.projectionMatrix);
    }

    /**
     * @param x [N, C] node/token features
     * @return [N, C]
     */
    public Tensor forward(Tensor x) {
        if (x == null || x.dim() != 2) {
            throw new IllegalArgumentException("x must be [N, C]");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }
        long N = x.size(0);
        long C = inChannels;

        Tensor q = linQ.forward(x).view(N, numHeads, headDim);
        Tensor k = linK.forward(x).view(N, numHeads, headDim);
        Tensor v = linV.forward(x).view(N, numHeads, headDim);

        // Kernel feature maps shared across heads: [N*H, D] → [N*H, M]
        // Use reshape (not view) — kernel output may be non-contiguous after exp.
        Tensor qPrime = AttentionUtils.kernel_performer(
                q.reshape(N * numHeads, headDim), projectionMatrix, true)
                .reshape(N, numHeads, numFeatures);
        Tensor kPrime = AttentionUtils.kernel_performer(
                k.reshape(N * numHeads, headDim), projectionMatrix, false)
                .reshape(N, numHeads, numFeatures);

        // φ(K)ᵀ V : [H, M, N] @ [H, N, D] → [H, M, D]
        Tensor kT = kPrime.permute(1, 2, 0);   // [H, M, N]
        Tensor vP = v.permute(1, 0, 2);        // [H, N, D]
        Tensor kv = kT.matmul(vP);             // [H, M, D]

        // Normalization: φ(K)ᵀ 1 → [H, M, 1]
        Tensor kSum = kT.sum(new long[]{2}, true, new ScalarTypeOptional());
        Tensor qP = qPrime.permute(1, 0, 2);   // [H, N, M]
        Tensor norm = qP.matmul(kSum).add(new Scalar(1e-6)); // [H, N, 1]

        // Numerator: φ(Q) (φ(K)ᵀ V)
        Tensor out = qP.matmul(kv).div(norm);  // [H, N, D]
        out = out.permute(1, 0, 2).reshape(N, C);
        return linOut.forward(out);
    }

    public long getNumHeads() {
        return numHeads;
    }

    public long getNumFeatures() {
        return numFeatures;
    }

    public long getInChannels() {
        return inChannels;
    }
}
