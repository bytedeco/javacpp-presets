package org.bytedeco.pytorch.geometric.nn.norm;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * Feature-wise LayerNorm over the last dimension (PyG / Transformer style).
 *
 * <pre>
 *   y = (x − μ) / √(σ² + ε)  ⊙ γ + β
 * </pre>
 * Works for any rank ≥ 1; normalization is always over the last axis of size
 * {@code inChannels}. Affine γ, β are optional learnable vectors of length C.
 */
public class LayerNorm extends Module {

    private final long inChannels;
    private final double eps;
    private final boolean affine;
    private final Parameter weight; // γ
    private final Parameter bias;   // β

    public LayerNorm(long inChannels) {
        this(inChannels, 1e-5, true);
    }

    public LayerNorm(long inChannels, double eps, boolean affine) {
        super();
        if (inChannels <= 0) {
            throw new IllegalArgumentException("inChannels must be > 0");
        }
        this.inChannels = inChannels;
        this.eps = eps;
        this.affine = affine;

        if (affine) {
            TensorOptions fOpt = new TensorOptions()
                    .dtype(new ScalarTypeOptional(torch.ScalarType.Float));
            Tensor w = torch.ones(new long[]{inChannels}, fOpt).clone().requires_grad_(true);
            Tensor b = torch.zeros(new long[]{inChannels}, fOpt).clone().requires_grad_(true);
            this.weight = new Parameter(w, true);
            this.bias = new Parameter(b, true);
            register_parameter("weight", this.weight);
            register_parameter("bias", this.bias);
        } else {
            this.weight = null;
            this.bias = null;
        }
    }

    /**
     * @param x arbitrary rank with last dim = inChannels
     * @return same shape as x
     */
    public Tensor forward(Tensor x) {
        if (x == null) {
            throw new NullPointerException("x must not be null");
        }
        x = x.contiguous();
        long last = x.dim() - 1;
        if (x.size(last) != inChannels) {
            throw new IllegalArgumentException(
                    "last dim " + x.size(last) + " != inChannels " + inChannels);
        }

        // var(dim, unbiased, keepdim) — keep last dim for broadcast
        long[] reduceDims = {last};
        Tensor mean = x.mean(reduceDims, true, new ScalarTypeOptional());
        Tensor var = x.var(reduceDims, /*unbiased=*/false, /*keepdim=*/true);
        Tensor out = x.sub(mean).div(var.add(new Scalar(eps)).sqrt());

        if (affine) {
            // Broadcast γ, β over leading dims: [C] → [1,...,1,C]
            Tensor w = weight;
            Tensor b = bias;
            for (int i = 0; i < x.dim() - 1; i++) {
                w = w.unsqueeze(0);
                b = b.unsqueeze(0);
            }
            out = out.mul(w).add(b);
        }
        return out;
    }

    public long getInChannels() {
        return inChannels;
    }

    public double getEps() {
        return eps;
    }

    public boolean isAffine() {
        return affine;
    }
}
