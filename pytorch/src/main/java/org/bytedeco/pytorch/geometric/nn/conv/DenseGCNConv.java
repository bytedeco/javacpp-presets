package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Dense (batched) GCN convolution for adjacency tensors.
 *
 * <pre>
 *   Â = A + I   (or A + 2I if improved)
 *   D̃_{ii} = Σ_j Â_{ij}
 *   X' = D̃^{-1/2} Â D̃^{-1/2} X Θ
 * </pre>
 * Inputs are batched dense tensors {@code x [B,N,F]}, {@code adj [B,N,N]}.
 * Extends {@link MessagePassing} for API uniformity but does <b>not</b> use
 * sparse edge_index propagate (PyG DenseGCNConv is a plain Module).
 */
public class DenseGCNConv extends MessagePassing {

    private final LinearImpl lin;
    private final boolean improved;
    private final long inChannels;
    private final long outChannels;

    public DenseGCNConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, false);
    }

    public DenseGCNConv(long inChannels, long outChannels, boolean improved) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.improved = improved;
        this.lin = register_module("lin", new LinearImpl(inChannels, outChannels));
    }

    /**
     * Dense forward. Second arg is adjacency {@code [B,N,N]} (not sparse edge_index).
     * Java cannot overload two (Tensor,Tensor) methods, so this is the sole 2-arg entry.
     */
    @Override
    public Tensor forward(Tensor x, Tensor adj) {
        return forward(x, adj, new TensorOptional());
    }

    /**
     * @param x    [B, N, inChannels]
     * @param adj  [B, N, N] (0/1 or weighted)
     * @param mask optional [B, N] node mask (padding)
     */
    public Tensor forward(Tensor x, Tensor adj, TensorOptional mask) {
        if (x == null || adj == null) {
            throw new NullPointerException("x and adj must not be null");
        }
        if (x.dim() != 3) {
            throw new IllegalArgumentException("x must be [B,N,F], dim=" + x.dim());
        }
        if (adj.dim() != 3) {
            throw new IllegalArgumentException("adj must be [B,N,N], dim=" + adj.dim());
        }
        if (x.size(2) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(2)=" + x.size(2) + " != inChannels=" + inChannels);
        }
        long B = x.size(0);
        long N = x.size(1);
        if (adj.size(0) != B || adj.size(1) != N || adj.size(2) != N) {
            throw new IllegalArgumentException("adj shape must be [B,N,N] matching x");
        }

        // Â = A + I  (or A + 2I)
        Tensor eye = torch.eye(N, x.options()).unsqueeze(0).expand(new long[]{B, N, N});
        Tensor hatA = adj.add(eye);
        if (improved) {
            hatA = hatA.add(eye);
        }

        // D̃^{-1/2}
        Tensor deg = hatA.sum(new long[]{2}, false, new ScalarTypeOptional());
        Tensor degInvSqrt = deg.pow(new Scalar(-0.5));
        degInvSqrt = degInvSqrt.masked_fill(degInvSqrt.isinf(), new Scalar(0));
        degInvSqrt = degInvSqrt.masked_fill(degInvSqrt.isnan(), new Scalar(0));

        // D̃^{-1/2} Â D̃^{-1/2}
        Tensor normAdj = hatA.mul(degInvSqrt.unsqueeze(2)).mul(degInvSqrt.unsqueeze(1));

        Tensor xW = lin.forward(x);          // [B,N,out]
        Tensor out = normAdj.matmul(xW);     // [B,N,out]

        if (mask != null && mask.has_value()) {
            Tensor m = mask.get();
            if (m.dim() != 2 || m.size(0) != B || m.size(1) != N) {
                throw new IllegalArgumentException("mask must be [B,N]");
            }
            out = out.mul(m.unsqueeze(2).to(out.dtype()));
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // Dense path does not use sparse message passing
        return x_j;
    }

    public LinearImpl getLin() {
        return lin;
    }

    public boolean isImproved() {
        return improved;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
