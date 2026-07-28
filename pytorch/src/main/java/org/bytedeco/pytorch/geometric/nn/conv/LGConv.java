package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;

/**
 * LightGCN convolution (He et al., SIGIR 2020) — pure neighborhood aggregation
 * without feature transformation or nonlinearities.
 *
 * <pre>
 *   x'_i = Σ_{j ∈ N(i)} (d_i d_j)^{-1/2} x_j     (when normalize=true)
 * </pre>
 */
public class LGConv extends MessagePassing {

    private final boolean normalize;
    private final boolean addSelfLoops;

    public LGConv() {
        this(true, false);
    }

    public LGConv(boolean normalize) {
        this(normalize, false);
    }

    public LGConv(boolean normalize, boolean addSelfLoops) {
        super("sum");
        this.normalize = normalize;
        this.addSelfLoops = addSelfLoops;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        long N = x.size(0);
        Tensor ei = edge_index;
        Tensor ew = edge_weight;

        if (normalize) {
            torch.ScalarType dtype = x.scalar_type().intern();
            // LightGCN typically does NOT add self-loops; honor flag if set
            Tensor[] normed = GraphUtils.gcn_norm(ei, ew, N, addSelfLoops, dtype);
            ei = normed[0];
            ew = normed[1];
        } else if (addSelfLoops) {
            if (ew == null) {
                ei = GraphUtils.add_self_loops(ei, N);
            } else {
                Tensor[] pair = GraphUtils.add_self_loops(ei, ew, N, 1.0);
                ei = pair[0];
                ew = pair[1];
            }
        }

        return propagate(ei, x, ew);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (edge_attr != null) {
            if (edge_attr.dim() == 1) {
                return x_j.mul(edge_attr.view(new long[]{-1, 1}));
            }
            return x_j.mul(edge_attr);
        }
        return x_j;
    }

    public boolean isNormalize() {
        return normalize;
    }
}
