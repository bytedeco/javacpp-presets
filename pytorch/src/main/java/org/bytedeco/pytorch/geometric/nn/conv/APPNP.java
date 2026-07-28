package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;

/**
 * APPNP: Approximate Personalized Propagation of Neural Predictions
 * (Klicpera et al., ICLR 2019).
 *
 * <pre>
 *   Z^{(0)} = H
 *   Z^{(k+1)} = (1-α) Ã_sym Z^{(k)} + α H
 *   Z = Z^{(K)}
 * </pre>
 * Topic-sensitive PageRank smoothing of (usually MLP) predictions {@code H}.
 * This layer itself has no trainable weights.
 */
public class APPNP extends MessagePassing {

    private final int K;
    private final double alpha;
    private final double dropout;
    private final boolean addSelfLoops;
    private final boolean normalize;

    public APPNP(int K, double alpha) {
        this(K, alpha, 0.0, true, true);
    }

    public APPNP(int K, double alpha, double dropout, boolean addSelfLoops, boolean normalize) {
        super("sum");
        if (K < 1) {
            throw new IllegalArgumentException("K must be >= 1");
        }
        if (alpha < 0.0 || alpha > 1.0) {
            throw new IllegalArgumentException("alpha must be in [0,1]");
        }
        this.K = K;
        this.alpha = alpha;
        this.dropout = dropout;
        this.addSelfLoops = addSelfLoops;
        this.normalize = normalize;
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

        // Optional edge dropout on normalized weights (train-time)
        if (dropout > 0.0 && this.is_training()) {
            if (ew == null) {
                ew = torch.ones(new long[]{ei.size(1)}, x.options());
            }
            ew = torch.dropout(ew, dropout, /*train=*/true);
        }

        Tensor x0 = x;
        Tensor xk = x;
        for (int k = 0; k < K; k++) {
            Tensor aggr = propagate(ei, xk, ew);
            xk = aggr.mul(new Scalar(1.0 - alpha)).add(x0.mul(new Scalar(alpha)));
        }
        return xk;
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

    public int getK() {
        return K;
    }

    public double getAlpha() {
        return alpha;
    }
}
