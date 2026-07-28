package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * GCNII / GCN2Conv (Chen et al., ICML 2020) — deep GCN with initial residual
 * and identity mapping.
 *
 * <pre>
 *   X' = ((1-α) Ã X + α X^{(0)}) · ((1-β) I + β W)
 * </pre>
 * When {@code sharedWeights=false} (GCNII*), separate transforms are applied
 * to the aggregated path and the initial residual path.
 *
 * <p>Primary API: {@link #forward(Tensor, Tensor, Tensor, Tensor)} with
 * {@code (x, x0, edge_index, edge_weight)}. The 2-arg form uses {@code x0 = x}.
 */
public class GCN2Conv extends MessagePassing {

    private final LinearImpl lin;
    private final LinearImpl linRes;
    private final float alpha;
    private final float beta;
    private final boolean sharedWeights;
    private final boolean normalize;
    private final boolean addSelfLoops;
    private final long channels;

    public GCN2Conv(long channels, float alpha, Float theta, Integer layer,
                    boolean sharedWeights, boolean normalize) {
        this(channels, alpha, theta, layer, sharedWeights, normalize, true);
    }

    public GCN2Conv(long channels, float alpha, Float theta, Integer layer,
                    boolean sharedWeights, boolean normalize, boolean addSelfLoops) {
        super("sum");
        if (channels <= 0) {
            throw new IllegalArgumentException("channels must be > 0");
        }
        this.channels = channels;
        this.alpha = alpha;
        this.sharedWeights = sharedWeights;
        this.normalize = normalize;
        this.addSelfLoops = addSelfLoops;

        if (theta != null && layer != null && layer > 0) {
            this.beta = (float) Math.log(theta / layer + 1.0);
        } else {
            this.beta = 0.1f;
        }

        this.lin = register_module("lin", new LinearImpl(channels, channels));
        if (!sharedWeights) {
            this.linRes = register_module("lin_res", new LinearImpl(channels, channels));
        } else {
            this.linRes = null;
        }
    }

    /** Convenience: treats current features as initial features. */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, x, edge_index, null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        return forward(x, x, edge_index, edge_weight);
    }

    /**
     * @param x           current-layer features [N, C]
     * @param x0          initial features X^{(0)} [N, C]
     * @param edge_index  [2, E]
     * @param edge_weight optional [E]
     */
    public Tensor forward(Tensor x, Tensor x0, Tensor edge_index, Tensor edge_weight) {
        if (x == null || x0 == null || edge_index == null) {
            throw new NullPointerException("x, x0, edge_index must not be null");
        }
        if (x.size(1) != channels || x0.size(1) != channels) {
            throw new IllegalArgumentException(
                    "feature dim must equal channels=" + channels);
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

        // Ã X
        Tensor out = propagate(ei, x, ew);

        // (1-α) ÃX + α X0
        out = out.mul(new Scalar(1.0 - alpha)).add(x0.mul(new Scalar(alpha)));

        // Identity mapping: (1-β)I + β W
        if (sharedWeights || linRes == null) {
            out = out.mul(new Scalar(1.0 - beta)).add(lin.forward(out).mul(new Scalar(beta)));
        } else {
            // GCNII* variant
            Tensor out1 = lin.forward(out);
            Tensor out2 = linRes.forward(x0);
            out = out1.add(out2).mul(new Scalar(beta)).add(out.mul(new Scalar(1.0 - beta)));
        }
        return out;
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

    public float getAlpha() {
        return alpha;
    }

    public float getBeta() {
        return beta;
    }

    public long getChannels() {
        return channels;
    }
}
