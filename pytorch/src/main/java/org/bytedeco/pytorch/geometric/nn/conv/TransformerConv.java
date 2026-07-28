package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Graph Transformer convolution (Shi et al. / PyG {@code TransformerConv}).
 *
 * <pre>
 *   α_{ij} = softmax_j( (Q_i · K_j) / √d )
 *   h'_i   = Σ_j α_{ij} V_j  (+ residual W_s h_i)
 * </pre>
 * Multi-head; output is concatenated heads by default.
 */
public class TransformerConv extends MessagePassing {

    private final LinearImpl linQuery;
    private final LinearImpl linKey;
    private final LinearImpl linValue;
    private final LinearImpl linSkip;
    private final long heads;
    private final long outChannels;
    private final boolean concat;
    private final boolean rootWeight;
    private final double beta; // residual gate (0 = plain residual)

    // Transient multi-head Q/K during one propagate (not closed)
    private Tensor _q;
    private Tensor _k;

    public TransformerConv(long inChannels, long outChannels, long heads) {
        this(inChannels, outChannels, heads, true, true, 0.0);
    }

    public TransformerConv(long inChannels, long outChannels, long heads,
                           boolean concat, boolean rootWeight, double beta) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || heads < 1) {
            throw new IllegalArgumentException("invalid TransformerConv dims");
        }
        this.heads = heads;
        this.outChannels = outChannels;
        this.concat = concat;
        this.rootWeight = rootWeight;
        this.beta = beta;

        this.linQuery = register_module("lin_query", new LinearImpl(inChannels, heads * outChannels));
        this.linKey = register_module("lin_key", new LinearImpl(inChannels, heads * outChannels));
        this.linValue = register_module("lin_value", new LinearImpl(inChannels, heads * outChannels));
        this.linSkip = rootWeight
                ? register_module("lin_skip", new LinearImpl(inChannels,
                concat ? heads * outChannels : outChannels))
                : null;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        long N = x.size(0);

        // Precompute Q, K, V on nodes [N, H, C] — industrial efficiency
        this._q = linQuery.forward(x).view(N, heads, outChannels);
        this._k = linKey.forward(x).view(N, heads, outChannels);
        Tensor v = linValue.forward(x).view(N, heads, outChannels);

        Tensor out;
        try {
            // Propagate values; attention reads Q/K via transient state + indices
            out = propagate(edge_index, v, edge_attr);
        } finally {
            this._q = null;
            this._k = null;
        }

        if (concat) {
            out = out.view(N, heads * outChannels);
        } else {
            out = out.mean(1);
        }

        if (rootWeight && linSkip != null) {
            Tensor skip = linSkip.forward(x);
            if (beta != 0.0) {
                // gated residual: (1-β)·skip + β·out  (simplified)
                out = skip.mul(new Scalar(1.0 - beta)).add(out.mul(new Scalar(beta)));
            } else {
                out = out.add(skip);
            }
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j is V_j [E, H, C]
        Tensor index_i = _index_i != null ? _index_i : edge_index.select(0, 1);
        Tensor index_j = _index_j != null ? _index_j : edge_index.select(0, 0);
        index_i = AggrUtils.asLongIndex(index_i);
        index_j = AggrUtils.asLongIndex(index_j);

        Tensor q_i = _q.index_select(0, index_i); // [E, H, C]
        Tensor k_j = _k.index_select(0, index_j);

        // α = (Q·K)/√d
        Tensor alpha = q_i.mul(k_j).sum(-1).div(new Scalar(Math.sqrt(outChannels)));

        // Optional edge bias: if edge_attr provided as [E] or [E,H], add to logits
        if (edge_attr != null) {
            if (edge_attr.dim() == 1) {
                alpha = alpha.add(edge_attr.unsqueeze(-1));
            } else if (edge_attr.dim() == 2 && edge_attr.size(1) == heads) {
                alpha = alpha.add(edge_attr);
            }
        }

        long n = numNodes > 0 ? numNodes : (_size != null ? _size[1] : 0);
        alpha = AggrUtils.scatter_softmax(alpha, index_i, n);
        return x_j.mul(alpha.unsqueeze(-1));
    }

    public long getHeads() {
        return heads;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
