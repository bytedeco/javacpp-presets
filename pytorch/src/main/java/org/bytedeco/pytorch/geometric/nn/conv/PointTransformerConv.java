package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * Point Transformer convolution (Zhao et al. / PyG {@code PointTransformerConv}).
 *
 * <pre>
 *   δ_{ij} = PosNN(p_i - p_j)
 *   α_{ij} = softmax_j( AttnNN(q_i - k_j + δ_{ij}) )
 *   y_i    = Σ_j α_{ij} ⊙ (v_j + δ_{ij})
 * </pre>
 * Uses industrial {@link MessagePassing} (sum aggregation) with transient Q/K/V/pos.
 * No intermediate Tensor.close(); Module parameters stay registered for optimizers.
 */
public class PointTransformerConv extends MessagePassing {

    private final LinearImpl linQ;
    private final LinearImpl linK;
    private final LinearImpl linV;
    private final Module posNN;
    private final Module attnNN;
    private final long inChannels;
    private final long outChannels;
    private final long posDim;

    // Transient state for one forward
    private Tensor _q;
    private Tensor _k;
    private Tensor _v;
    private Tensor _pos;

    public PointTransformerConv(long inChannels, long outChannels, long posDim,
                                Module posNN, Module attnNN) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || posDim <= 0) {
            throw new IllegalArgumentException("PointTransformerConv dims must be > 0");
        }
        if (posNN == null || attnNN == null) {
            throw new IllegalArgumentException("posNN and attnNN must not be null");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.posDim = posDim;
        this.linQ = register_module("lin_q", new LinearImpl(inChannels, outChannels));
        this.linK = register_module("lin_k", new LinearImpl(inChannels, outChannels));
        this.linV = register_module("lin_v", new LinearImpl(inChannels, outChannels));
        this.posNN = register_module("pos_nn", posNN);
        this.attnNN = register_module("attn_nn", attnNN);
    }

    /** Convenience when posDim is known to be 3 (point clouds). */
    public PointTransformerConv(long inChannels, long outChannels,
                                Module posNN, Module attnNN) {
        this(inChannels, outChannels, 3, posNN, attnNN);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        // Fallback: use first posDim feature channels as coordinates
        if (x.size(1) < posDim) {
            throw new IllegalArgumentException(
                    "forward(x, edge_index) needs x.size(1) >= posDim=" + posDim
                            + " to derive coordinates, or pass pos explicitly");
        }
        Tensor pos = x.narrow(1, 0, posDim);
        return forward(x, pos, edge_index);
    }

    /**
     * @param x          node features [N, inChannels]
     * @param pos        coordinates [N, posDim]
     * @param edge_index [2, E]
     */
    public Tensor forward(Tensor x, Tensor pos, Tensor edge_index) {
        if (x == null || pos == null || edge_index == null) {
            throw new NullPointerException("x, pos, edge_index must not be null");
        }
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException("x must be [N," + inChannels + "]");
        }
        if (pos.dim() != 2 || pos.size(1) != posDim) {
            throw new IllegalArgumentException("pos must be [N," + posDim + "]");
        }
        if (pos.size(0) != x.size(0)) {
            throw new IllegalArgumentException("pos and x must share N");
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }

        this._q = linQ.forward(x);
        this._k = linK.forward(x);
        this._v = linV.forward(x);
        this._pos = pos;
        try {
            // Lift V as the message feature carrier; Q/K/pos read via indices in message
            return propagate(edge_index, this._v);
        } finally {
            this._q = null;
            this._k = null;
            this._v = null;
            this._pos = null;
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j is V_j [E, out]
        if (_q == null || _k == null || _pos == null) {
            throw new IllegalStateException("PointTransformerConv message requires active forward");
        }
        Tensor index_j = AggrUtils.asLongIndex(
                _index_j != null ? _index_j : edge_index.select(0, 0));
        Tensor index_i = AggrUtils.asLongIndex(
                _index_i != null ? _index_i : edge_index.select(0, 1));

        Tensor q_i = _q.index_select(0, index_i);
        Tensor k_j = _k.index_select(0, index_j);
        Tensor v_j = x_j; // already V_j
        Tensor pos_i = _pos.index_select(0, index_i);
        Tensor pos_j = _pos.index_select(0, index_j);

        // Relative position encoding δ = PosNN(p_i - p_j)
        Tensor rel = pos_i.sub(pos_j);
        Tensor delta = forwardMlp(posNN, rel); // [E, out]
        if (delta.size(1) != outChannels) {
            throw new IllegalStateException(
                    "posNN must output outChannels=" + outChannels + ", got " + delta.size(1));
        }

        // Attention logits: AttnNN(q_i - k_j + δ)
        Tensor attnInput = q_i.sub(k_j).add(delta);
        Tensor attn = forwardMlp(attnNN, attnInput); // [E, out] or [E,1]
        // Segment softmax over target nodes (feature-wise if multi-dim)
        long n = numNodes > 0 ? numNodes : (_size != null ? _size[1] : 0);
        attn = AggrUtils.scatter_softmax(attn, index_i, n);

        // α ⊙ (v_j + δ)
        return attn.mul(v_j.add(delta));
    }

    private static Tensor forwardMlp(Module m, Tensor in) {
        if (m instanceof SequentialImpl) {
            return ((SequentialImpl) m).forward(in);
        }
        if (m instanceof LinearImpl) {
            return ((LinearImpl) m).forward(in);
        }
        return m.asSequential().forward(in);
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public long getPosDim() {
        return posDim;
    }
}
