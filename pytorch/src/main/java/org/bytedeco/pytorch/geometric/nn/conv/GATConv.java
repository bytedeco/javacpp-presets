package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Graph Attention Network convolution (Veličković et al.).
 *
 * <pre>
 *   α_{ij} = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )
 *   h'_i   = σ( Σ_j α_{ij} W h_j )
 * </pre>
 *
 * Multi-head: features are shaped [N, heads, outChannels] during propagate.
 * Requires destination features ({@link #needsX_i()}) for attention logits.
 */
public class GATConv extends MessagePassing {

    private final LinearImpl lin;
    private final Tensor att;          // [1, heads, 2 * outChannels]
    private final long heads;
    private final long outChannels;
    private final double negativeSlope;
    private final boolean concat;
    private final boolean addSelfLoops;

    public GATConv(long inChannels, long outChannels, long heads, double negativeSlope) {
        this(inChannels, outChannels, heads, true, negativeSlope, true);
    }

    public GATConv(long inChannels, long outChannels, long heads,
                   boolean concat, double negativeSlope) {
        this(inChannels, outChannels, heads, concat, negativeSlope, true);
    }

    public GATConv(long inChannels, long outChannels, long heads,
                   boolean concat, double negativeSlope, boolean addSelfLoops) {
        super("sum");
        this.heads = heads;
        this.outChannels = outChannels;
        this.negativeSlope = negativeSlope;
        this.concat = concat;
        this.addSelfLoops = addSelfLoops;

        this.lin = register_module("lin", new LinearImpl(inChannels, heads * outChannels));

        // Attention vector a — keep a retained leaf handle (ByRef register_parameter)
        Tensor attInit = torch.randn(new long[]{1, heads, 2 * outChannels});
        torch.xavier_uniform_(attInit);
        // Clone so register_parameter storage and our field stay valid independently
        this.att = attInit.clone();
        register_parameter("att", this.att);
    }

    @Override
    protected boolean needsX_i() {
        return true;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        long N = x.size(0);
        Tensor ei = edge_index;
        if (addSelfLoops) {
            ei = org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops(ei, N);
        }

        // [N, heads * out] → [N, heads, out]
        Tensor xLin = lin.forward(x).view(N, heads, outChannels);

        // Full pipeline lifts x_i and x_j; attention runs in message(...)
        Tensor out = propagate(ei, xLin);

        if (concat) {
            out = out.view(N, heads * outChannels);
        } else {
            out = out.mean(1);
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index,
                          Tensor edge_attr, long numNodes) {
        // x_i, x_j: [E, heads, outChannels]
        Tensor targetIdx = _index_i != null ? _index_i : edge_index.select(0, 1);
        targetIdx = AggrUtils.asLongIndex(targetIdx);

        // e_ij = a^T [Wh_i || Wh_j]
        Tensor catFeat = torch.cat(new TensorVector(x_i, x_j), -1); // [E, H, 2*C]
        Tensor alpha = catFeat.mul(this.att).sum(-1);                 // [E, H]
        alpha = torch.leaky_relu(alpha, new Scalar(negativeSlope));

        // Segment softmax over target nodes
        long n = numNodes > 0 ? numNodes : (_size != null ? _size[1] : 0);
        alpha = AggrUtils.scatter_softmax(alpha, targetIdx, n);

        // α ⊙ x_j
        return x_j.mul(alpha.unsqueeze(-1));
    }

    @Override
    public Tensor update(Tensor inputs, Tensor x) {
        // Inputs already [N, H, C] from aggregate; concat/mean done in forward
        return inputs;
    }

    public LinearImpl getLin() {
        return lin;
    }

    public Tensor getAtt() {
        return att;
    }

    public long getHeads() {
        return heads;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public boolean isConcat() {
        return concat;
    }
}
