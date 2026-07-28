package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.AggrUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * GATv2 convolution (Brody et al., ICLR 2022) — dynamic attention.
 *
 * <pre>
 *   e_{ij} = a^T LeakyReLU( W_l h_i + W_r h_j [+ W_e e_{ij}] )
 *   α_{ij} = softmax_j(e_{ij})
 *   h'_i   = σ( Σ_j α_{ij} W_r h_j )
 * </pre>
 * Independent source/destination projections fix static attention of classic GAT.
 */
public class GATv2Conv extends MessagePassing {

    private final LinearImpl linSrc;
    private final LinearImpl linDst;
    private final LinearImpl linEdge;
    private final Tensor att; // [1, heads, outChannels] — retained leaf
    private final Parameter bias;
    private final long heads;
    private final long outChannels;
    private final boolean concat;
    private final double negativeSlope;
    private final boolean addSelfLoops;

    public GATv2Conv(long inChannels, long outChannels, long heads) {
        this(inChannels, outChannels, heads, true, 0.2, null, true, true);
    }

    public GATv2Conv(long inChannels, long outChannels, long heads, boolean concat,
                     double negativeSlope, Integer edgeDim, boolean hasBias) {
        this(inChannels, outChannels, heads, concat, negativeSlope, edgeDim, hasBias, true);
    }

    public GATv2Conv(long inChannels, long outChannels, long heads, boolean concat,
                     double negativeSlope, Integer edgeDim, boolean hasBias, boolean addSelfLoops) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || heads < 1) {
            throw new IllegalArgumentException("invalid GATv2 dimensions");
        }
        this.heads = heads;
        this.outChannels = outChannels;
        this.concat = concat;
        this.negativeSlope = negativeSlope;
        this.addSelfLoops = addSelfLoops;

        this.linSrc = register_module("lin_src", new LinearImpl(inChannels, heads * outChannels));
        this.linDst = register_module("lin_dst", new LinearImpl(inChannels, heads * outChannels));

        Tensor attInit = torch.randn(new long[]{1, heads, outChannels});
        torch.xavier_uniform_(attInit);
        this.att = attInit.clone();
        register_parameter("att", this.att);

        if (edgeDim != null && edgeDim > 0) {
            this.linEdge = register_module("lin_edge", new LinearImpl(edgeDim, heads * outChannels));
        } else {
            this.linEdge = null;
        }

        if (hasBias) {
            long biasDim = concat ? heads * outChannels : outChannels;
            Tensor b = torch.zeros(new long[]{biasDim},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
            this.bias = new Parameter(b, true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    @Override
    protected boolean needsX_i() {
        return true;
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
        Tensor ei = edge_index;
        if (addSelfLoops) {
            ei = org.bytedeco.pytorch.geometric.utils.GraphUtils.add_self_loops(ei, N);
        }

        // Independent projections → [N, H, C]
        Tensor xSrc = linSrc.forward(x).view(N, heads, outChannels);
        Tensor xDst = linDst.forward(x).view(N, heads, outChannels);

        // Bipartite-style propagate: different lifted features for j and i
        Tensor out = super.propagate(ei, xSrc, xDst, edge_attr);

        if (concat) {
            out = out.view(N, heads * outChannels);
        } else {
            out = out.mean(1);
        }
        if (bias != null) {
            out = out.add(bias);
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // e_ij = a^T LeakyReLU(x_i + x_j [+ e])
        Tensor out = x_i.add(x_j);
        if (edge_attr != null && linEdge != null) {
            Tensor e = linEdge.forward(edge_attr).view(-1, heads, outChannels);
            out = out.add(e);
        }
        out = torch.leaky_relu(out, new Scalar(negativeSlope));
        Tensor alpha = out.mul(att).sum(-1); // [E, H]

        Tensor targetIdx = _index_i != null ? _index_i : edge_index.select(0, 1);
        targetIdx = AggrUtils.asLongIndex(targetIdx);
        long n = numNodes > 0 ? numNodes : (_size != null ? _size[1] : 0);
        alpha = AggrUtils.scatter_softmax(alpha, targetIdx, n);
        return x_j.mul(alpha.unsqueeze(-1));
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
