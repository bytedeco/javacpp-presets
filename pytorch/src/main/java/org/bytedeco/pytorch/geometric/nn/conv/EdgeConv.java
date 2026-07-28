package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * EdgeConv / Dynamic EdgeConv base operator (Wang et al., DGCNN).
 *
 * <pre>
 *   e_{ij} = NN( [ x_i || (x_j - x_i) ] )
 *   x'_i   = max_{j ∈ N(i)} e_{ij}
 * </pre>
 *
 * Uses industrial MessagePassing: {@link #needsX_i()} so both endpoints are lifted;
 * max aggregation is the DGCNN default (configurable via ctor).
 */
public class EdgeConv extends MessagePassing {

    private final Module nn;
    private final long inChannels;
    private final long outChannels;

    /** Default: internal MLP {@code 2*in → out → ReLU → out}, max aggr. */
    public EdgeConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, null, "max");
    }

    public EdgeConv(long inChannels, long outChannels, String aggr) {
        this(inChannels, outChannels, null, aggr);
    }

    /**
     * @param nn external edge MLP taking {@code 2 * inChannels} features; null → default MLP
     */
    public EdgeConv(long inChannels, long outChannels, Module nn, String aggr) {
        super(aggr != null ? aggr : "max");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        if (nn != null) {
            this.nn = register_module("nn", nn);
        } else {
            SequentialImpl mlp = new SequentialImpl();
            mlp.push_back(new LinearImpl(2 * inChannels, outChannels));
            mlp.push_back(new ReLUImpl());
            mlp.push_back(new LinearImpl(outChannels, outChannels));
            this.nn = register_module("nn", mlp);
        }
    }

    @Override
    protected boolean needsX_i() {
        return true;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.dim() != 2 || edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("x must be [N,F], edge_index [2,E]");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }
        return propagate(edge_index, x);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // e_ij = NN([x_i || x_j - x_i])
        Tensor diff = x_j.sub(x_i);
        Tensor catFeat = torch.cat(new TensorVector(x_i, diff), 1);
        if (catFeat.size(1) != 2 * inChannels) {
            throw new IllegalStateException(
                    "EdgeConv cat dim " + catFeat.size(1) + " != " + (2 * inChannels));
        }
        if (nn instanceof SequentialImpl) {
            return ((SequentialImpl) nn).forward(catFeat);
        }
        if (nn instanceof LinearImpl) {
            return ((LinearImpl) nn).forward(catFeat);
        }
        return nn.asSequential().forward(catFeat);
    }

    public Module getNn() {
        return nn;
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
