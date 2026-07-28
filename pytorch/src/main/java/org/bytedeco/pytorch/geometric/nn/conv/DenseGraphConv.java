package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Dense (batched) GraphConv (Morris et al.).
 *
 * <pre>
 *   x' = W_rel · (A X) + W_root · X
 * </pre>
 * Inputs {@code x [B,N,F]}, {@code adj [B,N,N]}.
 */
public class DenseGraphConv extends MessagePassing {

    private final LinearImpl linRel;
    private final LinearImpl linRoot;
    private final long inChannels;
    private final long outChannels;

    public DenseGraphConv(long inChannels, long outChannels) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.linRel = register_module("linRel", new LinearImpl(inChannels, outChannels));
        this.linRoot = register_module("linRoot", new LinearImpl(inChannels, outChannels));
    }

    /**
     * Dense forward. Second arg is adjacency {@code [B,N,N]}.
     * @param x   [B, N, inChannels]
     * @param adj [B, N, N]
     */
    @Override
    public Tensor forward(Tensor x, Tensor adj) {
        if (x == null || adj == null) {
            throw new NullPointerException("x and adj must not be null");
        }
        if (x.dim() != 3 || adj.dim() != 3) {
            throw new IllegalArgumentException("x must be [B,N,C], adj [B,N,N]");
        }
        if (x.size(2) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(2)=" + x.size(2) + " != inChannels=" + inChannels);
        }
        Tensor neighbor = adj.matmul(x);
        return linRel.forward(neighbor).add(linRoot.forward(x));
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        return x_j;
    }

    public LinearImpl getLinRel() {
        return linRel;
    }

    public LinearImpl getLinRoot() {
        return linRoot;
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
