package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * GraphConv (Morris et al. / PyG):
 *
 * <pre>
 *   x'_i = W_root x_i + W_rel · Σ_{j ∈ N(i)} w_{ij} x_j
 * </pre>
 *
 * Supports homogeneous and bipartite graphs (different in-dims for src/dst).
 *
 * <p>Java cannot overload {@code forward(Tensor,Tensor,Tensor)} twice, so the
 * 3-arg form dispatches by shape of the second argument:
 * <ul>
 *   <li>rank-2 feature matrix → bipartite {@code (xSrc, xDst, edge_index)}</li>
 *   <li>rank-1 edge weight / null → homogeneous {@code (x, edge_index, edge_weight)}</li>
 * </ul>
 */
public class GraphConv extends MessagePassing {

    private final LinearImpl linRoot;
    private final LinearImpl linRel;
    private final long inDimSrc;
    private final long inDimDst;
    private final long outChannels;

    /** Homogeneous: same in-dim for self and neighbors. */
    public GraphConv(long inChannels, long outChannels) {
        this(inChannels, inChannels, outChannels);
    }

    /**
     * Bipartite-capable constructor.
     *
     * @param inDimSrc source / neighbor feature dim
     * @param inDimDst destination / root feature dim
     * @param outDim   output dim
     */
    public GraphConv(long inDimSrc, long inDimDst, long outDim) {
        super("sum");
        if (inDimSrc <= 0 || inDimDst <= 0 || outDim <= 0) {
            throw new IllegalArgumentException("feature dims must be > 0");
        }
        this.inDimSrc = inDimSrc;
        this.inDimDst = inDimDst;
        this.outChannels = outDim;
        this.linRel = register_module("linRel", new LinearImpl(inDimSrc, outDim));
        this.linRoot = register_module("linRoot", new LinearImpl(inDimDst, outDim));
    }

    /** Homogeneous forward. */
    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forwardHomogeneous(x, edge_index, null);
    }

    /**
     * Unified 3-arg forward (Java overload limit).
     * <ul>
     *   <li>If {@code second} is rank-2 → bipartite {@code (xSrc, xDst, edge_index)}</li>
     *   <li>Else → homogeneous {@code (x, edge_index, edge_weight)} ({@code second} may be null)</li>
     * </ul>
     */
    @Override
    public Tensor forward(Tensor first, Tensor second, Tensor third) {
        if (second != null && second.dim() == 2 && third != null && third.dim() == 2
                && third.size(0) == 2) {
            // (xSrc, xDst, edge_index)
            return forwardBipartite(first, second, third);
        }
        // (x, edge_index, edge_weight) — second is edge_index, third is weight
        return forwardHomogeneous(first, second, third);
    }

    private Tensor forwardHomogeneous(Tensor x, Tensor edge_index, Tensor edge_weight) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.size(1) != inDimSrc && inDimSrc == inDimDst) {
            // soft check only when homogeneous dims match
        }
        Tensor rel = linRel.forward(x);
        Tensor out = propagate(edge_index, rel, edge_weight);
        Tensor root = linRoot.forward(x);
        return out.add(root);
    }

    /**
     * Bipartite forward: aggregate transformed {@code xSrc} onto {@code xDst}.
     *
     * @param xSrc       [N_src, inDimSrc]
     * @param xDst       [N_dst, inDimDst]
     * @param edge_index [2, E] src→dst
     */
    public Tensor forwardBipartite(Tensor xSrc, Tensor xDst, Tensor edge_index) {
        if (xSrc == null || xDst == null || edge_index == null) {
            throw new NullPointerException("xSrc, xDst, edge_index must not be null");
        }
        if (xSrc.size(1) != inDimSrc) {
            throw new IllegalArgumentException(
                    "xSrc.size(1)=" + xSrc.size(1) + " != inDimSrc=" + inDimSrc);
        }
        if (xDst.size(1) != inDimDst) {
            throw new IllegalArgumentException(
                    "xDst.size(1)=" + xDst.size(1) + " != inDimDst=" + inDimDst);
        }
        Tensor rel = linRel.forward(xSrc);
        long[] size = new long[]{xSrc.size(0), xDst.size(0)};
        Tensor out = propagate(edge_index, rel, size);
        Tensor root = linRoot.forward(xDst);
        return out.add(root);
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        if (edge_attr != null) {
            if (edge_attr.dim() == 1) {
                return x_j.mul(edge_attr.view(new long[]{-1, 1}));
            }
            // [E,1] or broadcastable
            return x_j.mul(edge_attr);
        }
        return x_j;
    }

    public LinearImpl getLinRoot() {
        return linRoot;
    }

    public LinearImpl getLinRel() {
        return linRel;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
