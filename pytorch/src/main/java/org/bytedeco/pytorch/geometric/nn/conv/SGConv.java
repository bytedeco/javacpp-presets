package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * Simple Graph Convolution (Wu et al., ICML 2019).
 *
 * <pre>
 *   S = D̃^{-1/2} Ã D̃^{-1/2}
 *   Y = S^K X Θ
 * </pre>
 * K-hop linear propagation then a single linear transform (no intermediate nonlinearities).
 */
public class SGConv extends MessagePassing {

    private final LinearImpl lin;
    private final int K;
    private final boolean addSelfLoops;
    private final boolean cached;
    private Tensor cachedEdgeIndex;
    private Tensor cachedNorm;
    private final long inChannels;
    private final long outChannels;

    public SGConv(long inChannels, long outChannels, int K) {
        this(inChannels, outChannels, K, true, false);
    }

    public SGConv(long inChannels, long outChannels, int K, boolean addSelfLoops, boolean cached) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || K < 1) {
            throw new IllegalArgumentException("in/out > 0 and K >= 1 required");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.K = K;
        this.addSelfLoops = addSelfLoops;
        this.cached = cached;
        LinearOptions opt = new LinearOptions(inChannels, outChannels);
        this.lin = register_module("lin", new LinearImpl(opt));
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
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }

        long N = x.size(0);
        Tensor ei;
        Tensor norm;

        if (cached && cachedEdgeIndex != null && cachedNorm != null) {
            ei = cachedEdgeIndex;
            norm = cachedNorm;
        } else {
            torch.ScalarType dtype = x.scalar_type().intern();
            Tensor[] normed = GraphUtils.gcn_norm(edge_index, edge_weight, N, addSelfLoops, dtype);
            ei = normed[0];
            norm = normed[1];
            if (cached) {
                cachedEdgeIndex = ei;
                cachedNorm = norm;
            }
        }

        Tensor xRun = x;
        for (int i = 0; i < K; i++) {
            xRun = propagate(ei, xRun, norm);
        }
        return lin.forward(xRun);
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

    /** Drop cached Ã_norm (call after graph structure changes). */
    public void clearCache() {
        cachedEdgeIndex = null;
        cachedNorm = null;
    }

    public LinearImpl getLin() {
        return lin;
    }

    public int getK() {
        return K;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
