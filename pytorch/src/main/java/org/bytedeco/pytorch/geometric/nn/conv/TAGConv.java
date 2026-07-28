package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.ArrayList;
import java.util.List;

/**
 * Topology Adaptive Graph Convolution (Du et al., TAGConv).
 *
 * <pre>
 *   X'_k = Ã X'_{k-1}     (X'_0 = X)
 *   Y    = Σ_{k=0}^{K} X'_k  Θ_k
 * </pre>
 * Uses industrial MessagePassing for each hop; no intermediate Tensor.close().
 */
public class TAGConv extends MessagePassing {

    private final List<LinearImpl> lins;
    private final int K;
    private final long inChannels;
    private final long outChannels;
    private final boolean normalize;
    private final boolean addSelfLoops;

    public TAGConv(long inChannels, long outChannels, int K) {
        this(inChannels, outChannels, K, true, true);
    }

    public TAGConv(long inChannels, long outChannels, int K,
                   boolean normalize, boolean addSelfLoops) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        if (K < 0) {
            throw new IllegalArgumentException("K must be >= 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.K = K;
        this.normalize = normalize;
        this.addSelfLoops = addSelfLoops;
        this.lins = new ArrayList<>(K + 1);
        for (int k = 0; k <= K; k++) {
            LinearImpl lin = new LinearImpl(inChannels, outChannels);
            lins.add(lin);
            register_module("lin_" + k, lin);
        }
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
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x must be [N," + inChannels + "], got dim=" + x.dim()
                            + " F=" + (x.dim() > 1 ? x.size(1) : -1));
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
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

        // Y = Θ_0 X + Θ_1 ÃX + … + Θ_K Ã^K X
        Tensor out = lins.get(0).forward(x);
        Tensor current = x;
        for (int k = 1; k <= K; k++) {
            current = propagate(ei, current, ew);
            out = out.add(lins.get(k).forward(current));
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

    public void reset_parameters() {
        for (LinearImpl lin : lins) {
            lin.reset_parameters();
        }
    }

    public int getK() {
        return K;
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
