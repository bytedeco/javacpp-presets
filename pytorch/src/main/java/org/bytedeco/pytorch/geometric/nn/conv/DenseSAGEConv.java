package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Dense (batched) GraphSAGE convolution.
 *
 * <pre>
 *   x'_i = W_rel · mean_j(A_{ij} x_j) + W_root · x_i
 * </pre>
 * Optional L2 row-normalization. Inputs {@code x [B,N,F]}, {@code adj [B,N,N]}.
 */
public class DenseSAGEConv extends MessagePassing {

    private final LinearImpl linRel;
    private final LinearImpl linRoot;
    private final boolean normalize;
    private final long inChannels;
    private final long outChannels;

    public DenseSAGEConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, false);
    }

    public DenseSAGEConv(long inChannels, long outChannels, boolean normalize) {
        super("mean");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.normalize = normalize;
        this.linRel = register_module("lin_rel", new LinearImpl(inChannels, outChannels));
        this.linRoot = register_module("lin_root", new LinearImpl(inChannels, outChannels));
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

        Tensor neighborSum = adj.matmul(x);
        Tensor deg = adj.sum(new long[]{2}, true, new ScalarTypeOptional(torch.ScalarType.Float));
        Tensor neighborMean = neighborSum.div(deg.add(new Scalar(1e-6)));

        Tensor out = linRel.forward(neighborMean).add(linRoot.forward(x));

        if (normalize) {
            Tensor norm = out.norm(new ScalarOptional(new Scalar(2.0)), new long[]{-1}, true);
            out = out.div(norm.clamp_min(new Scalar(1e-12)));
        }
        return out;
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

    public boolean isNormalize() {
        return normalize;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
