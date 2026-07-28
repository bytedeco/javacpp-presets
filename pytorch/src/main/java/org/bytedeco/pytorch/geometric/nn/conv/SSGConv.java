package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Simple Spectral Graph Convolution (Zhu & Koniusz, AAAI 2021) — SSGC.
 *
 * <pre>
 *   Z = Σ_{k=0}^{K} α (1-α)^k  Ã^k X     (PyG: closed form via iteration)
 *   Y = Z Θ + b
 * </pre>
 * Common practical form used by PyG:
 * {@code out = (1-α) Ã^K X + α X} then linear.
 */
public class SSGConv extends MessagePassing {

    private final LinearImpl lin;
    private final Parameter bias;
    private final double alpha;
    private final int K;
    private final boolean addSelfLoops;
    private final long inChannels;
    private final long outChannels;

    public SSGConv(long inChannels, long outChannels, double alpha, int K) {
        this(inChannels, outChannels, alpha, K, true, true);
    }

    public SSGConv(long inChannels, long outChannels, double alpha, int K,
                   boolean hasBias) {
        this(inChannels, outChannels, alpha, K, hasBias, true);
    }

    public SSGConv(long inChannels, long outChannels, double alpha, int K,
                   boolean hasBias, boolean addSelfLoops) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || K < 1) {
            throw new IllegalArgumentException("invalid SSGConv dims / K");
        }
        if (alpha < 0.0 || alpha > 1.0) {
            throw new IllegalArgumentException("alpha must be in [0,1]");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.alpha = alpha;
        this.K = K;
        this.addSelfLoops = addSelfLoops;
        this.lin = register_module("lin", new LinearImpl(inChannels, outChannels));
        if (hasBias) {
            Tensor b = torch.zeros(new long[]{outChannels},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
            this.bias = new Parameter(b, true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
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
        long N = x.size(0);
        torch.ScalarType dtype = x.scalar_type().intern();
        Tensor[] normed = GraphUtils.gcn_norm(edge_index, edge_weight, N, addSelfLoops, dtype);
        Tensor ei = normed[0];
        Tensor norm = normed[1];

        Tensor x0 = x;
        Tensor xk = x;
        for (int k = 0; k < K; k++) {
            xk = propagate(ei, xk, norm);
        }
        // (1-α) Ã^K X + α X
        Tensor mixed = xk.mul(new Scalar(1.0 - alpha)).add(x0.mul(new Scalar(alpha)));
        Tensor result = lin.forward(mixed);
        if (bias != null) {
            result = result.add(bias);
        }
        return result;
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

    public double getAlpha() {
        return alpha;
    }

    public int getK() {
        return K;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
