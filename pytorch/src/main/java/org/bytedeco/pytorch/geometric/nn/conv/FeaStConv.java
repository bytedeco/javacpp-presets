package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * Feature Steered graph convolution (Verma et al., FeaStNet / PyG {@code FeaStConv}).
 *
 * <pre>
 *   t_m(i,j) = softmax_m( u_m^T x_j + v_m^T x_i )   // soft assignment over heads
 *   y_i      = Σ_j Σ_m t_m(i,j) · (W_m x_j) (+ b)
 * </pre>
 * Softmax is over the <b>head</b> dimension (not neighbors). Messages are then
 * sum-aggregated by industrial {@link MessagePassing}.
 */
public class FeaStConv extends MessagePassing {

    private final LinearImpl linWeights; // in → heads * out
    private final LinearImpl linSrc;     // in → heads (neighbor j)
    private final LinearImpl linDst;     // in → heads (center i)
    private final Parameter bias;
    private final int heads;
    private final long outChannels;
    private final long inChannels;

    /** Raw node features stashed for soft-assignment scores during one forward. */
    private Tensor _xRaw;

    public FeaStConv(long inChannels, long outChannels, int heads) {
        this(inChannels, outChannels, heads, true);
    }

    public FeaStConv(long inChannels, long outChannels, int heads, boolean hasBias) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0 || heads < 1) {
            throw new IllegalArgumentException("FeaStConv dims invalid");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.heads = heads;

        this.linWeights = register_module("lin_weights",
                new LinearImpl(new LinearOptions(inChannels, (long) heads * outChannels)));
        this.linSrc = register_module("lin_src", new LinearImpl(inChannels, heads));
        this.linDst = register_module("lin_dst", new LinearImpl(inChannels, heads));

        if (hasBias) {
            Tensor b = torch.zeros(new long[]{outChannels},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
            this.bias = new Parameter(b.clone().requires_grad_(true), true);
            register_parameter("bias", this.bias);
        } else {
            this.bias = null;
        }
    }

    @Override
    protected boolean needsX_i() {
        // Soft assignment uses both endpoints via raw features + indices
        return false;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.dim() != 2 || x.size(1) != inChannels) {
            throw new IllegalArgumentException("x must be [N," + inChannels + "]");
        }
        if (edge_index.dim() != 2 || edge_index.size(0) != 2) {
            throw new IllegalArgumentException("edge_index must be [2,E]");
        }

        long N = x.size(0);
        if (edge_index.size(1) == 0) {
            Tensor out = torch.zeros(new long[]{N, outChannels}, x.options());
            return bias != null ? out.add(bias) : out;
        }

        this._xRaw = x;
        try {
            // Multi-head projection on nodes → [N, H, C]
            Tensor xH = linWeights.forward(x).view(N, heads, outChannels);
            Tensor out = propagate(edge_index, xH);
            if (bias != null) {
                out = out.add(bias);
            }
            return out;
        } finally {
            this._xRaw = null;
        }
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // x_j: [E, H, C] lifted multi-head features
        if (_xRaw == null || _index_j == null || _index_i == null) {
            // Degenerate fallback: average heads
            return x_j.mean(1);
        }
        Tensor xjRaw = _xRaw.index_select(0, _index_j);
        Tensor xiRaw = _xRaw.index_select(0, _index_i);
        // Soft assignment over heads (dim=-1)
        Tensor q = linSrc.forward(xjRaw).add(linDst.forward(xiRaw)); // [E, H]
        Tensor alpha = torch.softmax(q, -1);                          // [E, H]
        // Σ_m α_m · (W_m x_j)_m
        return x_j.mul(alpha.unsqueeze(-1)).sum(1);                   // [E, C]
    }

    public int getHeads() {
        return heads;
    }

    public long getOutChannels() {
        return outChannels;
    }

    /** Alias used by older tests. */
    public long getOutChannelsPerHead() {
        return outChannels;
    }

    public LinearImpl getLinWeights() {
        return linWeights;
    }

    public LinearImpl getLinSrc() {
        return linSrc;
    }

    public LinearImpl getLinDst() {
        return linDst;
    }

    public Parameter getBias() {
        return bias;
    }
}
