package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Local Extrema Convolution (Ranjan et al. / ASAP).
 *
 * <pre>
 *   m_i = Σ_{j ∈ N(i)} w_{ij} (x_i - x_j)
 *   x'_i = W_1 x_i + W_2 m_i + b
 * </pre>
 * Difference operator for local extrema; foundation of ASAP pooling.
 */
public class LEConv extends MessagePassing {

    private final LinearImpl lin1;
    private final LinearImpl lin2;
    private final Parameter bias;
    private final long inChannels;
    private final long outChannels;

    public LEConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, true);
    }

    public LEConv(long inChannels, long outChannels, boolean hasBias) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.lin1 = register_module("lin1", new LinearImpl(inChannels, outChannels));
        this.lin2 = register_module("lin2", new LinearImpl(inChannels, outChannels));
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
    protected boolean needsX_i() {
        return true; // message uses x_i - x_j
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
        Tensor out = propagate(edge_index, x, edge_weight);
        Tensor res = lin1.forward(x).add(lin2.forward(out));
        if (bias != null) {
            res = res.add(bias);
        }
        return res;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        Tensor msg = x_i.sub(x_j);
        if (edge_attr != null) {
            if (edge_attr.dim() == 1) {
                msg = msg.mul(edge_attr.view(new long[]{-1, 1}));
            } else {
                msg = msg.mul(edge_attr);
            }
        }
        return msg;
    }

    public LinearImpl getLin1() {
        return lin1;
    }

    public LinearImpl getLin2() {
        return lin2;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
