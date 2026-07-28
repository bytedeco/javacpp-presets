package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * Edge-Conditioned Convolution / NNConv (Gilmer et al., MPNN; Simonovsky & Komodakis).
 *
 * <pre>
 *   W_{ij} = NN(e_{ij}) ∈ R^{F_in × F_out}
 *   m_{ij} = x_j  W_{ij}
 *   x'_i   = Σ_j m_{ij}  (+ W_root x_i + b)
 * </pre>
 * {@code nn} must map edge features → {@code inChannels * outChannels}.
 */
public class NNConv extends MessagePassing {

    private final long inChannels;
    private final long outChannels;
    private final Module nn;
    private final LinearImpl linRoot;
    private final Parameter bias;

    public NNConv(long inChannels, long outChannels, Module nn) {
        this(inChannels, outChannels, nn, "sum", true, true);
    }

    public NNConv(long inChannels, long outChannels, Module nn, String aggr,
                  boolean rootWeight, boolean hasBias) {
        super(aggr != null ? aggr : "sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        if (nn == null) {
            throw new IllegalArgumentException("nn (edge MLP) must not be null");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.nn = register_module("nn", nn);

        if (rootWeight) {
            this.linRoot = register_module("lin_root", new LinearImpl(inChannels, outChannels));
        } else {
            this.linRoot = null;
        }

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
        throw new UnsupportedOperationException(
                "NNConv requires edge_attr — use forward(x, edge_index, edge_attr)");
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (edge_attr == null) {
            throw new IllegalArgumentException("NNConv requires edge_attr");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }

        Tensor out = propagate(edge_index, x, edge_attr);
        if (linRoot != null) {
            out = out.add(linRoot.forward(x));
        }
        if (bias != null) {
            out = out.add(bias);
        }
        return out;
    }

    @Override
    public Tensor message(Tensor x_j, Tensor x_i, Tensor edge_index, Tensor edge_attr, long numNodes) {
        // W(e) : [E, in*out] → [E, in, out]
        Tensor weight = forwardNn(edge_attr).view(new long[]{-1, inChannels, outChannels});
        // x_j [E, in] → [E, 1, in] @ [E, in, out] → [E, out]
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1);
    }

    private Tensor forwardNn(Tensor edgeAttr) {
        if (nn instanceof SequentialImpl) {
            return ((SequentialImpl) nn).forward(edgeAttr);
        }
        if (nn instanceof LinearImpl) {
            return ((LinearImpl) nn).forward(edgeAttr);
        }
        return nn.asSequential().forward(edgeAttr);
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }

    public Module getNn() {
        return nn;
    }
}
