package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.nn.Parameter;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Residual Gated Graph ConvNet (Bresson & Laurent).
 *
 * <pre>
 *   η_{ij} = σ( D x_i + E x_j )
 *   m_{ij} = η_{ij} ⊙ (A x_j + B x_i)
 *   x'_i   = Σ_j m_{ij}  (+ W_root x_i + b)
 * </pre>
 * Gate is computed per edge; sum aggregation over gated messages.
 */
public class ResGatedGraphConv extends MessagePassing {

    private final LinearImpl linA;
    private final LinearImpl linB;
    private final LinearImpl linD;
    private final LinearImpl linE;
    private final LinearImpl linRoot;
    private final LinearImpl linEdge;
    private final Parameter bias;
    private final long inChannels;
    private final long outChannels;

    // Transient precomputed node transforms for one forward
    private Tensor _Ax;
    private Tensor _Bx;
    private Tensor _Dx;
    private Tensor _Ex;

    public ResGatedGraphConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, null, true, true);
    }

    public ResGatedGraphConv(long inChannels, long outChannels, Integer edgeDim,
                             boolean rootWeight, boolean hasBias) {
        super("sum");
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("in/out channels must be > 0");
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;

        this.linA = register_module("lin_a", new LinearImpl(inChannels, outChannels));
        this.linB = register_module("lin_b", new LinearImpl(inChannels, outChannels));
        this.linD = register_module("lin_d", new LinearImpl(inChannels, outChannels));
        this.linE = register_module("lin_e", new LinearImpl(inChannels, outChannels));

        if (edgeDim != null && edgeDim > 0) {
            this.linEdge = register_module("lin_edge", new LinearImpl(edgeDim, outChannels));
        } else {
            this.linEdge = null;
        }

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
    protected boolean needsX_i() {
        return true;
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_attr) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.size(1) != inChannels) {
            throw new IllegalArgumentException(
                    "x.size(1)=" + x.size(1) + " != inChannels=" + inChannels);
        }

        this._Ax = linA.forward(x);
        this._Bx = linB.forward(x);
        this._Dx = linD.forward(x);
        this._Ex = linE.forward(x);

        Tensor out;
        try {
            // Dummy features for lift; real values come from transients in message
            out = propagate(edge_index, x, edge_attr);
        } finally {
            this._Ax = null;
            this._Bx = null;
            this._Dx = null;
            this._Ex = null;
        }

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
        Tensor index_j = _index_j != null ? _index_j : edge_index.select(0, 0);
        Tensor index_i = _index_i != null ? _index_i : edge_index.select(0, 1);

        Tensor Ax_j = _Ax.index_select(0, index_j);
        Tensor Bx_i = _Bx.index_select(0, index_i);
        Tensor Dx_i = _Dx.index_select(0, index_i);
        Tensor Ex_j = _Ex.index_select(0, index_j);

        // η = σ(D x_i + E x_j [+ edge])
        Tensor gateLogits = Dx_i.add(Ex_j);
        if (edge_attr != null && linEdge != null) {
            gateLogits = gateLogits.add(linEdge.forward(edge_attr));
        }
        Tensor gate = torch.sigmoid(gateLogits);

        // m = η ⊙ (A x_j + B x_i)
        return gate.mul(Ax_j.add(Bx_i));
    }

    public long getOutChannels() {
        return outChannels;
    }
}
