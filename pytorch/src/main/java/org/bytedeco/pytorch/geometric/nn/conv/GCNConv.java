package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.utils.GraphUtils;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

/**
 * Graph Convolutional Network layer (Kipf & Welling).
 *
 * <pre>
 *   X' = D̃^{-1/2} Ã D̃^{-1/2} X Θ
 * </pre>
 * where Ã = A + I. Aggregation is sum over normalized neighbor messages.
 */
public class GCNConv extends MessagePassing {

    private final LinearImpl lin;
    private final long inChannels;
    private final long outChannels;
    private final boolean addSelfLoops;
    private final boolean normalize;

    public GCNConv(Pointer p) {
        super(p);
        this.lin = null;
        this.inChannels = 0;
        this.outChannels = 0;
        this.addSelfLoops = true;
        this.normalize = true;
    }

    public GCNConv(long inChannels, long outChannels) {
        this(inChannels, outChannels, true, true);
    }

    public GCNConv(long inChannels, long outChannels, boolean addSelfLoops, boolean normalize) {
        super("sum");
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.addSelfLoops = addSelfLoops;
        this.normalize = normalize;
        this.lin = register_module("lin", new LinearImpl(inChannels, outChannels));
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forward(x, edge_index, (Tensor) null);
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index, Tensor edge_weight) {
        // 1. Linear transform first (PyG default: improved=False path still does XΘ then propagate)
        Tensor xLin = lin.forward(x);
        long numNodes = xLin.size(0);

        Tensor ei = edge_index;
        Tensor ew = edge_weight;

        if (normalize) {
            torch.ScalarType dtype = xLin.scalar_type().intern();
            Tensor[] normed = GraphUtils.gcn_norm(ei, ew, numNodes, addSelfLoops, dtype);
            ei = normed[0];
            ew = normed[1];
        } else if (addSelfLoops) {
            if (ew == null) {
                ei = GraphUtils.add_self_loops(ei, numNodes);
            } else {
                Tensor[] pair = GraphUtils.add_self_loops(ei, ew, numNodes, 1.0);
                ei = pair[0];
                ew = pair[1];
            }
        }

        // 2. Propagate with (optional) normalized edge weights as edge_attr
        return propagate(ei, xLin, ew);
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
        // LinearImpl typically self-inits; expose hook for callers
        if (lin != null) {
            // no-op if LinearImpl has no public reset; kept for API parity
        }
    }

    public LinearImpl getLin() {
        return lin;
    }

    public long getInChannels() {
        return inChannels;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
