package org.bytedeco.pytorch.geometric.nn.conv;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.modules.GRUCellImpl;

/**
 * Gated Graph Sequence Neural Network convolution (Li et al., ICLR 2016).
 *
 * <pre>
 *   m^{(t)} = A h^{(t-1)}          (sum-aggregate neighbor states)
 *   h^{(t)} = GRU(m^{(t)}, h^{(t-1)})
 * </pre>
 * Input feature dim must equal {@code outChannels} (hidden size of the GRU cell).
 */
public class GatedGraphConv extends MessagePassing {

    private final GRUCellImpl gru;
    private final int numLayers;
    private final long outChannels;

    public GatedGraphConv(long outChannels, int numLayers) {
        super("sum");
        if (outChannels <= 0 || numLayers < 1) {
            throw new IllegalArgumentException("outChannels > 0 and numLayers >= 1 required");
        }
        this.outChannels = outChannels;
        this.numLayers = numLayers;
        this.gru = register_module("gru", new GRUCellImpl(outChannels, outChannels));
    }

    @Override
    public Tensor forward(Tensor x, Tensor edge_index) {
        if (x == null || edge_index == null) {
            throw new NullPointerException("x and edge_index must not be null");
        }
        if (x.size(1) != outChannels) {
            throw new IllegalArgumentException(
                    "GatedGraphConv expects x.size(1)==outChannels=" + outChannels
                            + ", got " + x.size(1));
        }
        Tensor h = x;
        for (int t = 0; t < numLayers; t++) {
            Tensor m = propagate(edge_index, h);
            h = gru.forward(m, h);
        }
        return h;
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

    public int getNumLayers() {
        return numLayers;
    }

    public long getOutChannels() {
        return outChannels;
    }
}
