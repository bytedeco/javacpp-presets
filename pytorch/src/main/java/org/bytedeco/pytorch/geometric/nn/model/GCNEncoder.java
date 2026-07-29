package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

import static org.bytedeco.pytorch.global.torch.relu;

/**
 * Two-layer GCN encoder commonly used by {@link GAE}/{@link VGAE}.
 *
 * <p>Exposes both {@code forward(x, edge_index)} (for Module dispatch) and
 * {@code forward(Tensor...)} (GenericModule varargs).
 */
public class GCNEncoder extends GenericModule {
    private final GCNConv conv1;
    private final GCNConv conv2;

    public GCNEncoder(long in, long out) {
        this.conv1 = new GCNConv(in, 2 * out);
        this.conv2 = new GCNConv(2 * out, out);
        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    /** Standard (x, edge_index) path — picked up by ModuleAsHelper. */
    public Tensor forward(Tensor x, Tensor edge_index) {
        return forwardImpl(x, edge_index);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        if (inputs == null || inputs.length < 2) {
            throw new IllegalArgumentException("GCNEncoder.forward expects [x, edge_index]");
        }
        return forwardImpl(inputs[0], inputs[1]);
    }

    private Tensor forwardImpl(Tensor x, Tensor edge_index) {
        x = relu(conv1.forward(x, edge_index));
        x = conv2.forward(x, edge_index);
        return x;
    }
}
