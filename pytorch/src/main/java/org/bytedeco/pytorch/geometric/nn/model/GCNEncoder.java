package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.nn.conv.GCNConv;

import static org.bytedeco.pytorch.global.torch.relu;

public class GCNEncoder extends GenericModule {
    private GCNConv conv1;
    private GCNConv conv2;

    public GCNEncoder(long in, long out) {
        this.conv1 = new GCNConv(in, 2 * out);
        this.conv2 = new GCNConv(2 * out, out);
        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        Tensor x = inputs[0];
        Tensor edge_index = inputs[1];

        x = relu(conv1.forward(x, edge_index));
        x = conv2.forward(x, edge_index);
        return x;
    }

//    @Override
//    public Tensor forward(Tensor... inputs) {
//        return null;
//    }
}
