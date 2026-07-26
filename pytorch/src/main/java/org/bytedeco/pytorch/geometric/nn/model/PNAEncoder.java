package org.bytedeco.pytorch.geometric.nn.model;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.geometric.nn.conv.PNAConv;

public class PNAEncoder extends GenericModule {
    private PNAConv conv1;

    public PNAEncoder(long in, long out, double avgDegree) {
        String[] aggs = {"mean", "max", "sum"};
        String[] scs = {"identity", "amplification"};
        this.conv1 = new PNAConv(in, out, aggs, scs, avgDegree);
        register_module("conv1", conv1);
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        return conv1.forward(inputs[0], inputs[1]);
    }
}
