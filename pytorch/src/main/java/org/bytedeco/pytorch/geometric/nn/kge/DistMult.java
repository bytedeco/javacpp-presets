package org.bytedeco.pytorch.geometric.nn.kge;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;

public class DistMult extends KGEModel {

    public DistMult(long numNodes, long numRels, long hiddenChannels) {

        super(numNodes, numRels, hiddenChannels);
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        Tensor h = nodeEmb.forward(head);
        Tensor r = relEmb.forward(relation);
        Tensor t = nodeEmb.forward(tail);

        // h * r * t sum over dim 1
        return h.mul(r).mul(t).sum(new long[]{1}, false,new ScalarTypeOptional());
    }
}