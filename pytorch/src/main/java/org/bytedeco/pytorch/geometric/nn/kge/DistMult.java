package org.bytedeco.pytorch.geometric.nn.kge;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;

/**
 * DistMult (Yang et al.): {@code score = ⟨h, r, t⟩ = Σ_i h_i r_i t_i}.
 */
public class DistMult extends KGEModel {

    public DistMult(long numNodes, long numRels, long hiddenChannels) {
        super(numNodes, numRels, hiddenChannels);
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        Tensor h = nodeEmb.forward(head);
        Tensor r = relEmb.forward(relation);
        Tensor t = nodeEmb.forward(tail);
        return h.mul(r).mul(t).sum(new long[]{1}, false, new ScalarTypeOptional());
    }
}
