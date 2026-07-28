package org.bytedeco.pytorch.geometric.nn.kge;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;

/**
 * TransE (Bordes et al.): {@code score = −‖h + r − t‖_p}.
 * Entity embeddings are L2-normalized before scoring.
 */
public class TransE extends KGEModel {

    private final long pNorm;

    public TransE(long numNodes, long numRels, long hiddenChannels) {
        this(numNodes, numRels, hiddenChannels, 1);
    }

    public TransE(long numNodes, long numRels, long hiddenChannels, long p) {
        super(numNodes, numRels, hiddenChannels);
        if (p != 1 && p != 2) {
            throw new IllegalArgumentException("TransE p-norm must be 1 or 2, got " + p);
        }
        this.pNorm = p;
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        Tensor h = nodeEmb.forward(head);
        Tensor r = relEmb.forward(relation);
        Tensor t = nodeEmb.forward(tail);

        NormalizeFuncOptions normOpts = new NormalizeFuncOptions();
        normOpts.p().put(2);
        normOpts.dim().put(1);
        h = torch.normalize(h, normOpts);
        t = torch.normalize(t, normOpts);

        Tensor dist = h.add(r).sub(t)
                .norm(new ScalarOptional(new Scalar(pNorm)), new long[]{1}, false);
        return dist.neg(); // higher = better
    }

    public long getPNorm() {
        return pNorm;
    }
}
