package org.bytedeco.pytorch.geometric.nn.kge;
import org.bytedeco.pytorch.nn.options.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;

public class TransE extends KGEModel {
    private long p; // L1 or L2 norm

    public TransE(long numNodes, long numRels, long hiddenChannels, long p) {
        super(numNodes, numRels, hiddenChannels);
        this.p = p;
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        Tensor h = nodeEmb.forward(head);
        Tensor r = relEmb.forward(relation);
        Tensor t = nodeEmb.forward(tail);

        // TransE 也经常对 Embedding 归一化 (|h|=1)
        NormalizeFuncOptions normOpts = new NormalizeFuncOptions();
        normOpts.p().put(2);
        normOpts.dim().put(1);
        h = torch.normalize(h, normOpts);
        t = torch.normalize(t, normOpts);

        // score = - || h + r - t ||
        Tensor dist = h.add(r).sub(t).norm(new ScalarOptional(new Scalar(p)), new long[]{1}, false);
        return dist.neg(); // 返回负距离，使得 loss 函数逻辑统一（越大越好）
    }
}