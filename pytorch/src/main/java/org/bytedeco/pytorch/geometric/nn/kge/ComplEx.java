package org.bytedeco.pytorch.geometric.nn.kge;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

public class ComplEx extends KGEModel {
    // 继承的 nodeEmb, relEmb 作为实部 (Re)
    private EmbeddingImpl nodeEmbIm; // 虚部 (Im)
    private EmbeddingImpl relEmbIm;

    public ComplEx(long numNodes, long numRels, long hiddenChannels) {
        super(numNodes, numRels, hiddenChannels);

        this.nodeEmbIm = new EmbeddingImpl(numNodes, hiddenChannels);
        this.relEmbIm = new EmbeddingImpl(numRels, hiddenChannels);

        torch.xavier_normal_(nodeEmbIm.weight());
        torch.xavier_normal_(relEmbIm.weight());

        register_module("nodeEmbIm", nodeEmbIm);
        register_module("relEmbIm", relEmbIm);
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        // Re
        Tensor h_re = nodeEmb.forward(head);
        Tensor r_re = relEmb.forward(relation);
        Tensor t_re = nodeEmb.forward(tail);

        // Im
        Tensor h_im = nodeEmbIm.forward(head);
        Tensor r_im = relEmbIm.forward(relation);
        Tensor t_im = nodeEmbIm.forward(tail);

        // ComplEx Score: Re( <h, r, conjugate(t)> )
        // = <h_re, r_re, t_re> + <h_re, r_im, t_im> + <h_im, r_re, t_im> - <h_im, r_im, t_re>
        // component-wise multiply and sum

        Tensor score = h_re.mul(r_re).mul(t_re)
                .add(h_re.mul(r_im).mul(t_im))
                .add(h_im.mul(r_re).mul(t_im))
                .sub(h_im.mul(r_im).mul(t_re));

        return score.sum(new long[]{1}, false,new ScalarTypeOptional());
    }
}