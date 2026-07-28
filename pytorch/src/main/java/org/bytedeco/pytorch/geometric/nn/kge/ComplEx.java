package org.bytedeco.pytorch.geometric.nn.kge;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;

/**
 * ComplEx (Trouillon et al.): complex-valued bilinear scoring.
 *
 * <pre>
 *   score = Re( ⟨h, r, conjugate(t)⟩ )
 *         = ⟨h_re,r_re,t_re⟩ + ⟨h_re,r_im,t_im⟩
 *         + ⟨h_im,r_re,t_im⟩ − ⟨h_im,r_im,t_re⟩
 * </pre>
 * Real parts live in the base {@code nodeEmb/relEmb}; imaginary parts in
 * {@code nodeEmbIm/relEmbIm}.
 */
public class ComplEx extends KGEModel {

    private final EmbeddingImpl nodeEmbIm;
    private final EmbeddingImpl relEmbIm;

    public ComplEx(long numNodes, long numRels, long hiddenChannels) {
        super(numNodes, numRels, hiddenChannels);
        this.nodeEmbIm = register_module("nodeEmbIm", new EmbeddingImpl(numNodes, hiddenChannels));
        this.relEmbIm = register_module("relEmbIm", new EmbeddingImpl(numRels, hiddenChannels));
        torch.xavier_uniform_(nodeEmbIm.weight());
        torch.xavier_uniform_(relEmbIm.weight());
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        Tensor hRe = nodeEmb.forward(head);
        Tensor rRe = relEmb.forward(relation);
        Tensor tRe = nodeEmb.forward(tail);
        Tensor hIm = nodeEmbIm.forward(head);
        Tensor rIm = relEmbIm.forward(relation);
        Tensor tIm = nodeEmbIm.forward(tail);

        Tensor score = hRe.mul(rRe).mul(tRe)
                .add(hRe.mul(rIm).mul(tIm))
                .add(hIm.mul(rRe).mul(tIm))
                .sub(hIm.mul(rIm).mul(tRe));
        return score.sum(new long[]{1}, false, new ScalarTypeOptional());
    }

    public EmbeddingImpl nodeImagEmbeddings() {
        return nodeEmbIm;
    }

    public EmbeddingImpl relImagEmbeddings() {
        return relEmbIm;
    }
}
