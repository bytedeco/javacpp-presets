package org.bytedeco.pytorch.geometric.nn.kge;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;

/**
 * RotatE (Sun et al.): relations as rotations in complex space.
 *
 * <pre>
 *   h, t ∈ ℂ^{d}  (stored as [re; im] of length 2d)
 *   r = e^{iθ}    (θ ∈ R^{d} from relation embedding)
 *   score = −‖ h ⊙ r − t ‖₂
 * </pre>
 * Entity embedding dim is {@code 2 * hiddenChannels}; relation emb dim is
 * {@code hiddenChannels} (phase only).
 */
public class RotatE extends KGEModel {

    private final long complexDim; // d = hiddenChannels from user
    private final EmbeddingImpl entityEmb; // [numNodes, 2d]
    private final EmbeddingImpl phaseEmb;  // [numRels, d]
    private final double phaseInit;

    /**
     * @param hiddenChannels complex dimension d (entity params = 2d)
     */
    public RotatE(long numNodes, long numRels, long hiddenChannels) {
        this(numNodes, numRels, hiddenChannels, Math.PI);
    }

    public RotatE(long numNodes, long numRels, long hiddenChannels, double phaseInit) {
        // Base still creates nodeEmb/relEmb at dim=2d for API compatibility of getters;
        // we use dedicated entity/phase embeddings for the actual scoring.
        super(numNodes, numRels, hiddenChannels * 2);
        if (hiddenChannels <= 0) {
            throw new IllegalArgumentException("hiddenChannels (complex dim) must be > 0");
        }
        this.complexDim = hiddenChannels;
        this.phaseInit = phaseInit;

        this.entityEmb = register_module("entityEmb",
                new EmbeddingImpl(numNodes, hiddenChannels * 2));
        this.phaseEmb = register_module("phaseEmb",
                new EmbeddingImpl(numRels, hiddenChannels));
        torch.xavier_uniform_(entityEmb.weight());
        // Phase uniform in (-phaseInit, phaseInit)
        torch.uniform_(phaseEmb.weight(), -phaseInit, phaseInit);
    }

    @Override
    public Tensor forward(Tensor head, Tensor relation, Tensor tail) {
        Tensor h = entityEmb.forward(head); // [B, 2d]
        Tensor t = entityEmb.forward(tail);
        Tensor theta = phaseEmb.forward(relation); // [B, d]

        TensorVector hParts = torch.chunk(h, 2, 1);
        Tensor hRe = hParts.get(0);
        Tensor hIm = hParts.get(1);
        TensorVector tParts = torch.chunk(t, 2, 1);
        Tensor tRe = tParts.get(0);
        Tensor tIm = tParts.get(1);

        Tensor rRe = torch.cos(theta);
        Tensor rIm = torch.sin(theta);

        // (h_re + i h_im)(r_re + i r_im)
        Tensor rotRe = hRe.mul(rRe).sub(hIm.mul(rIm));
        Tensor rotIm = hRe.mul(rIm).add(hIm.mul(rRe));

        Tensor dist = rotRe.sub(tRe).pow(new Scalar(2))
                .add(rotIm.sub(tIm).pow(new Scalar(2)))
                .sum(new long[]{1}, false, new ScalarTypeOptional())
                .sqrt();
        return dist.neg();
    }

    public long getComplexDim() {
        return complexDim;
    }

    public EmbeddingImpl entityEmbeddings() {
        return entityEmb;
    }

    public EmbeddingImpl phaseEmbeddings() {
        return phaseEmb;
    }
}
