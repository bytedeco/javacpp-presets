/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/FM.scala (FMInteraction)
 *
 * 2nd-order FM interaction only (no first-order).
 * Python原版对照: FM(reduce_sum=False) returns (batch, embed_dim)
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * 2nd-order FM interaction only (no first-order).
 * Computes: 0.5 * (sum_over_fields(v_i)^2 - sum_over_fields(v_i^2))
 * which gives pairwise interaction vectors of shape (batch, embed_dim).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FMInteraction extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;

    public FMInteraction(int embedDim) {
        super("FMInteraction");
        this.embedDim = embedDim;
    }

    public int embedDim() {
        return embedDim;
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch_size, num_fields, embed_dim)
        Scalar twoScalar = new Scalar(2.0f);
        Scalar halfScalar = new Scalar(0.5f);

        // sum of squared: (batch, embed_dim)
        Tensor squaredSum = torch.pow(embeddings, twoScalar).sum(1);
        // squared sum: (batch, embed_dim)
        Tensor sumSquared = torch.pow(embeddings.sum(1), twoScalar);
        // interaction: (batch, embed_dim) - NOT summing at the end
        return sumSquared.sub(squaredSum).mul(halfScalar);
    }
}
