/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/FM.scala
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * Factorization Machine for 2nd-order feature interactions.
 * FM: y = sum_i w_i * x_i + sum_i sum_j&lt;i &lt;w_i, w_j&gt; * x_i * x_j
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final String device;

    public FM() {
        this(8, DeviceSupport.backend());
    }

    public FM(int embedDim) {
        this(embedDim, DeviceSupport.backend());
    }

    public FM(int embedDim, String device) {
        super("FM");
        this.embedDim = embedDim;
        this.device = device;
    }

    public int embedDim() {
        return embedDim;
    }

    public String device() {
        return device;
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch_size, num_fields, embed_dim)
        // First order: sum of embeddings
        Tensor firstOrder = embeddings.sum(1); // (batch_size, embed_dim)

        // Second order: sum of squared - squared sum
        Scalar twoScalar = new Scalar(2.0f);
        Tensor squaredSum = torch.pow(embeddings, twoScalar).sum(1);
        Tensor sumSquared = torch.pow(embeddings.sum(1), twoScalar);

        // Interaction: 0.5 * (sum^2 - squared_sum)
        Scalar halfScalar = new Scalar(0.5f);
        Tensor interactions = sumSquared.sub(squaredSum).mul(halfScalar);

        // FM output: sum of first order + sum of interactions
        return firstOrder.add(interactions).sum(1).unsqueeze(1);
    }
}
