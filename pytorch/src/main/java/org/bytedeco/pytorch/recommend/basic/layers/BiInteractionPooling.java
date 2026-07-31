/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/BiInteractionPooling.scala
 *
 * Bi-Interaction Pooling layer for NFM.
 * Reference: "Neural Factorization Machines for Sparse Predictive Analytics" (SIGIR 2017)
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
 * Bi-Interaction Pooling layer for NFM.
 * Combines pairwise feature interactions via element-wise product,
 * then pools them using sum (or mean) aggregation.
 *
 * <p>Implemented as: 0.5 * (sum(S)^2 - sum(S^2)) where S = sum_i V_i * x_i
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BiInteractionPooling extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final String device;

    public BiInteractionPooling() {
        this(DeviceSupport.backend());
    }

    public BiInteractionPooling(String device) {
        super("BiInteractionPooling");
        this.device = device;
    }

    public String device() {
        return device;
    }

    @Override
    public Tensor forward(Tensor embeddings) {
        // embeddings: (batch, num_fields, embed_dim)
        Tensor sumOfEmbeddings = embeddings.sum(1);  // (batch, embed_dim)
        Tensor squaredSum = torch.pow(embeddings, new Scalar(2.0f)).sum(1);  // (batch, embed_dim)
        Tensor sumSquared = torch.pow(sumOfEmbeddings, new Scalar(2.0f));  // (batch, embed_dim)
        Scalar half = new Scalar(0.5f);
        return sumSquared.sub(squaredSum).mul(half);
    }
}
