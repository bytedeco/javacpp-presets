/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/FactorizationMachineLayer.scala
 *
 * Full Factorization Machine — 1st-order (linear) + 2nd-order (interaction) terms.
 * Reference: "Factorization Machines" (Rendle, ICDM 2010)
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
 * Full Factorization Machine — 1st-order + 2nd-order terms.
 * Expects pre-embedded inputs (batch, num_fields, embed_dim).
 * No learnable parameters — interactions computed via embeddings.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FactorizationMachineLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final String device;

    public FactorizationMachineLayer() {
        this(8, DeviceSupport.backend());
    }

    public FactorizationMachineLayer(int embedDim) {
        this(embedDim, DeviceSupport.backend());
    }

    public FactorizationMachineLayer(int embedDim, String device) {
        super("FactorizationMachineLayer");
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
        // embeddings: (batch, num_fields, embed_dim)
        // 1st-order: sum of embeddings
        Tensor firstOrder = embeddings.sum(1);  // (batch, embed_dim)

        // 2nd-order: 0.5 * (||sum V||^2 - sum ||V||^2)
        Scalar two = new Scalar(2.0f);
        Scalar half = new Scalar(0.5f);
        Tensor squaredSum = torch.pow(embeddings, two).sum(1);
        Tensor sumSquared = torch.pow(embeddings.sum(1), two);
        Tensor secondOrder = sumSquared.sub(squaredSum).mul(half);

        return firstOrder.add(secondOrder);
    }

    /** Compute scalar FM output (sum over embedding dimension). */
    public Tensor forwardScalar(Tensor embeddings) {
        return forward(embeddings).sum(1).unsqueeze(1);  // (batch, 1)
    }
}
