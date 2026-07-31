/*
 * Ported from torch-rechub-scala: torchrec/models/multi_task/AITM.scala (AttentionLayer)
 *
 * AttentionLayer for AITM information transfer (KDD'2021).
 * Q/K/V are Linear(dim, dim, bias=False).
 * scores = softmax(sum(Q*K, -1) / sqrt(dim)); output = sum(unsqueeze(scores,-1)*V, dim=1)
 * Shape: Input (batch, 2, dim) → Output (batch, dim)
 */
package org.bytedeco.pytorch.recommend.models.multi_task;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AttentionLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int dim;
    private final LinearImpl qLayer;
    private final LinearImpl kLayer;
    private final LinearImpl vLayer;

    public AttentionLayer(int dim) {
        this(dim, DeviceSupport.backend());
    }

    public AttentionLayer(int dim, String device) {
        super("AttentionLayer");
        this.dim = dim;

        // bias=false; Module.to() intentionally skipped (matches Scala note on bytedeco crash)
        this.qLayer = new LinearImpl(new LinearOptions(dim, dim).bias(false));
        register_module("q_layer", qLayer);
        this.kLayer = new LinearImpl(new LinearOptions(dim, dim).bias(false));
        register_module("k_layer", kLayer);
        this.vLayer = new LinearImpl(new LinearOptions(dim, dim).bias(false));
        register_module("v_layer", vLayer);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor Q = qLayer.forward(x);
        Tensor K = kLayer.forward(x);
        Tensor V = vLayer.forward(x);

        Scalar scale = new Scalar((float) Math.sqrt(dim));
        Tensor a = torch.mul(Q, K).sum(-1).div(scale).softmax(1);
        return torch.mul(a.unsqueeze(-1), V).sum(1L);
    }
}
