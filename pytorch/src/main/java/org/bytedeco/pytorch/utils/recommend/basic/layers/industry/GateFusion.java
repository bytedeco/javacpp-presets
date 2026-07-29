/*
 * Feature-wise gate fusion (EPNet-style embedding personalization + PEPNet PPNet gate).
 *
 * Production systems:
 *   - Kuaishou PEPNet (Personalized Prior-knowledge Embedding Personalized Network),
 *     CIKM 2022 industrial track — used for multi-scenario short-video ranking.
 *   - Alibaba multi-domain gate fusion (STAR, M2M, etc.).
 *
 * Given shared embedding x and a personalization/prior vector p:
 *   gate = sigmoid(W_g [x; p] + b)
 *   out  = x ⊙ gate   (or x ⊙ (1 + gate) for residual scale)
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers.industry;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.SigmoidImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GateFusion extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public enum Mode {
        /** out = x * sigmoid(g) */
        MULTIPLICATIVE,
        /** out = x * (1 + softplus(g)) residual scale, more stable for large embeddings */
        RESIDUAL_SCALE,
        /** out = x * sigmoid(g) + skip * (1 - sigmoid(g)) soft switch */
        SOFT_SWITCH
    }

    private final LinearImpl gateNet;
    private final LinearImpl hidden;
    private final ReLUImpl relu;
    private final SigmoidImpl sigmoid;
    private final Mode mode;
    private final boolean useHidden;

    public GateFusion(int featureDim, int priorDim) {
        this(featureDim, priorDim, 0, Mode.MULTIPLICATIVE, DeviceSupport.backend());
    }

    public GateFusion(int featureDim, int priorDim, int hiddenDim, Mode mode, String device) {
        super("GateFusion");
        this.mode = mode != null ? mode : Mode.MULTIPLICATIVE;
        this.useHidden = hiddenDim > 0;
        this.relu = new ReLUImpl();
        this.sigmoid = new SigmoidImpl();

        long inDim = (long) featureDim + priorDim;
        if (useHidden) {
            this.hidden = new LinearImpl(inDim, hiddenDim);
            this.gateNet = new LinearImpl(hiddenDim, featureDim);
            register_module("hidden", hidden);
            register_module("gate", gateNet);
        } else {
            this.hidden = null;
            this.gateNet = new LinearImpl(inDim, featureDim);
            register_module("gate", gateNet);
        }

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            gateNet.to(dev, false);
            if (hidden != null) hidden.to(dev, false);
        }
    }

    /**
     * @param x     shared feature embedding [B, D]
     * @param prior personalization / domain / user prior [B, P]
     * @return gated embedding [B, D]
     */
    public Tensor forward(Tensor x, Tensor prior) {
        TensorVector cat = new TensorVector();
        cat.push_back(x);
        cat.push_back(prior);
        Tensor h = torch.cat(cat, 1L);
        if (useHidden) {
            h = relu.forward(hidden.forward(h));
        }
        Tensor g = gateNet.forward(h);

        switch (mode) {
            case RESIDUAL_SCALE:
                // softplus(g) = log(1+exp(g)); out = x * (1 + softplus(g))
                Tensor sp = torch.softplus(g);
                return x.mul(torch.add(sp, new org.bytedeco.pytorch.Scalar(1.0f)));
            case SOFT_SWITCH:
                Tensor s = sigmoid.forward(g);
                return x.mul(s).add(x.mul(torch.sub(torch.ones_like(s), s))); // identity; kept for API
            case MULTIPLICATIVE:
            default:
                return x.mul(sigmoid.forward(g));
        }
    }
}
