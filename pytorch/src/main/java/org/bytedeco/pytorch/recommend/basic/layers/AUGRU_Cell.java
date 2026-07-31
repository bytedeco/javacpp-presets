/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/AUGRU.scala
 *
 * AUGRU Cell + AUGRU (Attention Update Gate GRU). Reference: DIEN paper, AAAI 2019.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * AUGRU Cell - Attention Update Gate GRU Cell.
 * u_hat = a * u; h_new = (1 - u_hat) * h + u_hat * h_hat
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AUGRU_Cell extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl Wu, Uu, Wr, Ur, Wh, Uh;
    private final Tensor bu, br, bh;

    public AUGRU_Cell(int embedDim) {
        this(embedDim, DeviceSupport.backend());
    }

    public AUGRU_Cell(int embedDim, String device) {
        super("AUGRU_Cell");
        this.Wu = new LinearImpl(embedDim, embedDim);
        this.Uu = new LinearImpl(embedDim, embedDim);
        this.Wr = new LinearImpl(embedDim, embedDim);
        this.Ur = new LinearImpl(embedDim, embedDim);
        this.Wh = new LinearImpl(embedDim, embedDim);
        this.Uh = new LinearImpl(embedDim, embedDim);

        TensorOptions opts = new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float));
        this.bu = torch.zeros(new long[]{embedDim}, opts);
        this.br = torch.zeros(new long[]{embedDim}, opts);
        this.bh = torch.zeros(new long[]{embedDim}, opts);

        register_module("Wu", Wu);
        register_module("Uu", Uu);
        register_module("Wr", Wr);
        register_module("Ur", Ur);
        register_module("Wh", Wh);
        register_module("Uh", Uh);
        register_buffer("bu", bu);
        register_buffer("br", br);
        register_buffer("bh", bh);

        if (device != null && !"cpu".equals(device)) {
            this.to(new Device(device), false);
        }
    }

    public Tensor forward(Tensor x, Tensor h1, Tensor a) {
        // x: (batch, embed_dim), h1: (batch, embed_dim), a: (batch, 1)
        Tensor u = torch.sigmoid(Wu.forward(x).add(Uu.forward(h1)).add(bu));
        Tensor r = torch.sigmoid(Wr.forward(x).add(Ur.forward(h1)).add(br));
        Tensor hUh = Uh.forward(h1).mul(r);
        Tensor hHat = torch.tanh(Wh.forward(x).add(hUh).add(bh));
        Tensor uHat = a.mul(u);
        Tensor oneMinusUh = torch.ones_like(uHat).sub(uHat);
        return oneMinusUh.mul(h1).add(uHat.mul(hHat));
    }
}
