/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/ActivationUnit.scala
 * (ActivationUnit + Attention classes)
 *
 * Activation Unit for DIN-style attention. Reference: Alibaba DIN, KDD 2018.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

/**
 * Activation Unit for DIN-style attention.
 * Computes attention weight between target item and historical items.
 * Input: two item embeddings [batch, embed_dim]
 * Output: attention weight [batch, 1]
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ActivationUnit extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int hiddenSize;
    private final String activation;
    private final LinearImpl linear1;
    private final DiceActivation diceAct;
    private final LinearImpl linear2;

    public ActivationUnit(int embedDim) {
        this(embedDim, 36, "dice", DeviceSupport.backend());
    }

    public ActivationUnit(int embedDim, int hiddenSize) {
        this(embedDim, hiddenSize, "dice", DeviceSupport.backend());
    }

    public ActivationUnit(int embedDim, int hiddenSize, String activation, String device) {
        super("ActivationUnit");
        this.embedDim = embedDim;
        this.hiddenSize = hiddenSize;
        this.activation = activation;

        this.linear1 = new LinearImpl(embedDim * 3L, hiddenSize);
        register_module("linear1", linear1);

        this.diceAct = new DiceActivation(hiddenSize);
        register_module("diceAct", diceAct);

        this.linear2 = new LinearImpl(hiddenSize, 1);
        register_module("linear2", linear2);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            linear1.to(dev, false);
            linear2.to(dev, false);
        }
    }

    public int embedDim() {
        return embedDim;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public Tensor forward(Tensor item1, Tensor item2) {
        // item1: [batch, embed_dim], item2: [batch, embed_dim]
        Tensor cross = item1.mul(item2);
        TensorVector vec = new TensorVector();
        vec.push_back(item1);
        vec.push_back(cross);
        vec.push_back(item2);
        Tensor concat = torch.cat(vec, 1);
        Tensor h = linear1.forward(concat);
        Tensor activated = diceAct.forward(h);
        return linear2.forward(activated);
    }
}
