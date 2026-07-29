/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/Memory.scala
 *
 * Erase-and-Add gate used by DKVMN, GKT.
 * Memory update: M_new = M * (1 - attn * erase_gate) + attn * add_gate
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class EraseAddGate extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl erase;
    private final LinearImpl add;
    private final Tensor weight;

    public EraseAddGate(int dim) {
        this(dim, DeviceSupport.backend());
    }

    public EraseAddGate(int dim, String device) {
        super("EraseAddGate");
        this.erase = new LinearImpl(dim, dim);
        this.add = new LinearImpl(dim, dim);

        Tensor w = torch.ones(
                new long[]{dim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        this.weight = w;
        register_parameter("weight", weight);

        register_module("erase", erase);
        register_module("add", add);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            erase.to(dev, false);
            add.to(dev, false);
        }
    }

    /**
     * @param x           (batch, seq, dim) or (batch, dim)
     * @param eraseVector (batch, seq, dim) or (batch, dim)
     * @param addVector   (batch, seq, dim) or (batch, dim)
     * @param attention   (batch, seq) or (batch,)
     */
    public Tensor forward(Tensor x, Tensor eraseVector, Tensor addVector, Tensor attention) {
        Tensor eraseGate = torch.sigmoid(erase.forward(eraseVector));
        Tensor addGate = torch.tanh(add.forward(addVector));

        Tensor expandedAttn = attention.dim() == 2L
                ? attention.unsqueeze(2)
                : attention.unsqueeze(1);

        // Memory update: M_new = M * (1 - attn * erase_gate) + attn * add_gate
        // Scala: x.mul(x.neg().add(1).mul(expandedAttn.unsqueeze(2).mul(eraseGate)))
        // Note: the Scala expression uses x.neg().add(1) which is (1 - x), then multiplies —
        // we mirror it exactly.
        Tensor erased = x.mul(x.neg().add(new Scalar(1.0)).mul(expandedAttn.unsqueeze(2).mul(eraseGate)));
        return erased.add(expandedAttn.unsqueeze(2).mul(addGate));
    }
}
