/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/AUGRU.scala (AUGRU class)
 *
 * AUGRU - Attention Update Gate GRU. Interest evolving layer from DIEN.
 */
package org.bytedeco.pytorch.utils.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AUGRU extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final int embed_dim;
    private final AUGRU_Cell augruCell;
    private final LinearImpl Wa;

    public AUGRU(int embedDim) {
        this(embedDim, DeviceSupport.backend());
    }

    public AUGRU(int embedDim, String device) {
        super("AUGRU");
        this.embed_dim = embedDim;
        this.augruCell = new AUGRU_Cell(embedDim, device);
        register_module("augru_cell", augruCell);
        this.Wa = new LinearImpl(embedDim, embedDim);
        register_module("Wa", Wa);

        if (device != null && !"cpu".equals(device)) {
            this.to(new Device(device), false);
        }
    }

    public T_TensorTensor_T forwardPair(Tensor x, Tensor item, Tensor mask) {
        return forwardPair(x, item, mask, false);
    }

    public T_TensorTensor_T forwardPair(Tensor x, Tensor item, Tensor mask, boolean r) {
        // Compute attention scores
        Tensor waOut = Wa.forward(x);
        Tensor itemUnsq = item.unsqueeze(1);
        Tensor scores = waOut.mul(itemUnsq).sum(2);

        // Apply mask
        Tensor maskedScores = scores.masked_fill(mask.bitwise_not(), new Scalar(Float.NEGATIVE_INFINITY));

        // Softmax
        Tensor attn = torch.softmax(maskedScores, 1);

        // Handle NaN rows
        Tensor nanRows = attn.isnan().any(1);
        if (nanRows.count_nonzero().item().toInt() > 0) {
            Tensor numValid = mask.sum(1).unsqueeze(1);
            attn = torch.where(nanRows.unsqueeze(1), torch.ones_like(attn).div(numValid), attn);
        }

        attn = attn.unsqueeze(2);

        // Initialize hidden state
        int batchSize = (int) x.size(0);
        Tensor h = torch.zeros(new long[]{batchSize, embed_dim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .to(x.device(), ScalarType.Float);

        // Run AUGRU cell step by step
        List<Tensor> outs = new ArrayList<>();
        int seqLen = (int) x.size(1);
        // Mirror Scala: h is not reassigned in the loop (uses initial zero state each step).
        for (int i = 0; i < seqLen; i++) {
            Tensor stepInput = x.select(1, i);
            Tensor stepAttn = attn.select(1, i);
            Tensor newH = augruCell.forward(stepInput, h, stepAttn);
            outs.add(newH.unsqueeze(1));
        }

        TensorVector vec = new TensorVector();
        for (Tensor t : outs) vec.push_back(t);
        Tensor outputTensor = torch.cat(vec, 1);
        return new T_TensorTensor_T(outputTensor, h);
    }
}
