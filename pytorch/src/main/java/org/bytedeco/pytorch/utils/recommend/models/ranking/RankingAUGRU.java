/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DIEN.scala (AUGRU wrapper)
 *
 * Ranking-local AUGRU: forward(history, target, mask) returns (allSteps, lastHidden).
 * Named RankingAUGRU to avoid clash with basic.layers.AUGRU.
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RankingAUGRU extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final RankingAUGRUCell cell;

    public RankingAUGRU(int embedDim) {
        this(embedDim, DeviceSupport.backend());
    }

    public RankingAUGRU(int embedDim, String device) {
        super("AUGRU");
        this.embedDim = embedDim;
        this.cell = new RankingAUGRUCell(embedDim);
        register_module("cell", cell);
    }

    /**
     * Run AUGRU over sequence.
     * @return array of [allSteps, lastHidden]
     */
    public Tensor[] run(Tensor history, Tensor target, Tensor mask) {
        // simplified attention
        Tensor scores = history.mul(target.unsqueeze(1L)).sum(-1L);
        Tensor m = mask.gt(new Scalar(0L)).toType(ScalarType.Bool);
        Tensor masked = scores.masked_fill(m.logical_not(), new Scalar(Double.NEGATIVE_INFINITY));
        Tensor attn = masked.softmax(1L).unsqueeze(-1L);

        long batch = history.size(0);
        long time = history.size(1);
        Tensor h = torch.zeros(new long[]{batch, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        List<Tensor> outs = new ArrayList<>();
        for (long t = 0; t < time; t++) {
            h = cell.forward(history.select(1L, t), h, attn.select(1L, t));
            outs.add(h.unsqueeze(1L));
        }
        TensorVector vec = new TensorVector();
        for (Tensor o : outs) vec.push_back(o);
        Tensor allSteps = torch.cat(vec, 1L);
        return new Tensor[]{allSteps, h};
    }
}
