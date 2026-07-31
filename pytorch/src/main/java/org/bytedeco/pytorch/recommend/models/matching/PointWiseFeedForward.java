/*
 * Ported from torch-rechub-scala: torchrec/models/matching/SASRec.scala (PointWiseFeedForward)
 *
 * SASRec's per-position feed-forward: Conv1d→ReLU→Conv1d with residual.
 * Module.to(Device) intentionally skipped for Conv1dImpl (bytedeco crash risk).
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PointWiseFeedForward extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final float dropout;
    private final Conv1dImpl conv1;
    private final Conv1dImpl conv2;

    public PointWiseFeedForward(int hiddenDim, int ffnDim, float dropout) {
        this(hiddenDim, ffnDim, dropout, DeviceSupport.backend());
    }

    public PointWiseFeedForward(int hiddenDim, int ffnDim, float dropout, String device) {
        super("PointWiseFeedForward");
        this.dropout = dropout;

        LongPointer k1 = new LongPointer(new long[]{1L});
        this.conv1 = new Conv1dImpl(new Conv1dOptions(hiddenDim, ffnDim, k1));
        register_module("conv1", conv1);

        LongPointer k2 = new LongPointer(new long[]{1L});
        this.conv2 = new Conv1dImpl(new Conv1dOptions(ffnDim, hiddenDim, k2));
        register_module("conv2", conv2);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, len, hidden) → (batch, hidden, len) for Conv1d
        Tensor xt = x.transpose(-1L, -2L);
        Tensor h = conv1.forward(xt);
        Tensor hDrop = torch.dropout(h.relu(), dropout, false);
        Tensor h2 = conv2.forward(hDrop);
        Tensor h2Drop = torch.dropout(h2, dropout, false);
        return h2Drop.transpose(-1L, -2L).add(x);
    }
}
