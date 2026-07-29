/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/DIN.scala
 * (ActivationUnit class — ranking-local, different from basic.layers.ActivationUnit)
 *
 * DIN per-position target attention sublayer.
 * MLP(4 * emb_dim, dims=[36], activation="dice") on
 * [target, history, target - history, target * history].
 */
package org.bytedeco.pytorch.utils.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

/**
 * DIN Activation Unit (ranking package).
 * Input history: (batch, seqLen, embedDim), target: (batch, embedDim)
 * Output: (batch, embedDim) — softmax-weighted sum over the sequence.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DINActivationUnit extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final MLP attention;

    public DINActivationUnit(int embedDim) {
        this(embedDim, 36, DeviceSupport.backend());
    }

    public DINActivationUnit(int embedDim, int hiddenUnits, String device) {
        super("ActivationUnit");
        this.embedDim = embedDim;
        // Scala uses activation "dice"; MLP falls back to ReLU for unknown acts (same as Scala MLP).
        this.attention = new MLP(4L * embedDim, new long[]{hiddenUnits}, 1L, "dice", 0.0f, false, device);
        register_module("attention", attention);
    }

    public Tensor forward(Tensor history, Tensor target) {
        long seqLen = history.size(1);
        Tensor targetExp = target.unsqueeze(1L).expand(-1L, seqLen, -1L);
        TensorVector vec = new TensorVector();
        vec.push_back(targetExp);
        vec.push_back(history);
        vec.push_back(targetExp.sub(history));
        vec.push_back(targetExp.mul(history));
        Tensor attInput = torch.cat(vec, 2L); // (batch, seqLen, 4*embedDim)
        Tensor flat = attInput.view(-1L, 4L * embedDim);
        Tensor scores = attention.forward(flat).view(-1L, seqLen); // (batch, seqLen)
        Tensor weights = scores.softmax(1L).unsqueeze(-1L);        // (batch, seqLen, 1)
        return weights.mul(history).sum(1L);                        // (batch, embedDim)
    }
}
