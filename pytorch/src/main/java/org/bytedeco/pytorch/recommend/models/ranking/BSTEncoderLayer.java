/*
 * Ported from torch-rechub-scala: torchrec/models/ranking/BST.scala (BSTEncoderLayer)
 *
 * BST Encoder Layer - Transformer Encoder Layer with LeakyReLU activation.
 * Note: BST.forward currently bypasses the full transformer path (mirrors Scala).
 */
package org.bytedeco.pytorch.recommend.models.ranking;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.recommend.basic.layers.RecLeakyReLU;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BSTEncoderLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int itemDim;
    private final int numHeads;
    private final int headDim;
    private final float dropout;
    private final LinearImpl attnLinear;
    private final LinearImpl attnOutProj;
    private final LinearImpl ffnLinear1;
    private final LinearImpl ffnLinear2;
    private final LayerNormImpl norm1;
    private final LayerNormImpl norm2;
    private final RecLeakyReLU leakyReLU;

    public BSTEncoderLayer(int itemDim, int numHeads, float dropout, String device) {
        super("BSTEncoderLayer");
        this.itemDim = itemDim;
        this.numHeads = numHeads;
        this.headDim = itemDim / numHeads;
        this.dropout = dropout;
        Device targetDevice = new Device(device);

        this.attnLinear = new LinearImpl(itemDim, 3L * itemDim);
        this.attnOutProj = new LinearImpl(itemDim, itemDim);
        this.ffnLinear1 = new LinearImpl(itemDim, itemDim * 4L);
        this.ffnLinear2 = new LinearImpl(itemDim * 4L, itemDim);

        LongVector n1 = new LongVector(1);
        n1.put(0, itemDim);
        this.norm1 = new LayerNormImpl(n1);
        LongVector n2 = new LongVector(1);
        n2.put(0, itemDim);
        this.norm2 = new LayerNormImpl(n2);

        this.leakyReLU = new RecLeakyReLU();

        register_module("attnLinear", attnLinear);
        register_module("attnOutProj", attnOutProj);
        register_module("ffnLinear1", ffnLinear1);
        register_module("ffnLinear2", ffnLinear2);
        register_module("norm1", norm1);
        register_module("norm2", norm2);
        register_module("leakyReLU", leakyReLU);

        for (Module m : new Module[]{attnLinear, attnOutProj, ffnLinear1, ffnLinear2, norm1, norm2}) {
            m.to(targetDevice, false);
        }
    }

    public Tensor forward(Tensor x, Tensor keyPaddingMask) {
        int bs = (int) x.size(0);
        int sl = (int) x.size(1);

        Tensor nx = norm1.forward(x);

        Tensor qkv = attnLinear.forward(nx).view(bs, sl, 3, numHeads, headDim).transpose(1, 2);
        Tensor q = qkv.select(2, 0);
        Tensor k = qkv.select(2, 1);
        Tensor v = qkv.select(2, 2);

        float scale = 1.0f / (float) Math.sqrt(headDim);
        Tensor attnScores = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale));

        Tensor attnWeights;
        if (keyPaddingMask != null && keyPaddingMask.numel() > 0) {
            Tensor expandedMask = keyPaddingMask.unsqueeze(1).unsqueeze(2);
            Tensor maskedScores = attnScores.masked_fill(expandedMask, new Scalar(Double.NEGATIVE_INFINITY));
            attnWeights = maskedScores.softmax(-1);
        } else {
            attnWeights = attnScores.softmax(-1);
        }

        Tensor droppedAttn = torch.dropout(attnWeights, dropout, is_training());
        Tensor attnOut = torch.matmul(droppedAttn, v);
        Tensor attnOutPermuted = attnOut.transpose(1, 2).contiguous().view(bs, sl, itemDim);
        Tensor attnProjected = attnOutProj.forward(attnOutPermuted);

        Tensor residual1 = x.add(attnProjected);

        Tensor nx2 = norm2.forward(residual1);
        Tensor ffnHidden = leakyReLU.forward(ffnLinear1.forward(nx2));
        Tensor droppedFfn = torch.dropout(ffnHidden, dropout, is_training());
        Tensor ffnOut = ffnLinear2.forward(droppedFfn);

        return residual1.add(ffnOut);
    }
}
