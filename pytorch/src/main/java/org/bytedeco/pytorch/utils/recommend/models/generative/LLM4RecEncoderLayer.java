/*
 * Ported from torch-rechub-scala: torchrec/models/generative/LLM4Rec.scala (LLM4RecEncoderLayer)
 *
 * Transformer encoder layer for LLM4Rec: multi-head self-attention + FFN with residuals.
 * Uses functional torch.dropout instead of DropoutImpl.
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

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
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LLM4RecEncoderLayer extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final float dropout;
    private final LinearImpl attnLinear;
    private final LinearImpl attnOutProj;
    private final LinearImpl ffnLinear1;
    private final LinearImpl ffnLinear2;
    private final LayerNormImpl norm1;
    private final LayerNormImpl norm2;
    private boolean isTrainingMode = true;

    public LLM4RecEncoderLayer(int embedDim, int numHeads, long ffDim, float dropout, String device) {
        super("LLM4RecEncoderLayer");
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.dropout = dropout;
        Device targetDevice = new Device(device);

        this.attnLinear = new LinearImpl(embedDim, 3L * embedDim);
        this.attnOutProj = new LinearImpl(embedDim, embedDim);
        this.ffnLinear1 = new LinearImpl(embedDim, ffDim);
        this.ffnLinear2 = new LinearImpl(ffDim, embedDim);
        this.norm1 = new LayerNormImpl(new LongVector(new long[]{(long) embedDim}));
        this.norm2 = new LayerNormImpl(new LongVector(new long[]{(long) embedDim}));

        attnLinear.to(targetDevice, false);
        attnOutProj.to(targetDevice, false);
        ffnLinear1.to(targetDevice, false);
        ffnLinear2.to(targetDevice, false);
        norm1.to(targetDevice, false);
        norm2.to(targetDevice, false);
    }

    public LLM4RecEncoderLayer(int embedDim, int numHeads, long ffDim, float dropout) {
        this(embedDim, numHeads, ffDim, dropout, DeviceSupport.backend());
    }

    @Override
    public void train(boolean on) {
        isTrainingMode = on;
        attnLinear.train(on);
        attnOutProj.train(on);
        ffnLinear1.train(on);
        ffnLinear2.train(on);
        norm1.train(on);
        norm2.train(on);
    }

    @Override
    public Tensor forward(Tensor x) {
        long bs = x.size(0);
        long sl = x.size(1);

        // Attention
        Tensor nx = norm1.forward(x);
        // reshape instead of view for non-contiguous safety
        Tensor qkv = attnLinear.forward(nx).reshape(bs, sl, 3, numHeads, headDim).permute(3, 0, 1, 4, 2);
        Tensor q = qkv.select(4, 0);
        Tensor k = qkv.select(4, 1);
        Tensor v = qkv.select(4, 2);

        float scale = 1.0f / (float) Math.sqrt(headDim);
        Tensor attn = torch.matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale)).softmax(-1);

        Tensor droppedAttn = torch.dropout(attn, (double) dropout, isTrainingMode);
        Tensor valOut = torch.matmul(droppedAttn, v);

        Tensor attnOut = attnOutProj.forward(valOut.permute(1, 2, 0, 3).reshape(bs, sl, embedDim));
        Tensor residual1 = x.add(attnOut);

        // FFN
        Tensor nx2 = norm2.forward(residual1);
        Tensor ffnHidden = ffnLinear1.forward(nx2).relu();
        Tensor droppedFfn = torch.dropout(ffnHidden, (double) dropout, isTrainingMode);
        Tensor ffn = ffnLinear2.forward(droppedFfn);

        return residual1.add(ffn);
    }
}
