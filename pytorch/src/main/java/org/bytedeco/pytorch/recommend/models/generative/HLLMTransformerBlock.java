/*
 * Ported from torch-rechub-scala: torchrec/models/generative/HLLM.scala (HLLMTransformerBlock)
 *
 * Single transformer block with multi-head self-attention + FFN residuals.
 * Used by HLLM generative recommender.
 */
package org.bytedeco.pytorch.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class HLLMTransformerBlock extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int dModel;
    private final int nHeads;
    private final int headDim;
    private final float scale;
    private final LinearImpl W_Q;
    private final LinearImpl W_K;
    private final LinearImpl W_V;
    private final LinearImpl W_O;
    private final LinearImpl ffnLinear1;
    private final ReLUImpl ffnRelu;
    private final LinearImpl ffnLinear2;
    private final SequentialImpl ffn;
    private final LayerNormImpl norm1;
    private final LayerNormImpl norm2;
    private final DropoutImpl attnDropout;

    public HLLMTransformerBlock() {
        this(512, 8, 0.1f, DeviceSupport.backend());
    }

    public HLLMTransformerBlock(int dModel, int nHeads, float dropout, String device) {
        super("HLLMTransformerBlock");
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException(
                    "dModel (" + dModel + ") must be divisible by nHeads (" + nHeads + ")");
        }
        this.dModel = dModel;
        this.nHeads = nHeads;
        this.headDim = dModel / nHeads;
        this.scale = (float) Math.pow(headDim, -0.5);
        Device targetDevice = new Device(device);

        this.W_Q = new LinearImpl(dModel, dModel);
        this.W_Q.to(targetDevice, false);
        register_module("W_Q", W_Q);

        this.W_K = new LinearImpl(dModel, dModel);
        this.W_K.to(targetDevice, false);
        register_module("W_K", W_K);

        this.W_V = new LinearImpl(dModel, dModel);
        this.W_V.to(targetDevice, false);
        register_module("W_V", W_V);

        this.W_O = new LinearImpl(dModel, dModel);
        this.W_O.to(targetDevice, false);
        register_module("W_O", W_O);

        int ffnHidden = 4 * dModel;
        // Strong refs to avoid GC; Dropout layers intentionally not pushed into Sequential
        // (mirrors Scala comment: push_back of DropoutImpl can crash).
        this.ffnLinear1 = new LinearImpl(dModel, ffnHidden);
        this.ffnRelu = new ReLUImpl();
        this.ffnLinear2 = new LinearImpl(ffnHidden, dModel);

        SequentialImpl seq = new SequentialImpl();
        seq.push_back("ffn_lin1", ffnLinear1);
        seq.push_back("ffn_relu", ffnRelu);
        seq.push_back("ffn_lin2", ffnLinear2);
        seq.to(targetDevice, false);
        this.ffn = seq;
        register_module("ffn", ffn);

        LongVector lnShape1 = new LongVector(1);
        lnShape1.put(0, dModel);
        this.norm1 = new LayerNormImpl(lnShape1);
        this.norm1.to(targetDevice, false);
        register_module("norm1", norm1);

        LongVector lnShape2 = new LongVector(1);
        lnShape2.put(0, dModel);
        this.norm2 = new LayerNormImpl(lnShape2);
        this.norm2.to(targetDevice, false);
        register_module("norm2", norm2);

        this.attnDropout = new DropoutImpl(dropout);
        register_module("attnDropout", attnDropout);
    }

    public void initWeights() {
        LinearImpl[] linears = {W_Q, W_K, W_V, W_O, ffnLinear1, ffnLinear2};
        for (LinearImpl linear : linears) {
            Tensor weight = linear.weight();
            if (weight.dim() > 1) {
                torch.xavier_uniform_(weight);
            }
            Tensor bias = linear.bias();
            if (bias != null && !bias.isNull()) {
                torch.constant_(bias, new Scalar(0.0f));
            }
        }
    }

    public Tensor forward(Tensor x, Tensor relPosBias) {
        int batchSize = (int) x.size(0);
        int seqLen = (int) x.size(1);

        Tensor residual = x;
        Tensor normedX = norm1.forward(x);

        Tensor Q = W_Q.forward(normedX);
        Tensor K = W_K.forward(normedX);
        Tensor V = W_V.forward(normedX);

        Tensor QReshaped = Q.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2);
        Tensor KReshaped = K.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2);
        Tensor VReshaped = V.view(batchSize, seqLen, nHeads, headDim).transpose(1, 2);

        Tensor scores = torch.matmul(QReshaped, KReshaped.transpose(-2, -1)).mul(new Scalar(scale));

        Tensor causalMask = torch.tril(
                torch.ones(new long[]{seqLen, seqLen},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Bool)))
                        .to(x.device(), ScalarType.Bool));
        Tensor validMask = causalMask.unsqueeze(0).unsqueeze(0);

        Tensor maskedScores = scores.masked_fill(
                torch.logical_not(validMask),
                new Scalar(Double.NEGATIVE_INFINITY));

        Tensor scoresWithBias = relPosBias != null
                ? maskedScores.add(relPosBias)
                : maskedScores;

        Tensor attnWeights = attnDropout.forward(torch.silu(scoresWithBias).softmax(-1));

        Tensor attnOutput = torch.matmul(attnWeights, VReshaped);
        Tensor attnOutputReshaped = attnOutput.transpose(1, 2).contiguous()
                .view(batchSize, seqLen, dModel);
        Tensor attnOutProj = W_O.forward(attnOutputReshaped);
        Tensor attnOutFinal = residual.add(attnOutProj);

        Tensor residual2 = attnOutFinal;
        Tensor normed2 = norm2.forward(attnOutFinal);
        Tensor ffnOut = ffn.forward(normed2);
        return residual2.add(ffnOut);
    }

    @Override
    public Tensor forward(Tensor x) {
        return forward(x, (Tensor) null);
    }
}
