/*
 * Ported from torch-rechub-scala: torchrec/models/generative/LLM4Rec.scala
 *
 * LLM4Rec: Transformer Encoder for Sequential Recommendation.
 * Note: encoder layers and MLP are intentionally NOT register_module'd
 * (mirrors Scala workaround for Director crash); train() is overridden to
 * manually dispatch training mode.
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LLM4Rec extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final float dropout;
    private final Device targetDevice;
    private final EmbeddingImpl tokenEmbedding;
    private final Tensor clsTensor;
    private final EmbeddingImpl positionEmbedding;
    private final LayerNormImpl preNorm;
    private final List<LLM4RecEncoderLayer> encoderLayers = new ArrayList<>();
    private final MLP mlp;
    private boolean isTrainingMode = true;

    public LLM4Rec(long vocabSize) {
        this(vocabSize, 64, 4, 3, 50, new long[]{256L, 128L}, 0.1f, true, DeviceSupport.backend());
    }

    public LLM4Rec(
            long vocabSize,
            int embedDim,
            int numHeads,
            int numLayers,
            int maxSeqLen,
            long[] mlpDims,
            float dropout,
            boolean usePosEncoding,
            String device) {
        super("LLM4Rec");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.dropout = dropout;
        this.targetDevice = new Device(device);
        long ffDim = embedDim * 4L;

        this.tokenEmbedding = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim));
        this.tokenEmbedding.to(targetDevice, false);
        register_module("tokenEmbedding", tokenEmbedding);

        this.clsTensor = torch.zeros(
                new long[]{1L, 1L, embedDim},
                new TensorOptions()
                        .dtype(new ScalarTypeOptional(ScalarType.Float))
                        .device(new DeviceOptional(targetDevice)));
        this.clsTensor.fill_(new Scalar(0.02f));
        register_parameter("clsToken", clsTensor);

        this.positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxSeqLen + 1L, embedDim));
        this.positionEmbedding.to(targetDevice, false);
        register_module("positionEmbedding", positionEmbedding);

        LongVector lnShape = new LongVector(new long[]{(long) embedDim});
        this.preNorm = new LayerNormImpl(lnShape);
        this.preNorm.to(targetDevice, false);
        register_module("preNorm", preNorm);

        // Custom encoder layers: intentionally NOT register_module'd (matches Scala)
        for (int i = 0; i < numLayers; i++) {
            encoderLayers.add(new LLM4RecEncoderLayer(embedDim, numHeads, ffDim, dropout, device));
        }

        // MLP intentionally not register_module'd (matches Scala)
        this.mlp = new MLP(embedDim, mlpDims, 1L, "relu", dropout, false, false, true, device);
    }

    @Override
    public void train(boolean on) {
        isTrainingMode = on;
        tokenEmbedding.train(on);
        positionEmbedding.train(on);
        preNorm.train(on);
        for (LLM4RecEncoderLayer layer : encoderLayers) {
            layer.train(on);
        }
        try {
            mlp.train(on);
        } catch (Throwable ignored) {
        }
    }

    public Tensor forward(Tensor seqTokens, Tensor positions) {
        long batchSize = seqTokens.size(0);
        var dev = seqTokens.device();

        Tensor tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long));
        Tensor clsBatched = clsTensor.expand(new long[]{batchSize, 1L, embedDim});
        Tensor tokenEmbWithCls = torch.cat(new TensorVector(clsBatched, tokenEmb), 1);

        Tensor clsPos = torch.zeros(
                new long[]{batchSize, 1L},
                new TensorOptions()
                        .dtype(new ScalarTypeOptional(ScalarType.Long))
                        .device(new DeviceOptional(dev)));
        Tensor posWithCls = torch.cat(new TensorVector(clsPos, positions.toType(ScalarType.Long)), 1);
        Tensor posEmb = positionEmbedding.forward(posWithCls);

        Tensor x = torch.dropout(tokenEmbWithCls.add(posEmb), (double) dropout, isTrainingMode);

        for (LLM4RecEncoderLayer layer : encoderLayers) {
            Tensor out = layer.forward(x);
            x = x.add(out);
        }

        Tensor clsRep = preNorm.forward(x).select(1, 0);
        return mlp.forward(clsRep);
    }
}
