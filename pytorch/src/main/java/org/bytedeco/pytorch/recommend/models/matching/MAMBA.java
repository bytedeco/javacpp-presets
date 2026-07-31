/*
 * Ported from torch-rechub-scala: torchrec/models/matching/MAMBA.scala
 *
 * MAMBA: State Space Model for Sequential Recommendation.
 * Reference: Gu & Dao, 2023 — adapted for recommendation.
 */
package org.bytedeco.pytorch.recommend.models.matching;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MAMBA extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final EmbeddingImpl tokenEmbedding;
    private final Tensor clsTensor;
    private final List<MAMBABlock> layers = new ArrayList<>();
    private final LayerNormImpl layerNorm;
    private final DropoutImpl dropoutLayer;
    private final MLP mlp;

    public MAMBA(long vocabSize) {
        this(vocabSize, 64, 16, 2, 50, new long[]{128L, 64L}, 0.1f, DeviceSupport.backend());
    }

    public MAMBA(long vocabSize, int embedDim, int dState, int numLayers, int maxSeqLen,
                 long[] mlpDims, float dropout, String device) {
        super("MAMBA");
        // maxSeqLen kept for API parity
        this.embedDim = embedDim;

        this.tokenEmbedding = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim));
        tokenEmbedding.to(new Device(device), false);
        register_module("tokenEmbedding", tokenEmbedding);

        Tensor cls = torch.zeros(new long[]{1L, 1L, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        cls.fill_(new Scalar(0.02f));
        if (device != null && !"cpu".equals(device)) {
            cls.to(new Device(device), ScalarType.Float);
        }
        register_parameter("clsToken", cls);
        this.clsTensor = cls;

        for (int i = 0; i < numLayers; i++) {
            MAMBABlock layer = new MAMBABlock(embedDim, dState, dropout, device);
            register_module("mamba_" + i, layer);
            layers.add(layer);
        }

        LongVector lnShape = new LongVector(1);
        lnShape.put(0, embedDim);
        this.layerNorm = new LayerNormImpl(lnShape);
        register_module("layerNorm", layerNorm);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        this.mlp = new MLP(embedDim, mlpDims, 1L, "relu", dropout, false, device);
        register_module("mlp", mlp);

        if (device != null && !"cpu".equals(device)) {
            Device dev = new Device(device);
            tokenEmbedding.to(dev, false);
            for (MAMBABlock layer : layers) layer.to(dev, false);
            layerNorm.to(dev, false);
            dropoutLayer.to(dev, false);
            mlp.to(dev, false);
        }
    }

    public Tensor forward(Tensor seqTokens, Tensor positions) {
        // positions kept for API parity (Scala signature includes it)
        int batchSize = (int) seqTokens.size(0);

        Tensor tokenEmb = tokenEmbedding.forward(seqTokens.toType(ScalarType.Long));

        Tensor clsBatched = clsTensor.expand(batchSize, 1L, embedDim);
        TensorVector cVec = new TensorVector();
        cVec.push_back(clsBatched);
        cVec.push_back(tokenEmb);
        Tensor tokenEmbWithCls = torch.cat(cVec, 1);

        Tensor x = dropoutLayer.forward(tokenEmbWithCls);
        for (MAMBABlock layer : layers) {
            x = layer.forward(x).add(x); // residual
        }

        Tensor normed = layerNorm.forward(x);
        Tensor clsRep = normed.select(1, 0);
        return mlp.forward(clsRep);
    }
}
