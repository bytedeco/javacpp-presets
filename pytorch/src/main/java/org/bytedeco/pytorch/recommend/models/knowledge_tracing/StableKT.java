/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/StableKT.scala
 *
 * StableKT: Stable Knowledge Tracing with ALiBi and Penumbral Attention.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StableKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final EmbeddingImpl qEmbed;
    private final EmbeddingImpl qaEmbed;
    private final CosinePositionalEmbedding posEmb;
    private final List<StableTransformerBlock> blocks = new ArrayList<>();
    private final MLP outMLP;
    private final DropoutImpl dropoutLayer;

    public StableKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 1.0f, 1.0f, 0.2f, DeviceSupport.backend());
    }

    public StableKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            float r,
            float gamma,
            float dropout,
            String device) {
        super("StableKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("q_embed", qEmbed);

        this.qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2 + 2, embedDim));
        register_module("qa_embed", qaEmbed);

        this.posEmb = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_emb", posEmb);

        for (int i = 0; i < numBlocks; i++) {
            StableTransformerBlock block = new StableTransformerBlock(embedDim, numHeads, r, gamma, dropout, device);
            register_module("block_" + i, block);
            blocks.add(block);
        }

        this.outMLP = new MLP(embedDim * 2L, new long[]{(long) embedDim, embedDim / 2L}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("out_mlp", outMLP);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qEmbed.to(dev, false);
            qaEmbed.to(dev, false);
            outMLP.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor cLong = conceptIds.toType(ScalarType.Long);
        Tensor rLong = responses.toType(ScalarType.Long);

        Tensor conceptIdx = cLong.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)))
                .toType(ScalarType.Long);
        Tensor responseIdx = rLong.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)))
                .toType(ScalarType.Long);

        Tensor qaIds = conceptIdx.mul(new Scalar(2)).add(responseIdx);
        Tensor qaEmb = qaEmbed.forward(qaIds);
        Tensor qEmb = qEmbed.forward(conceptIdx);

        Tensor posEnc = posEmb.forward(qaEmb);
        Tensor qaWithPos = qaEmb.add(posEnc);
        Tensor qWithPos = qEmb.add(posEnc);

        Tensor y = qaWithPos;
        Tensor x = qWithPos;
        for (StableTransformerBlock block : blocks) {
            T_TensorTensor_T result = block.forwardPair(x, y);
            x = result.get0();
            y = result.get1();
        }

        Tensor concatQ = torch.cat(new TensorVector(x, qEmb), 2);
        Tensor logits = outMLP.forward(concatQ);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
