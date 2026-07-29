/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/AKT.scala (SparseKT)
 *
 * SparseKT: Knowledge Tracing with Sparse Attention (SimpleKT architecture + sparseRatio param).
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.TransformerLayer;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SparseKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final float sparseRatio;
    private final EmbeddingImpl qEmbed;
    private final EmbeddingImpl qaEmbed;
    private final CosinePositionalEmbedding posEmbed;
    private final List<TransformerLayer> blocks = new ArrayList<>();
    private final MLP outLayer;

    public SparseKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 256, 0.1f, 0.8f, DeviceSupport.backend());
    }

    public SparseKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            int ffnDim,
            float dropout,
            float sparseRatio,
            String device) {
        super("SparseKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;
        this.sparseRatio = sparseRatio;

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        this.qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        register_module("q_embed", qEmbed);
        register_module("qa_embed", qaEmbed);

        this.posEmbed = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_embed", posEmbed);

        for (int i = 0; i < numBlocks; i++) {
            TransformerLayer layer = new TransformerLayer(embedDim, numHeads, ffnDim, dropout, device);
            register_module("block_" + i, layer);
            blocks.add(layer);
        }

        this.outLayer = new MLP(embedDim * 2L, new long[]{(long) embedDim}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("out_layer", outLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qEmbed.to(dev, false);
            qaEmbed.to(dev, false);
            outLayer.to(dev, false);
        }
    }

    public float sparseRatio() {
        return sparseRatio;
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor conceptIdsLong = conceptIds.toType(ScalarType.Long);
        Tensor responsesLong = responses.toType(ScalarType.Long);
        long maxInteractionId = numConcepts * 2 - 1;
        Tensor interactionIdsBase = responsesLong.mul(new Scalar((double) numConcepts));
        Tensor interactionIdsRaw = conceptIdsLong.add(interactionIdsBase);
        Tensor interactionIdsClamped = interactionIdsRaw.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(maxInteractionId)))
                .toType(ScalarType.Long);

        Tensor qaEmb = qaEmbed.forward(interactionIdsClamped);
        Tensor posEnc = posEmbed.forward(qaEmb);
        qaEmb = qaEmb.add(posEnc);

        Tensor qIdsClamped = conceptIdsLong.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(maxInteractionId)))
                .toType(ScalarType.Long);
        Tensor qEmb = qEmbed.forward(qIdsClamped);
        Tensor posEncQ = posEmbed.forward(qEmb);
        Tensor adjQEmb = qEmb.add(posEncQ);

        Tensor x = adjQEmb;
        for (TransformerLayer block : blocks) {
            x = block.forward(x, 0);
        }

        Tensor concatQa = torch.cat(new TensorVector(x, adjQEmb), 2);
        Tensor logits = outLayer.forward(concatQa);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
