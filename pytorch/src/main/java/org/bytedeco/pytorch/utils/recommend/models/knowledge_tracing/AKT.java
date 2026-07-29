/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/AKT.scala
 *
 * AKT: Attention-based Knowledge Tracing (Pandey & Karypis, KDD 2019).
 * Question emb + Rasch difficulty + two-stage transformer → MLP.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.TransformerLayer;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final int numBlocks;
    private final EmbeddingImpl qEmbed;
    private final EmbeddingImpl qaEmbed;
    private final Tensor qDiff;
    private final Tensor qaDiff;
    private final CosinePositionalEmbedding posEmbed;
    private final List<TransformerLayer> blocks1 = new ArrayList<>();
    private final List<TransformerLayer> blocks2 = new ArrayList<>();
    private final MLP outLayer;
    private final DropoutImpl dropoutLayer;

    public AKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 256, 0.1f, DeviceSupport.backend());
    }

    public AKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            int ffnDim,
            float dropout,
            String device) {
        super("AKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;
        this.numBlocks = numBlocks;

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("q_embed", qEmbed);

        this.qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        register_module("qa_embed", qaEmbed);

        Tensor qd = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.1f));
        if (!"cpu".equals(device)) {
            qd = qd.to(new Device(device), ScalarType.Float);
        }
        this.qDiff = qd;
        register_parameter("q_diff", qDiff);

        Tensor qad = torch.randn(
                new long[]{numConcepts * 2, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.1f));
        if (!"cpu".equals(device)) {
            qad = qad.to(new Device(device), ScalarType.Float);
        }
        this.qaDiff = qad;
        register_parameter("qa_diff", qaDiff);

        this.posEmbed = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_embed", posEmbed);

        for (int i = 0; i < numBlocks; i++) {
            TransformerLayer layer1 = new TransformerLayer(embedDim, numHeads, ffnDim, dropout, device);
            register_module("block1_" + i, layer1);
            blocks1.add(layer1);

            TransformerLayer layer2 = new TransformerLayer(embedDim, numHeads, ffnDim, dropout, device);
            register_module("block2_" + i, layer2);
            blocks2.add(layer2);
        }

        this.outLayer = new MLP(embedDim * 2L, new long[]{(long) embedDim}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("out_layer", outLayer);

        this.dropoutLayer = new DropoutImpl(dropout);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qEmbed.to(dev, false);
            qaEmbed.to(dev, false);
            outLayer.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor conceptIdsLong = conceptIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0L)),
                new ScalarOptional(new Scalar(numConcepts)))
                .toType(ScalarType.Long);
        Tensor responsesLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0L)),
                new ScalarOptional(new Scalar(1L)))
                .toType(ScalarType.Long);

        Tensor interactionRaw = conceptIdsLong.add(responsesLong.mul(new Scalar(numConcepts)));
        long maxInteraction = numConcepts * 2 - 1;
        Tensor interactionIds = interactionRaw.clamp(
                new ScalarOptional(new Scalar(0L)),
                new ScalarOptional(new Scalar(maxInteraction)))
                .toType(ScalarType.Long);

        Tensor qaEmb = qaEmbed.forward(interactionIds);
        Tensor posEnc = posEmbed.forward(qaEmb);
        qaEmb = qaEmb.add(posEnc);

        // Stage 1: encode QA interactions (no causal mask)
        Tensor y = qaEmb;
        for (TransformerLayer block : blocks1) {
            y = block.forward(y, 1);
        }

        Tensor qEmb = qEmbed.forward(conceptIdsLong);

        Tensor qDiffEmb = qDiff.index_select(0, conceptIdsLong.toType(ScalarType.Long).view(-1L))
                .view(batchSize, seqLen, embedDim);
        Tensor qaDiffEmb = qaDiff.index_select(0, interactionIds.toType(ScalarType.Long).view(-1L))
                .view(batchSize, seqLen, embedDim);

        Tensor adjQEmb = qEmb.add(qDiffEmb);
        Tensor adjQaEmb = qaEmb.add(qaDiffEmb);

        // Stage 2: cross-attend with questions (causal mask)
        Tensor x = adjQEmb;
        for (int i = 0; i < numBlocks; i++) {
            x = blocks2.get(i).forward(x, 0);
        }

        Tensor concatQa = torch.cat(new TensorVector(x, adjQaEmb), 2);
        Tensor logits = outLayer.forward(concatQa);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
