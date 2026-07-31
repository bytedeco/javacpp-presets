/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/PromptKT.scala
 *
 * PromptKT: Prompt-based Knowledge Tracing.
 * Knowledge prompts + concept embeddings → Transformer → MLP.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

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
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.TransformerLayer;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PromptKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final int embedDim;
    private final Tensor datasetPrompt;
    private final EmbeddingImpl qEmbed;
    private final Tensor cEmbed;
    private final EmbeddingImpl qaEmbed;
    private final MLP promptMLP;
    private final CosinePositionalEmbedding posEmb;
    private final List<TransformerLayer> blocks = new ArrayList<>();
    private final MLP outMLP;
    private final DropoutImpl dropoutLayer;

    public PromptKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 0.2f, DeviceSupport.backend());
    }

    public PromptKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            float dropout,
            String device) {
        super("PromptKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;
        this.embedDim = embedDim;

        Tensor dp = torch.randn(
                new long[]{20L, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.datasetPrompt = dp;
        register_parameter("dataset_prompt", datasetPrompt);

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("q_embed", qEmbed);

        Tensor c = torch.randn(
                new long[]{numConcepts + 1, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                .mul(new Scalar(0.01f));
        this.cEmbed = c;
        register_parameter("c_embed", cEmbed);

        this.qaEmbed = new EmbeddingImpl(new EmbeddingOptions(2, embedDim));
        register_module("qa_embed", qaEmbed);

        this.promptMLP = new MLP(embedDim, new long[]{(long) embedDim, embedDim}, embedDim, "relu", dropout,
                false, false, true, device);
        register_module("prompt_mlp", promptMLP);

        this.posEmb = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_emb", posEmb);

        for (int i = 0; i < numBlocks; i++) {
            TransformerLayer block = new TransformerLayer(embedDim, numHeads, embedDim * 4, dropout, device);
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
            promptMLP.to(dev, false);
            outMLP.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        int batchSize = (int) conceptIds.size(0);
        int seqLen = (int) conceptIds.size(1);

        Tensor cIdsLong = conceptIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)));
        Tensor rLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)));

        Tensor qEmb = qEmbed.forward(cIdsLong.toType(ScalarType.Long));
        Tensor cEmb = cEmbed.index_select(0, cIdsLong.toType(ScalarType.Long).view(-1L))
                .view(batchSize, seqLen, embedDim);

        Tensor cMask = cIdsLong.ne(new Scalar(0)).unsqueeze(2).toType(ScalarType.Float);
        Tensor cSum = cEmb.mul(cMask).sum(1);
        Tensor cCount = cMask.sum(1).add(new Scalar(1e-8));
        Tensor cAvg = cSum.div(cCount);

        Tensor prompts = promptMLP.forward(cAvg);
        Tensor qaEmb = qaEmbed.forward(rLong.toType(ScalarType.Long));

        Tensor combinedEmb = qEmb.add(cEmb).add(prompts.unsqueeze(1));
        Tensor posEnc = posEmb.forward(combinedEmb);
        Tensor embWithPos = combinedEmb.add(posEnc);
        Tensor embWithQA = embWithPos.add(qaEmb);

        Tensor x = embWithQA;
        for (TransformerLayer block : blocks) {
            x = block.forward(x, 0);
        }

        Tensor concatQ = torch.cat(new TensorVector(x, qEmb), 2);
        Tensor logits = outMLP.forward(concatQ);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
