/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/MTKT.scala
 *
 * MTKT: Multi-Task Knowledge Tracing.
 * Question/Response aspect transformers with gating → MLP prediction.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

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
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.TransformerLayer;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MTKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final EmbeddingImpl qEmbed;
    private final EmbeddingImpl rEmbed;
    private final EmbeddingImpl qaEmbed;
    private final CosinePositionalEmbedding posEmbed;
    private final List<TransformerLayer> questionBlocks = new ArrayList<>();
    private final List<TransformerLayer> responseBlocks = new ArrayList<>();
    private final LinearImpl cWeight;
    private final LinearImpl tWeight;
    private final MLP outMLP;
    private final DropoutImpl dropoutLayer;

    public MTKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 0.2f, DeviceSupport.backend());
    }

    public MTKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            float dropout,
            String device) {
        super("MTKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numConcepts = numConcepts;

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("q_embed", qEmbed);

        this.rEmbed = new EmbeddingImpl(new EmbeddingOptions(2 + 1, embedDim));
        register_module("r_embed", rEmbed);

        this.qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        register_module("qa_embed", qaEmbed);

        this.posEmbed = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_embed", posEmbed);

        for (int i = 0; i < numBlocks; i++) {
            TransformerLayer qLayer = new TransformerLayer(embedDim, numHeads, embedDim * 4, dropout, device);
            register_module("question_block_" + i, qLayer);
            questionBlocks.add(qLayer);

            TransformerLayer rLayer = new TransformerLayer(embedDim, numHeads, embedDim * 4, dropout, device);
            register_module("response_block_" + i, rLayer);
            responseBlocks.add(rLayer);
        }

        this.cWeight = new LinearImpl(embedDim, embedDim);
        register_module("c_weight", cWeight);
        this.tWeight = new LinearImpl(embedDim, embedDim);
        register_module("t_weight", tWeight);

        this.outMLP = new MLP(embedDim * 2L, new long[]{(long) embedDim}, 1L, "relu", dropout,
                false, false, true, device);
        register_module("out_mlp", outMLP);

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropout", dropoutLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qEmbed.to(dev, false);
            rEmbed.to(dev, false);
            qaEmbed.to(dev, false);
            outMLP.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor cIdsLong = conceptIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numConcepts)));
        Tensor rLong = responses.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)));

        Tensor qEmb = qEmbed.forward(cIdsLong.toType(ScalarType.Long));
        Tensor rEmb = rEmbed.forward(rLong.toType(ScalarType.Long));

        Tensor qaIds = cIdsLong.add(rLong.mul(new Scalar((double) numConcepts)));
        Tensor qaEmb = qaEmbed.forward(qaIds.toType(ScalarType.Long));

        Tensor posEnc = posEmbed.forward(qEmb);
        Tensor qWithPos = qEmb.add(posEnc);
        Tensor rWithPos = rEmb.add(posEnc);

        Tensor qOut = qWithPos;
        for (TransformerLayer block : questionBlocks) {
            qOut = block.forward(qOut, 1);
        }

        Tensor rOut = rWithPos;
        for (TransformerLayer block : responseBlocks) {
            rOut = block.forward(rOut, 0);
        }

        // Gating: w * qOut + (1-w) * rOut
        Tensor w = torch.sigmoid(cWeight.forward(qOut).add(tWeight.forward(rOut)));
        Tensor dOutput = w.mul(qOut).add(w.mul(new Scalar(1.0)).neg().add(new Scalar(1.0)).mul(rOut));

        Tensor concatQ = torch.cat(new TensorVector(dOutput, qEmb), 2);
        Tensor logits = outMLP.forward(concatQ);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
