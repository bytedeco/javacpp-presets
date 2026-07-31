/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/CSKT.scala
 *
 * CSKT: Cone Shape Knowledge Tracing — transformer with cone-shaped attention.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.MLP;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.ConeAttention;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CSKT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private static final class Block {
        final ConeAttention coneAttn;
        final LinearImpl ffn1;
        final LinearImpl ffn2;
        final LayerNormImpl ln1;
        final LayerNormImpl ln2;
        final DropoutImpl drop;

        Block(ConeAttention coneAttn, LinearImpl ffn1, LinearImpl ffn2,
              LayerNormImpl ln1, LayerNormImpl ln2, DropoutImpl drop) {
            this.coneAttn = coneAttn;
            this.ffn1 = ffn1;
            this.ffn2 = ffn2;
            this.ln1 = ln1;
            this.ln2 = ln2;
            this.drop = drop;
        }
    }

    private final EmbeddingImpl qEmbed;
    private final EmbeddingImpl qaEmbed;
    private final CosinePositionalEmbedding posEmbed;
    private final List<Block> coneLayers = new ArrayList<>();
    private final MLP outLayer;

    public CSKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 1.0f, 1.0f, 0.1f, DeviceSupport.backend());
    }

    public CSKT(
            long numConcepts,
            int embedDim,
            int numHeads,
            int numBlocks,
            float r,
            float gamma,
            float dropout,
            String device) {
        super("CSKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        this.qEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        this.qaEmbed = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        register_module("q_embed", qEmbed);
        register_module("qa_embed", qaEmbed);

        this.posEmbed = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_embed", posEmbed);

        for (int i = 0; i < numBlocks; i++) {
            ConeAttention coneAttn = new ConeAttention(embedDim, numHeads, r, gamma, dropout, device);
            LinearImpl ffn1 = new LinearImpl(embedDim, embedDim * 4L);
            LinearImpl ffn2 = new LinearImpl(embedDim * 4L, embedDim);
            LayerNormImpl ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
            LayerNormImpl ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
            DropoutImpl drop = new DropoutImpl(dropout);

            register_module("cone_attn_" + i, coneAttn);
            register_module("ffn1_" + i, ffn1);
            register_module("ffn2_" + i, ffn2);
            register_module("ln1_" + i, ln1);
            register_module("ln2_" + i, ln2);

            coneLayers.add(new Block(coneAttn, ffn1, ffn2, ln1, ln2, drop));
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

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor interactionIds = conceptIds.mul(new Scalar(2.0)).add(responses).toType(ScalarType.Long);
        Tensor qaEmb = qaEmbed.forward(interactionIds);
        Tensor posEnc = posEmbed.forward(qaEmb);
        qaEmb = qaEmb.add(posEnc);

        Tensor qEmb = qEmbed.forward(conceptIds);
        Tensor posEncQ = posEmbed.forward(qEmb);
        Tensor adjQEmb = qEmb.add(posEncQ);

        Tensor x = adjQEmb;
        for (Block b : coneLayers) {
            Tensor attended = b.coneAttn.forward(x, x, 0);
            Tensor withRes1 = x.add(b.drop.forward(attended));
            Tensor normed1 = b.ln1.forward(withRes1);

            Tensor ffnOut = b.drop.forward(b.ffn2.forward(torch.relu(b.ffn1.forward(normed1))));
            x = b.ln2.forward(normed1.add(ffnOut));
        }

        Tensor concatQa = torch.cat(new TensorVector(x, adjQEmb), 2);
        Tensor logits = outLayer.forward(concatQa);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
