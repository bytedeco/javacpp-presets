/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/SAKT.scala (SAKTUnified)
 *
 * SAKT with unified query: uses the interaction embedding as query.
 */
package org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.CosinePositionalEmbedding;
import org.bytedeco.pytorch.utils.recommend.models.knowledge_tracing.layers.MultiHeadAttention;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SAKTUnified extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private static final class Block {
        final MultiHeadAttention attn;
        final LinearImpl ffn1;
        final LinearImpl ffn2;
        final LayerNormImpl ln1;
        final LayerNormImpl ln2;
        final DropoutImpl drop;

        Block(MultiHeadAttention attn, LinearImpl ffn1, LinearImpl ffn2,
              LayerNormImpl ln1, LayerNormImpl ln2, DropoutImpl drop) {
            this.attn = attn;
            this.ffn1 = ffn1;
            this.ffn2 = ffn2;
            this.ln1 = ln1;
            this.ln2 = ln2;
            this.drop = drop;
        }
    }

    private final EmbeddingImpl interactionEmb;
    private final CosinePositionalEmbedding posEmb;
    private final LinearImpl predLayer;
    private final DropoutImpl dropoutLayer;
    private final List<Block> blocks = new ArrayList<>();

    public SAKTUnified(long numConcepts) {
        this(numConcepts, 64, 8, 2, 0.2f, DeviceSupport.backend());
    }

    public SAKTUnified(long numConcepts, int embedDim, int numHeads, int numBlocks, float dropout, String device) {
        super("SAKTUnified");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        this.interactionEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        this.posEmb = new CosinePositionalEmbedding(embedDim, 512, device);
        this.predLayer = new LinearImpl(embedDim, 1);
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("interaction_emb", interactionEmb);
        register_module("pos_emb", posEmb);
        register_module("pred", predLayer);

        for (int i = 0; i < numBlocks; i++) {
            MultiHeadAttention attn = new MultiHeadAttention(embedDim, numHeads, dropout, device);
            LinearImpl ffn1 = new LinearImpl(embedDim, embedDim * 4L);
            LinearImpl ffn2 = new LinearImpl(embedDim * 4L, embedDim);
            LayerNormImpl ln1 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
            LayerNormImpl ln2 = new LayerNormImpl(new LayerNormOptions(layerNormShape(embedDim)));
            DropoutImpl drop = new DropoutImpl(dropout);

            register_module("attn_" + i, attn);
            register_module("ffn1_" + i, ffn1);
            register_module("ffn2_" + i, ffn2);
            register_module("ln1_" + i, ln1);
            register_module("ln2_" + i, ln2);

            blocks.add(new Block(attn, ffn1, ffn2, ln1, ln2, drop));
        }

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            interactionEmb.to(dev, false);
            predLayer.to(dev, false);
        }
    }

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor interactionIds = conceptIds.mul(new Scalar(2.0)).add(responses).toType(ScalarType.Long);
        Tensor x = interactionEmb.forward(interactionIds);
        Tensor posEnc = posEmb.forward(x);
        x = x.add(posEnc);

        for (Block b : blocks) {
            Tensor attended = b.attn.forward(x, x, x);
            Tensor withRes1 = x.add(b.drop.forward(attended));
            Tensor normed1 = b.ln1.forward(withRes1);
            Tensor ffnOut = b.drop.forward(b.ffn2.forward(torch.relu(b.ffn1.forward(normed1))));
            x = b.ln2.forward(normed1.add(ffnOut));
        }

        Tensor logits = predLayer.forward(dropoutLayer.forward(x));
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses) {
        return forward(conceptIds, responses);
    }
}
