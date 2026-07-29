/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/SAKT.scala
 *
 * SAKT: Self-Attentive Knowledge Tracing (Choi et al., KDD 2020).
 * Exercise emb + Interaction emb + Positional encoding → multi-head blocks → prediction.
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
public class SAKT extends Module {

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
    private final EmbeddingImpl exerciseEmb;
    private final CosinePositionalEmbedding posEmb;
    private final List<Block> blocks = new ArrayList<>();
    private final LinearImpl predLayer;
    private final DropoutImpl dropoutLayer;

    public SAKT(long numConcepts) {
        this(numConcepts, 64, 8, 2, 0.2f, DeviceSupport.backend());
    }

    public SAKT(long numConcepts, int embedDim, int numHeads, int numBlocks, float dropout, String device) {
        super("SAKT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }

        this.interactionEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts * 2, embedDim));
        register_module("interaction_emb", interactionEmb);

        this.exerciseEmb = new EmbeddingImpl(new EmbeddingOptions(numConcepts + 1, embedDim));
        register_module("exercise_emb", exerciseEmb);

        this.posEmb = new CosinePositionalEmbedding(embedDim, 512, device);
        register_module("pos_emb", posEmb);

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

        this.predLayer = new LinearImpl(embedDim, 1);
        register_module("pred", predLayer);

        this.dropoutLayer = new DropoutImpl(dropout);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            interactionEmb.to(dev, false);
            exerciseEmb.to(dev, false);
            predLayer.to(dev, false);
        }
    }

    private static LongVector layerNormShape(int d) {
        LongVector v = new LongVector(1);
        v.put(0, d);
        return v;
    }

    /**
     * @param conceptIds    (batch, seqLen) current concept
     * @param responses     (batch, seqLen) current response
     * @param targetConcept (batch, seqLen) shifted concept (query)
     * @return predictions (batch, seqLen)
     */
    public Tensor forward(Tensor conceptIds, Tensor responses, Tensor targetConcept) {
        // interaction_id = concept * 2 + response
        Tensor interactionIds = conceptIds.mul(new Scalar(2.0)).add(responses).toType(ScalarType.Long);

        Tensor xEmb = interactionEmb.forward(interactionIds);
        Tensor posEnc = posEmb.forward(xEmb);
        xEmb = xEmb.add(posEnc);

        Tensor qryEmb = exerciseEmb.forward(targetConcept);
        Tensor qryPos = posEmb.forward(qryEmb);
        Tensor qryEnc = qryEmb.add(qryPos);

        Tensor x = qryEnc;
        for (Block b : blocks) {
            // Cross-attention: x attends to xEmb
            Tensor attended = b.attn.forward(x, xEmb, xEmb);
            Tensor withRes1 = x.add(b.drop.forward(attended));
            Tensor normed1 = b.ln1.forward(withRes1);

            // Self-attention on x
            Tensor selfAttn = b.attn.forward(normed1, normed1, normed1);
            Tensor withRes2 = normed1.add(b.drop.forward(selfAttn));
            Tensor normed2 = b.ln2.forward(withRes2);

            // FFN
            Tensor ffnOut = b.drop.forward(b.ffn2.forward(torch.relu(b.ffn1.forward(normed2))));
            x = normed2.add(ffnOut);
        }

        Tensor logits = predLayer.forward(dropoutLayer.forward(x));
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor conceptIds, Tensor responses, Tensor targetConcept) {
        return forward(conceptIds, responses, targetConcept);
    }
}
