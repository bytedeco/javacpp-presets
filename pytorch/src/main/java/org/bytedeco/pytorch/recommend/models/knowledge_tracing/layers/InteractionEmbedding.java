/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/layers/PositionalEmbedding.scala
 *
 * Interaction embedding: combines concept ID and response into a single ID.
 * Used by DKT, DKVMN, DKTPlus.
 * interaction_id = conceptId + numConcepts * response
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class InteractionEmbedding extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numConcepts;
    private final long vocabSize;
    private final EmbeddingImpl interactionEmb;

    public InteractionEmbedding(long numConcepts, int embedDim) {
        this(numConcepts, embedDim, DeviceSupport.backend());
    }

    public InteractionEmbedding(long numConcepts, int embedDim, String device) {
        super("InteractionEmbedding");
        this.numConcepts = numConcepts;
        // vocab size = numConcepts * 2 + 1 to handle all interaction IDs + padding
        this.vocabSize = numConcepts * 2 + 1;
        this.interactionEmb = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim));
        register_module("interaction_emb", interactionEmb);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            interactionEmb.to(dev, false);
        }
    }

    public Tensor forward(Tensor conceptIds, Tensor responses) {
        Tensor conceptIdsLong = conceptIds.toType(ScalarType.Long);
        Tensor responsesLong = responses.toType(ScalarType.Long);
        long maxId = vocabSize - 1;
        Tensor interactionIdsRaw = conceptIdsLong.add(responsesLong.mul(new Scalar((double) numConcepts)));
        Tensor interactionIdsClamped = interactionIdsRaw.clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(maxId)))
                .toType(ScalarType.Long);
        return interactionEmb.forward(interactionIdsClamped);
    }
}
