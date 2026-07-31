/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/SAINT.scala
 *
 * SAINT: Self-Attentive Interpretable Knowledge Tracing (Shin et al., 2021).
 * Encoder (exercise+category+pos) → Decoder (response+pos with start token) → sigmoid.
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
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.LearnablePositionalEmbedding;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.SAINTDecoderBlock;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.SAINTEncoderBlock;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SAINT extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long numExercises;
    private final long numCategories;
    private final int embedDim;
    private final int startTokenId;
    private final String device;
    private final EmbeddingImpl exEmb;
    private final EmbeddingImpl catEmb;
    private final EmbeddingImpl resEmb;
    private final LearnablePositionalEmbedding posEmb;
    private final List<SAINTEncoderBlock> encoderBlocks = new ArrayList<>();
    private final List<SAINTDecoderBlock> decoderBlocks = new ArrayList<>();
    private final LinearImpl outLayer;

    public SAINT(long numExercises, long numCategories) {
        this(numExercises, numCategories, 2, 64, 8, 3, 3, 256, 0.2f, DeviceSupport.backend());
    }

    public SAINT(
            long numExercises,
            long numCategories,
            int numResponses,
            int embedDim,
            int numHeads,
            int numEncoderBlocks,
            int numDecoderBlocks,
            int ffnDim,
            float dropout,
            String device) {
        super("SAINT");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.numExercises = numExercises;
        this.numCategories = numCategories;
        this.embedDim = embedDim;
        this.device = device;
        this.startTokenId = numResponses + 1;

        this.exEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises + 1, embedDim));
        register_module("ex_emb", exEmb);

        this.catEmb = new EmbeddingImpl(new EmbeddingOptions(numCategories + 1, embedDim));
        register_module("cat_emb", catEmb);

        this.resEmb = new EmbeddingImpl(new EmbeddingOptions(numResponses + 2L, embedDim));
        register_module("res_emb", resEmb);

        this.posEmb = new LearnablePositionalEmbedding(512, embedDim, dropout, device);
        register_module("pos_emb", posEmb);

        for (int i = 0; i < numEncoderBlocks; i++) {
            SAINTEncoderBlock block = new SAINTEncoderBlock(embedDim, numHeads, ffnDim, dropout, device);
            register_module("encoder_" + i, block);
            encoderBlocks.add(block);
        }

        for (int i = 0; i < numDecoderBlocks; i++) {
            SAINTDecoderBlock block = new SAINTDecoderBlock(embedDim, numHeads, ffnDim, dropout, device);
            register_module("decoder_" + i, block);
            decoderBlocks.add(block);
        }

        this.outLayer = new LinearImpl(embedDim, 1);
        register_module("out", outLayer);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            exEmb.to(dev, false);
            catEmb.to(dev, false);
            resEmb.to(dev, false);
            outLayer.to(dev, false);
        }
    }

    public Tensor forward(Tensor exerciseIds, Tensor categoryIds, Tensor responseIds) {
        int batchSize = (int) exerciseIds.size(0);
        int seqLen = (int) exerciseIds.size(1);
        Device devObj = new Device(device);

        Tensor exIds = exerciseIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numExercises)));
        Tensor catIds = categoryIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar((double) numCategories)));
        Tensor resIds = responseIds.toType(ScalarType.Long).clamp(
                new ScalarOptional(new Scalar(0)),
                new ScalarOptional(new Scalar(1)));

        Tensor exIdsDev = exIds.to(devObj, ScalarType.Long);
        Tensor catIdsDev = catIds.to(devObj, ScalarType.Long);
        Tensor resIdsDev = resIds.to(devObj, ScalarType.Long);

        // Defensive clamp against embedding table sizes
        try {
            long exNum = exEmb.weight().size(0);
            long catNum = catEmb.weight().size(0);
            long resNum = resEmb.weight().size(0);
            exIdsDev = exIdsDev.clamp(
                    new ScalarOptional(new Scalar(0)),
                    new ScalarOptional(new Scalar((double) (exNum - 1))));
            catIdsDev = catIdsDev.clamp(
                    new ScalarOptional(new Scalar(0)),
                    new ScalarOptional(new Scalar((double) (catNum - 1))));
            resIdsDev = resIdsDev.clamp(
                    new ScalarOptional(new Scalar(0)),
                    new ScalarOptional(new Scalar((double) (resNum - 1))));
        } catch (Throwable ignored) {
        }

        Tensor exIdsForEmb = exIdsDev.toType(ScalarType.Long).contiguous();
        Tensor catIdsForEmb = catIdsDev.toType(ScalarType.Long).contiguous();

        Tensor exEmbOut = exEmb.forward(exIdsForEmb);
        Tensor catEmbOut = catEmb.forward(catIdsForEmb);
        Tensor posEnc = posEmb.forward(seqLen);
        Tensor posEx = posEnc.expand(batchSize, seqLen, embedDim);

        Tensor enOut = exEmbOut.add(catEmbOut);
        for (SAINTEncoderBlock block : encoderBlocks) {
            enOut = block.forward(enOut, catEmbOut, posEx);
        }

        // Decoder: prepend start token
        Tensor startTokens = torch.full(
                new long[]{batchSize, 1L},
                new Scalar((double) startTokenId),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        Tensor startTokensDev = !"cpu".equals(device)
                ? startTokens.to(devObj, ScalarType.Long)
                : startTokens;
        Tensor paddedResponses = torch.cat(new TensorVector(startTokensDev, resIdsDev), 1);
        Tensor paddedResponsesLong = paddedResponses.toType(ScalarType.Long)
                .to(devObj, ScalarType.Long).contiguous();

        Tensor resEmbOut = resEmb.forward(paddedResponsesLong);
        Tensor posDec = posEmb.forward(seqLen + 1L);
        Tensor posDecEx = posDec.expand(batchSize, seqLen + 1L, embedDim);

        Tensor decOut = resEmbOut;
        for (SAINTDecoderBlock block : decoderBlocks) {
            decOut = block.forward(decOut, posDecEx, enOut);
        }

        Tensor decOutShifted = decOut.narrow(1, 1, seqLen);
        Tensor logits = outLayer.forward(decOutShifted);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor exerciseIds, Tensor categoryIds, Tensor responseIds) {
        return forward(exerciseIds, categoryIds, responseIds);
    }
}
