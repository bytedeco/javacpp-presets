/*
 * Ported from torch-rechub-scala: torchrec/models/knowledge_tracing/SAINTPlusPlus.scala
 *
 * SAINT++: Enhanced SAINT with deeper encoder/decoder and question embedding.
 * Also includes SAINTEncoderBlockPlus, SAINTDecoderBlockPlus, MultiHeadAttentionPlus.
 */
package org.bytedeco.pytorch.recommend.models.knowledge_tracing;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
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
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.models.knowledge_tracing.layers.LearnablePositionalEmbedding;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SAINTPlusPlus extends Module {

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
    private final EmbeddingImpl questionEmb;
    private final LearnablePositionalEmbedding posEmb;
    private final List<SAINTEncoderBlockPlus> encoderBlocks = new ArrayList<>();
    private final List<SAINTDecoderBlockPlus> decoderBlocks = new ArrayList<>();
    private final LinearImpl outLayer;

    public SAINTPlusPlus(long numExercises, long numCategories) {
        this(numExercises, numCategories, 2, 64, 8, 3, 3, 256, 0.2f, DeviceSupport.backend());
    }

    public SAINTPlusPlus(
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
        super("SAINTPlusPlus");
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

        this.questionEmb = new EmbeddingImpl(new EmbeddingOptions(numExercises * 2 + 1, embedDim));
        register_module("question_emb", questionEmb);

        this.posEmb = new LearnablePositionalEmbedding(512, embedDim, dropout, device);
        register_module("pos_emb", posEmb);

        for (int i = 0; i < numEncoderBlocks; i++) {
            SAINTEncoderBlockPlus block = new SAINTEncoderBlockPlus(embedDim, numHeads, ffnDim, dropout, device);
            register_module("encoder_" + i, block);
            encoderBlocks.add(block);
        }

        for (int i = 0; i < numDecoderBlocks; i++) {
            SAINTDecoderBlockPlus block = new SAINTDecoderBlockPlus(embedDim, numHeads, ffnDim, dropout, device);
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
            questionEmb.to(dev, false);
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
        Tensor exIdsForEmb = exIdsDev.toType(ScalarType.Long).contiguous();
        Tensor catIdsForEmb = catIdsDev.toType(ScalarType.Long).contiguous();
        Tensor exEmbOut = exEmb.forward(exIdsForEmb);
        Tensor catEmbOut = catEmb.forward(catIdsForEmb);
        Tensor posEnc = posEmb.forward(seqLen);
        Tensor posEx = posEnc.expand(batchSize, seqLen, embedDim);

        Tensor enOut = exEmbOut.add(catEmbOut).add(posEx);
        for (SAINTEncoderBlockPlus block : encoderBlocks) {
            enOut = block.forward(enOut, catEmbOut, posEx);
        }

        Tensor startTokens = torch.full(
                new long[]{batchSize, 1L},
                new Scalar((double) startTokenId),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        Tensor startTokensDev = !"cpu".equals(device) ? startTokens.to(devObj, ScalarType.Long) : startTokens;
        Tensor resIdsDev = resIds.to(devObj, ScalarType.Long);
        try {
            long resNum = resEmb.weight().size(0);
            resIdsDev = resIdsDev.clamp(
                    new ScalarOptional(new Scalar(0)),
                    new ScalarOptional(new Scalar((double) (resNum - 1))));
        } catch (Throwable ignored) {
        }
        Tensor paddedResponses = torch.cat(new TensorVector(startTokensDev, resIdsDev), 1);
        Tensor paddedResponsesLong = paddedResponses.toType(ScalarType.Long)
                .to(devObj, ScalarType.Long).contiguous();

        Tensor resEmbOut = resEmb.forward(paddedResponsesLong);
        Tensor posDec = posEmb.forward(seqLen + 1L);
        Tensor posDecEx = posDec.expand(batchSize, seqLen + 1L, embedDim);

        // Question embedding for decoder
        Tensor exIdsLongForQuestion = exIds.toType(ScalarType.Long);
        Tensor resIdsLongForQuestion = resIds.toType(ScalarType.Long);
        Tensor exIdsClampedForQuestion;
        try {
            exIdsClampedForQuestion = exIdsLongForQuestion.clamp(
                    new ScalarOptional(new Scalar(0)),
                    new ScalarOptional(new Scalar((double) (numExercises - 1))));
        } catch (Throwable t) {
            exIdsClampedForQuestion = exIdsLongForQuestion;
        }

        Tensor questionRawForReal = exIdsClampedForQuestion.mul(new Scalar(2)).add(resIdsLongForQuestion);
        long padIndexForQuestion = 2 * numExercises;
        Tensor isPad = exIdsLongForQuestion.eq(new Scalar((double) numExercises));
        Tensor padIndexTensor = torch.full(
                questionRawForReal.shape(),
                new Scalar((double) padIndexForQuestion),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                .to(isPad.device(), ScalarType.Long);
        Tensor questionIdsLongUnclamped = torch.where(isPad, padIndexTensor, questionRawForReal);

        Tensor questionIdsLong;
        try {
            long qNum = questionEmb.weight().size(0);
            questionIdsLong = questionIdsLongUnclamped.clamp(
                    new ScalarOptional(new Scalar(0)),
                    new ScalarOptional(new Scalar((double) (qNum - 1))))
                    .toType(ScalarType.Long);
        } catch (Throwable t) {
            questionIdsLong = questionIdsLongUnclamped.toType(ScalarType.Long);
        }
        Tensor questionEmbOut = questionEmb.forward(questionIdsLong.contiguous());
        Tensor posQ = posEmb.forward(seqLen);
        Tensor questionEnc = questionEmbOut.add(posQ.expand(batchSize, seqLen, embedDim));

        Tensor zerosPad = torch.zeros(
                new long[]{batchSize, 1L, embedDim},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        if (!"cpu".equals(device)) {
            zerosPad = zerosPad.to(devObj, ScalarType.Float);
        }
        Tensor paddedQuestions = torch.cat(new TensorVector(zerosPad, questionEnc), 1);

        Tensor decOut = resEmbOut.add(posDecEx);
        for (SAINTDecoderBlockPlus block : decoderBlocks) {
            decOut = block.forward(decOut, paddedQuestions, enOut);
        }

        Tensor decOutShifted = decOut.narrow(1, 1, seqLen);
        Tensor logits = outLayer.forward(decOutShifted);
        return logits.sigmoid().squeeze(2);
    }

    public Tensor predict(Tensor exerciseIds, Tensor categoryIds, Tensor responseIds) {
        return forward(exerciseIds, categoryIds, responseIds);
    }
}

/** Enhanced SAINT Encoder Block. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
class SAINTEncoderBlockPlus extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final MultiHeadAttentionPlus multiEn;
    private final LinearImpl ffnEn1;
    private final LinearImpl ffnEn2;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;
    private final DropoutImpl dropoutLayer;

    public SAINTEncoderBlockPlus(int embedDim, int numHeads, int ffnDim, float dropout, String device) {
        super("SAINTEncoderBlockPlus");
        this.multiEn = new MultiHeadAttentionPlus(embedDim, numHeads, dropout, device);
        this.ffnEn1 = new LinearImpl(embedDim, ffnDim);
        this.ffnEn2 = new LinearImpl(ffnDim, embedDim);
        LongVector shape = new LongVector(1);
        shape.put(0, embedDim);
        this.ln1 = new LayerNormImpl(new LayerNormOptions(shape));
        this.ln2 = new LayerNormImpl(new LayerNormOptions(shape));
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("multi_en", multiEn);
        register_module("ffn_en1", ffnEn1);
        register_module("ffn_en2", ffnEn2);
        register_module("ln1", ln1);
        register_module("ln2", ln2);
    }

    public Tensor forward(Tensor inEx, Tensor inCat, Tensor inPos) {
        Tensor combined = inEx.add(inCat).add(inPos);
        Tensor attended = multiEn.forward(combined, combined, combined);
        Tensor withResidual1 = combined.add(dropoutLayer.forward(attended));
        Tensor normed1 = ln1.forward(withResidual1);

        Tensor ffnOut = dropoutLayer.forward(ffnEn2.forward(torch.relu(ffnEn1.forward(normed1))));
        return ln2.forward(normed1.add(ffnOut));
    }
}

/** Enhanced SAINT Decoder Block. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
class SAINTDecoderBlockPlus extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final MultiHeadAttentionPlus multiDe1;
    private final MultiHeadAttentionPlus multiDe2;
    private final LinearImpl ffnDe1;
    private final LinearImpl ffnDe2;
    private final LayerNormImpl ln1;
    private final LayerNormImpl ln2;
    private final LayerNormImpl ln3;
    private final DropoutImpl dropoutLayer;

    public SAINTDecoderBlockPlus(int embedDim, int numHeads, int ffnDim, float dropout, String device) {
        super("SAINTDecoderBlockPlus");
        this.multiDe1 = new MultiHeadAttentionPlus(embedDim, numHeads, dropout, device);
        this.multiDe2 = new MultiHeadAttentionPlus(embedDim, numHeads, dropout, device);
        this.ffnDe1 = new LinearImpl(embedDim, ffnDim);
        this.ffnDe2 = new LinearImpl(ffnDim, embedDim);
        LongVector shape = new LongVector(1);
        shape.put(0, embedDim);
        this.ln1 = new LayerNormImpl(new LayerNormOptions(shape));
        this.ln2 = new LayerNormImpl(new LayerNormOptions(shape));
        this.ln3 = new LayerNormImpl(new LayerNormOptions(shape));
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("multi_de1", multiDe1);
        register_module("multi_de2", multiDe2);
        register_module("ffn_de1", ffnDe1);
        register_module("ffn_de2", ffnDe2);
        register_module("ln1", ln1);
        register_module("ln2", ln2);
        register_module("ln3", ln3);
    }

    public Tensor forward(Tensor inRes, Tensor inQuestion, Tensor enOut) {
        Tensor combined = inRes;

        Tensor crossAttn = multiDe1.forward(combined, enOut, enOut);
        Tensor withResidual1 = combined.add(dropoutLayer.forward(crossAttn));
        Tensor normed1 = ln1.forward(withResidual1);

        Tensor selfAttn = multiDe2.forward(normed1, normed1, normed1);
        Tensor withResidual2 = normed1.add(dropoutLayer.forward(selfAttn));
        Tensor normed2 = ln2.forward(withResidual2);

        Tensor ffnOut = dropoutLayer.forward(ffnDe2.forward(torch.relu(ffnDe1.forward(normed2))));
        return ln3.forward(normed2.add(ffnOut));
    }
}

/** Enhanced Multi-head attention. */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
class MultiHeadAttentionPlus extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int embedDim;
    private final int numHeads;
    private final int headDim;
    private final LinearImpl qLinear;
    private final LinearImpl kLinear;
    private final LinearImpl vLinear;
    private final LinearImpl outLinear;
    private final DropoutImpl dropoutLayer;

    public MultiHeadAttentionPlus(int embedDim, int numHeads, float dropout, String device) {
        super("MultiHeadAttentionPlus");
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException("embedDim must be divisible by numHeads");
        }
        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;

        this.qLinear = new LinearImpl(embedDim, embedDim);
        this.kLinear = new LinearImpl(embedDim, embedDim);
        this.vLinear = new LinearImpl(embedDim, embedDim);
        this.outLinear = new LinearImpl(embedDim, embedDim);
        this.dropoutLayer = new DropoutImpl(dropout);

        register_module("q_linear", qLinear);
        register_module("k_linear", kLinear);
        register_module("v_linear", vLinear);
        register_module("out_linear", outLinear);

        if (!"cpu".equals(device)) {
            Device dev = new Device(device);
            qLinear.to(dev, false);
            kLinear.to(dev, false);
            vLinear.to(dev, false);
            outLinear.to(dev, false);
        }
    }

    public Tensor forward(Tensor q, Tensor k, Tensor v, Tensor mask) {
        int batchSize = (int) q.size(0);
        int seqLen = (int) q.size(1);
        int keySeqLen = (int) k.size(1);

        Tensor qProj = qLinear.forward(q).view(batchSize, seqLen, numHeads, headDim).transpose(1, 2);
        Tensor kProj = kLinear.forward(k).view(batchSize, keySeqLen, numHeads, headDim).transpose(1, 2);
        Tensor vProj = vLinear.forward(v).view(batchSize, keySeqLen, numHeads, headDim).transpose(1, 2);

        Scalar scale = new Scalar((float) Math.sqrt(headDim));
        Tensor scores = torch.matmul(qProj, kProj.transpose(2, 3)).div(scale);

        if (mask != null && !mask.isNull() && mask.numel() > 0) {
            scores = scores.add(mask);
        }

        Tensor attnWeights = scores.softmax(-1);
        Tensor attended = torch.matmul(dropoutLayer.forward(attnWeights), vProj);
        Tensor reshaped = attended.transpose(1, 2).contiguous().view(batchSize, seqLen, embedDim);
        return outLinear.forward(reshaped);
    }

    public Tensor forward(Tensor q, Tensor k, Tensor v) {
        return forward(q, k, v, null);
    }
}
