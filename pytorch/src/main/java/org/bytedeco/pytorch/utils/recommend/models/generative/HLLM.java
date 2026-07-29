/*
 * Ported from torch-rechub-scala: torchrec/models/generative/HLLM.scala
 *
 * HLLM: Hierarchical Large Language Model for Recommendation (ByteDance).
 * Pre-computed frozen item embeddings + stacked transformer blocks + cos-sim scoring.
 * Reference: https://github.com/bytedance/HLLM
 *
 * Note: train() intentionally does NOT call super.train (mirrors Scala workaround
 * for Director crash when traversing unregistered custom submodules).
 */
package org.bytedeco.pytorch.utils.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.basic.layers.RelativeBucketedTimeAndPositionBias;

import java.util.ArrayList;
import java.util.List;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class HLLM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final int dModel;
    private final int maxSeqLen;
    private final boolean useRelPosBias;
    private final boolean useTimeEmbedding;
    private final int numTimeBuckets;
    private final String timeBucketFn;
    private final float temperature;
    private final float l2NormEps = 1e-8f;
    private final Device targetDevice;

    private final Tensor normalizedItemEmbeddings;
    private final EmbeddingImpl positionEmbedding;
    private final EmbeddingImpl timeEmbedding; // null if !useTimeEmbedding
    private final List<HLLMTransformerBlock> transformerBlocks = new ArrayList<>();
    private final RelativeBucketedTimeAndPositionBias relPosBias; // null if !useRelPosBias
    private final DropoutImpl dropoutLayer;

    public HLLM(Tensor itemEmbeddings, long vocabSize) {
        this(itemEmbeddings, vocabSize, 512, 8, 4, 256, 0.1f, true, true, 2048, "sqrt", 0.07f,
                DeviceSupport.backend());
    }

    public HLLM(
            Tensor itemEmbeddings,
            long vocabSize,
            int dModel,
            int nHeads,
            int nLayers,
            int maxSeqLen,
            float dropout,
            boolean useRelPosBias,
            boolean useTimeEmbedding,
            int numTimeBuckets,
            String timeBucketFn,
            float temperature,
            String device) {
        super("HLLM");
        if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be positive");
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be positive");
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be positive");
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException(
                    "dModel (" + dModel + ") must be divisible by nHeads (" + nHeads + ")");
        }
        if (!"sqrt".equals(timeBucketFn) && !"log".equals(timeBucketFn)) {
            throw new IllegalArgumentException("timeBucketFn must be 'sqrt' or 'log'");
        }
        if (temperature <= 0) throw new IllegalArgumentException("temperature must be positive");
        if (itemEmbeddings.size(0) != vocabSize) {
            throw new IllegalArgumentException(
                    "item_embeddings.shape[0]=" + itemEmbeddings.size(0) + " != vocab_size=" + vocabSize);
        }
        if (itemEmbeddings.size(1) != dModel) {
            throw new IllegalArgumentException(
                    "item_embeddings.shape[1]=" + itemEmbeddings.size(1) + " != d_model=" + dModel);
        }

        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.useRelPosBias = useRelPosBias;
        this.useTimeEmbedding = useTimeEmbedding;
        this.numTimeBuckets = numTimeBuckets;
        this.timeBucketFn = timeBucketFn;
        this.temperature = temperature;
        this.targetDevice = new Device(device);

        // Frozen item embeddings — normalized once
        Tensor emb = itemEmbeddings.clone();
        NormalizeFuncOptions normOpt = new NormalizeFuncOptions();
        normOpt.p(2);
        normOpt.dim(-1);
        normOpt.eps(l2NormEps);
        this.normalizedItemEmbeddings = torch.normalize(emb.toType(ScalarType.Float), normOpt)
                .to(targetDevice, ScalarType.Float);
        register_buffer("item_embeddings", normalizedItemEmbeddings);

        this.positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxSeqLen, dModel));
        this.positionEmbedding.to(targetDevice, false);
        register_module("positionEmbedding", positionEmbedding);

        if (useTimeEmbedding) {
            EmbeddingOptions opts = new EmbeddingOptions(numTimeBuckets + 1L, dModel);
            opts.padding_idx().put(new LongOptional(0L));
            this.timeEmbedding = new EmbeddingImpl(opts);
            this.timeEmbedding.to(targetDevice, false);
            register_module("timeEmbedding", timeEmbedding);
        } else {
            this.timeEmbedding = null;
        }

        for (int i = 0; i < nLayers; i++) {
            HLLMTransformerBlock block = new HLLMTransformerBlock(dModel, nHeads, dropout, device);
            block.to(targetDevice, false);
            register_module("transformer_blocks_" + i, block);
            transformerBlocks.add(block);
        }

        if (useRelPosBias) {
            this.relPosBias = new RelativeBucketedTimeAndPositionBias(
                    nHeads, maxSeqLen, numTimeBuckets, timeBucketFn, 1.0f, "minutes", device);
            this.relPosBias.to(targetDevice, false);
            register_module("relPosBias", relPosBias);
        } else {
            this.relPosBias = null;
        }

        this.dropoutLayer = new DropoutImpl(dropout);
        register_module("dropoutLayer", dropoutLayer);

        initWeights();
    }

    /**
     * Intentionally does NOT call super.train — mirrors Scala workaround for
     * Director crash when C++ traverses unregistered custom submodules.
     */
    @Override
    public void train(boolean on) {
        positionEmbedding.train(on);
        if (timeEmbedding != null) {
            timeEmbedding.train(on);
        }
        dropoutLayer.train(on);
        for (HLLMTransformerBlock block : transformerBlocks) {
            block.train(on);
        }
        if (relPosBias != null) {
            try {
                relPosBias.train(on);
            } catch (Throwable ignored) {
            }
        }
    }

    private void initWeights() {
        Tensor peWeight = positionEmbedding.weight();
        if (peWeight.dim() > 1) {
            torch.xavier_uniform_(peWeight);
        }
        for (HLLMTransformerBlock block : transformerBlocks) {
            block.initWeights();
        }
    }

    private Tensor timeDiffToBucket(Tensor timeDiffs) {
        Tensor buckets = timeDiffs.toType(ScalarType.Float);
        buckets = buckets.div(new Scalar(60.0f));
        buckets = torch.clamp(buckets, new ScalarOptional(new Scalar(1e-6f)), new ScalarOptional());
        buckets = "sqrt".equals(timeBucketFn) ? torch.sqrt(buckets) : torch.log(buckets);
        return buckets.clamp(
                new ScalarOptional(new Scalar(0.0f)),
                new ScalarOptional(new Scalar((float) (numTimeBuckets - 1))))
                .toType(ScalarType.Long);
    }

    /**
     * Primary forward: look up frozen item emb, add pos/time, run transformer, cos-sim score.
     *
     * @param seqTokens (B, L) item ids
     * @param timeDiffs (B, L) optional time diffs in seconds; may be null
     * @return logits (B, L, vocabSize)
     */
    public Tensor forward(Tensor seqTokens, Tensor timeDiffs) {
        int batchSize = (int) seqTokens.size(0);
        int seqLen = (int) seqTokens.size(1);

        // 1. Look up item embeddings (flatten → index_select → view)
        Tensor flatTokens = seqTokens.contiguous().view(batchSize * seqLen).toType(ScalarType.Long);
        Tensor flatItemEmb = normalizedItemEmbeddings.index_select(0, flatTokens);
        Tensor itemEmb = flatItemEmb.view(batchSize, seqLen, dModel);

        // 2. Positional embedding
        Tensor positions = torch.arange(
                new Scalar(seqLen),
                new TensorOptions()
                        .dtype(new ScalarTypeOptional(ScalarType.Long))
                        .device(new DeviceOptional(seqTokens.device())));
        Tensor posEmb = positionEmbedding.forward(positions);
        Tensor embeddings = itemEmb.add(posEmb.unsqueeze(0));

        // 3. Time embedding
        if (useTimeEmbedding && timeEmbedding != null) {
            Tensor td = timeDiffs != null
                    ? timeDiffs
                    : torch.zeros(
                            new long[]{batchSize, seqLen},
                            new TensorOptions()
                                    .dtype(new ScalarTypeOptional(ScalarType.Long))
                                    .device(new DeviceOptional(seqTokens.device())));
            Tensor timeBuckets = timeDiffToBucket(td);
            Tensor timeEmb = timeEmbedding.forward(timeBuckets);
            embeddings = embeddings.add(timeEmb);
        }

        // 4. Dropout
        embeddings = dropoutLayer.forward(embeddings);

        // 5. Relative position bias
        Tensor relPosBiasTensor = relPosBias != null ? relPosBias.forward(seqLen) : null;

        // 6. Transformer blocks
        Tensor x = embeddings;
        for (HLLMTransformerBlock block : transformerBlocks) {
            x = block.forward(x, relPosBiasTensor);
        }

        // 7. Cosine-similarity scoring head
        NormalizeFuncOptions xNormedOpt = new NormalizeFuncOptions();
        xNormedOpt.p(2);
        xNormedOpt.dim(-1);
        xNormedOpt.eps(l2NormEps);
        Tensor xNormed = torch.normalize(x, xNormedOpt);
        Tensor itemEmbTransposed = normalizedItemEmbeddings.t();
        return torch.matmul(xNormed, itemEmbTransposed).div(new Scalar(temperature));
    }

    public Tensor forward(Tensor seqTokens) {
        return forward(seqTokens, (Tensor) null);
    }

    /**
     * Alternate forward (Scala forward2): index_select without flatten/view reshape.
     * Kept for parity with the Scala source.
     */
    public Tensor forward2(Tensor seqTokens, Tensor timeDiffs) {
        int batchSize = (int) seqTokens.size(0);
        int seqLen = (int) seqTokens.size(1);

        Tensor itemEmb = normalizedItemEmbeddings.index_select(0, seqTokens.toType(ScalarType.Long));

        Tensor positions = torch.arange(
                new Scalar(seqLen),
                new TensorOptions()
                        .dtype(new ScalarTypeOptional(ScalarType.Long))
                        .device(new DeviceOptional(seqTokens.device())));
        Tensor posEmb = positionEmbedding.forward(positions);
        Tensor embeddings = itemEmb.add(posEmb.unsqueeze(0));

        if (useTimeEmbedding && timeEmbedding != null) {
            Tensor td = timeDiffs != null
                    ? timeDiffs
                    : torch.zeros(
                            new long[]{batchSize, seqLen},
                            new TensorOptions()
                                    .dtype(new ScalarTypeOptional(ScalarType.Long))
                                    .device(new DeviceOptional(seqTokens.device())));
            Tensor timeBuckets = timeDiffToBucket(td);
            Tensor timeEmb = timeEmbedding.forward(timeBuckets);
            embeddings = embeddings.add(timeEmb);
        }

        embeddings = dropoutLayer.forward(embeddings);
        Tensor relPosBiasTensor = relPosBias != null ? relPosBias.forward(seqLen) : null;

        Tensor x = embeddings;
        for (HLLMTransformerBlock block : transformerBlocks) {
            x = block.forward(x, relPosBiasTensor);
        }

        NormalizeFuncOptions xNormedOpt = new NormalizeFuncOptions();
        xNormedOpt.p(2);
        xNormedOpt.dim(-1);
        xNormedOpt.eps(l2NormEps);
        Tensor xNormed = torch.normalize(x, xNormedOpt);
        Tensor itemEmbTransposed = normalizedItemEmbeddings.t();
        return torch.matmul(xNormed, itemEmbTransposed).div(new Scalar(temperature));
    }

    public Tensor getItemEmbedding(Tensor itemId) {
        return normalizedItemEmbeddings.index_select(0, itemId.toType(ScalarType.Long));
    }
}
