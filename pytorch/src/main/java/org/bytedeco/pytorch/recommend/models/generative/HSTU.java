/*
 * Ported from torch-rechub-scala: torchrec/models/generative/HSTU.scala
 *
 * HSTU: Hierarchical Sequential Transduction Units.
 * Autoregressive generative recommender stacking HSTUBlock layers.
 * Reference: Meta, 2024 - Generative Recommenders.
 *
 * Input: tokens (B, L), optional timeDiffs (B, L)
 * Output: logits (B, L, vocabSize)
 */
package org.bytedeco.pytorch.recommend.models.generative;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
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
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;
import org.bytedeco.pytorch.recommend.DeviceSupport;
import org.bytedeco.pytorch.recommend.basic.layers.HSTUBlock;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class HSTU extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final long vocabSize;
    private final int dModel;
    private final int maxSeqLen;
    private final boolean useTimeEmbedding;
    private final int numTimeBuckets;
    private final String timeBucketFn;
    private final float timeBucketDivisor;
    private final String timeBucketUnit;
    private final boolean tieEmbeddings;
    private final String scoreNorm;
    private final float temperature;
    private final boolean useOutputBias;
    private final boolean scaleInputEmbedding;
    private final float l2NormEps;
    private final String device;

    private final EmbeddingImpl tokenEmbedding;
    private final EmbeddingImpl positionEmbedding;
    private final EmbeddingImpl timeEmbedding; // may be null if !useTimeEmbedding
    private final HSTUBlock hstuBlock;
    private final LinearImpl outputProjection; // null if tieEmbeddings
    private final Tensor outputBiasParam; // null if not used
    private final DropoutImpl dropoutLayer;

    public HSTU(long vocabSize) {
        this(vocabSize, 512, 8, 4, 64, 64, 256, 0.1f, true, 128, "sqrt", 1.0f,
                "minutes", true, "none", 1.0f, true, false, 1e-6f, DeviceSupport.backend());
    }

    public HSTU(
            long vocabSize,
            int dModel,
            int nHeads,
            int nLayers,
            int dqk,
            int dv,
            int maxSeqLen,
            float dropout,
            boolean useTimeEmbedding,
            int numTimeBuckets,
            String timeBucketFn,
            float timeBucketDivisor,
            String timeBucketUnit,
            boolean tieEmbeddings,
            String scoreNorm,
            float temperature,
            boolean useOutputBias,
            boolean scaleInputEmbedding,
            float l2NormEps,
            String device) {
        super("HSTU");
        if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be positive");
        if (dModel <= 0) throw new IllegalArgumentException("dModel must be positive");
        if (nHeads <= 0) throw new IllegalArgumentException("nHeads must be positive");
        if (dModel % nHeads != 0) {
            throw new IllegalArgumentException("dModel (" + dModel + ") must be divisible by nHeads (" + nHeads + ")");
        }
        if (!"none".equals(scoreNorm) && !"l2".equals(scoreNorm)) {
            throw new IllegalArgumentException("scoreNorm must be 'none' or 'l2'");
        }
        if (temperature <= 0) throw new IllegalArgumentException("temperature must be positive");
        if (!"sqrt".equals(timeBucketFn) && !"log".equals(timeBucketFn)) {
            throw new IllegalArgumentException("timeBucketFn must be 'sqrt' or 'log'");
        }
        if (!"minutes".equals(timeBucketUnit) && !"seconds".equals(timeBucketUnit)) {
            throw new IllegalArgumentException("timeBucketUnit must be 'minutes' or 'seconds'");
        }

        this.vocabSize = vocabSize;
        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.useTimeEmbedding = useTimeEmbedding;
        this.numTimeBuckets = numTimeBuckets;
        this.timeBucketFn = timeBucketFn;
        this.timeBucketDivisor = timeBucketDivisor;
        this.timeBucketUnit = timeBucketUnit;
        this.tieEmbeddings = tieEmbeddings;
        this.scoreNorm = scoreNorm;
        this.temperature = temperature;
        this.useOutputBias = useOutputBias;
        this.scaleInputEmbedding = scaleInputEmbedding;
        this.l2NormEps = l2NormEps;
        this.device = device;

        Device dev = new Device(device);

        EmbeddingOptions tokenOpts = new EmbeddingOptions(vocabSize, dModel);
        tokenOpts.padding_idx().put(new LongOptional(0L));
        this.tokenEmbedding = new EmbeddingImpl(tokenOpts);
        this.tokenEmbedding.to(dev, false);
        register_module("tokenEmbedding", tokenEmbedding);

        this.positionEmbedding = new EmbeddingImpl(new EmbeddingOptions(maxSeqLen, dModel));
        this.positionEmbedding.to(dev, false);
        register_module("positionEmbedding", positionEmbedding);

        if (useTimeEmbedding) {
            this.timeEmbedding = new EmbeddingImpl(new EmbeddingOptions(numTimeBuckets, dModel));
            this.timeEmbedding.to(dev, false);
            register_module("timeEmbedding", timeEmbedding);
        } else {
            this.timeEmbedding = null;
        }

        this.hstuBlock = new HSTUBlock(dModel, nHeads, nLayers, dqk, dv, dropout, maxSeqLen,
                numTimeBuckets, timeBucketFn, timeBucketDivisor, timeBucketUnit, device);
        register_module("hstuBlock", hstuBlock);

        if (tieEmbeddings) {
            this.outputProjection = null;
            if (useOutputBias) {
                Tensor biasTensor = torch.zeros(
                        new long[]{vocabSize},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)))
                        .to(dev, ScalarType.Float);
                this.outputBiasParam = biasTensor;
                register_parameter("outputBias", outputBiasParam);
            } else {
                this.outputBiasParam = null;
            }
        } else {
            this.outputProjection = new LinearImpl(dModel, vocabSize);
            this.outputProjection.to(dev, false);
            register_module("outputProj", outputProjection);
            this.outputBiasParam = null;
        }

        this.dropoutLayer = new DropoutImpl(dropout);
        initWeights();
    }

    private void initWeights() {
        Tensor teWeight = tokenEmbedding.weight();
        if (teWeight.dim() > 1) {
            torch.xavier_uniform_(teWeight);
        }
        Tensor peWeight = positionEmbedding.weight();
        if (peWeight.dim() > 1) {
            torch.xavier_uniform_(peWeight);
        }
        if (tieEmbeddings && useOutputBias && outputBiasParam != null) {
            torch.constant_(outputBiasParam, new Scalar(0.0f));
        }
    }

    private Tensor timeDiffToBucket(Tensor timeDiffs) {
        Tensor buckets = timeDiffs.toType(ScalarType.Float);
        if ("minutes".equals(timeBucketUnit)) {
            buckets = buckets.div(new Scalar(60.0f));
        }
        buckets = torch.clamp(buckets, new ScalarOptional(new Scalar(1e-6f)), new ScalarOptional());
        if ("sqrt".equals(timeBucketFn)) {
            buckets = torch.sqrt(buckets);
        } else if ("log".equals(timeBucketFn)) {
            buckets = torch.log(buckets);
        }
        buckets = buckets.div(new Scalar(timeBucketDivisor))
                .clamp(new ScalarOptional(new Scalar(0.0f)),
                        new ScalarOptional(new Scalar((float) (numTimeBuckets - 1))));
        return buckets.toType(ScalarType.Long);
    }

    /**
     * @param tokens    (B, L) token ids; 0 is PAD
     * @param timeDiffs (B, L) optional time diffs in seconds; may be null
     * @return logits (B, L, vocabSize)
     */
    public Tensor forward(Tensor tokens, Tensor timeDiffs) {
        Device dev = new Device(device);
        Tensor tokensOn;
        try {
            tokensOn = tokens.to(dev, ScalarType.Long);
        } catch (Throwable t) {
            tokensOn = tokens;
        }
        Tensor timeDiffsOn = null;
        if (timeDiffs != null) {
            try {
                timeDiffsOn = timeDiffs.to(dev, ScalarType.Float);
            } catch (Throwable t) {
                timeDiffsOn = timeDiffs;
            }
        }

        try {
            this.to(dev, false);
        } catch (Throwable ignored) {
        }

        int batchSize = (int) tokensOn.size(0);
        int seqLen = (int) tokensOn.size(1);
        if (seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                    "Input seq_len (" + seqLen + ") exceeds max_seq_len (" + maxSeqLen + "). "
                            + "Either truncate the input or rebuild the model with a larger max_seq_len.");
        }

        Tensor paddingMask = torch.ne(tokensOn, new Scalar(0L));

        Tensor embeddings = tokenEmbedding.forward(tokensOn.toType(ScalarType.Long));
        if (scaleInputEmbedding) {
            embeddings = embeddings.mul(new Scalar((float) Math.sqrt(dModel)));
        }

        Tensor positions = torch.arange(
                new Scalar(seqLen),
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                .to(dev, ScalarType.Long);
        Tensor posEmb = positionEmbedding.forward(positions.toType(ScalarType.Long));
        try {
            posEmb = posEmb.to(embeddings.device(), posEmb.dtype());
        } catch (Throwable ignored) {
        }
        embeddings = embeddings.add(posEmb.unsqueeze(0));

        if (useTimeEmbedding && timeEmbedding != null) {
            Tensor effectiveTimeDiffs = timeDiffsOn != null
                    ? timeDiffsOn
                    : torch.zeros(new long[]{batchSize, seqLen},
                            new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)))
                            .to(dev, ScalarType.Long);
            Tensor timeBuckets = timeDiffToBucket(effectiveTimeDiffs);
            Tensor timeEmb = timeEmbedding.forward(timeBuckets);
            try {
                timeEmb = timeEmb.to(embeddings.device(), timeEmb.dtype());
            } catch (Throwable ignored) {
            }
            embeddings = embeddings.add(timeEmb);
        }

        Tensor maskExpand = paddingMask.unsqueeze(-1).to(embeddings.dtype());
        try {
            maskExpand = maskExpand.to(embeddings.device(), maskExpand.dtype());
        } catch (Throwable ignored) {
        }
        embeddings = embeddings.mul(maskExpand);
        embeddings = dropoutLayer.forward(embeddings);

        Tensor hstuOutput = hstuBlock.forward(embeddings, paddingMask, timeDiffsOn);
        hstuOutput = hstuOutput.mul(paddingMask.unsqueeze(-1).to(hstuOutput.dtype()));

        Tensor outputWeight;
        Tensor outputBiasOpt;
        if (tieEmbeddings) {
            outputWeight = tokenEmbedding.weight();
            outputBiasOpt = outputBiasParam;
        } else {
            outputWeight = outputProjection.weight();
            outputBiasOpt = null;
        }

        Tensor finalOutput = hstuOutput;
        Tensor finalWeight = outputWeight;
        if ("l2".equals(scoreNorm)) {
            NormalizeFuncOptions opt = new NormalizeFuncOptions();
            opt.p(2);
            opt.dim(-1);
            opt.eps(l2NormEps);
            finalOutput = torch.normalize(finalOutput, opt);
            finalWeight = torch.normalize(finalWeight, opt);
        }

        Tensor logits = outputBiasOpt != null
                ? torch.linear(finalOutput, finalWeight, outputBiasOpt)
                : torch.linear(finalOutput, finalWeight);

        if (temperature != 1.0f) {
            logits = logits.div(new Scalar(temperature));
        }
        return logits;
    }

    /** Convenience overload without timeDiffs. */
    public Tensor forward(Tensor tokens) {
        return forward(tokens, (Tensor) null);
    }
}
