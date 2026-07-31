/*
 * Ported from torch-rechub-scala: torchrec/basic/layers/Initializers.scala
 *
 * Embedding weight initializers: RandomNormal, RandomUniform, XavierNormal,
 * XavierUniform, Pretrained.
 */
package org.bytedeco.pytorch.recommend.basic.layers;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;

/**
 * Base interface for embedding initializers.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public interface EmbeddingInitializer {

    EmbeddingImpl apply(long vocabSize, long embedDim, Long paddingIdx);

    default EmbeddingImpl apply(long vocabSize, long embedDim) {
        return apply(vocabSize, embedDim, null);
    }

    static EmbeddingImpl createEmbedding(long vocabSize, long embedDim, Long paddingIdx) {
        EmbeddingOptions options = new EmbeddingOptions(vocabSize, embedDim);
        if (paddingIdx != null) {
            options.padding_idx().put(new LongOptional(paddingIdx));
        }
        return new EmbeddingImpl(options);
    }

    static void setPaddingZero(EmbeddingImpl embed, Long paddingIdx) {
        if (paddingIdx == null) {
            return;
        }
        // Mirror Scala setPaddingZero (index_copy_ path is commented out in source).
        // Keep no-op body for API parity with the active Scala implementation.
    }

    static void setPaddingZero2(EmbeddingImpl embed, Long paddingIdx) {
        if (paddingIdx == null) {
            return;
        }
        Tensor weight = embed.weight();
        Tensor idxTensor = torch.tensor(new long[]{paddingIdx},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long)));
        weight.index_fill_(0, idxTensor, new Scalar(0f));
    }

    static RandomNormal randomNormal() {
        return new RandomNormal(0.0f, 1.0f);
    }

    static RandomNormal randomNormal(float mean, float std) {
        return new RandomNormal(mean, std);
    }

    static RandomUniform randomUniform() {
        return new RandomUniform(0.0f, 1.0f);
    }

    static RandomUniform randomUniform(float minval, float maxval) {
        return new RandomUniform(minval, maxval);
    }

    static XavierNormal xavierNormal() {
        return new XavierNormal(1.0f);
    }

    static XavierNormal xavierNormal(float gain) {
        return new XavierNormal(gain);
    }

    static XavierUniform xavierUniform() {
        return new XavierUniform(1.0f);
    }

    static XavierUniform xavierUniform(float gain) {
        return new XavierUniform(gain);
    }

    static Pretrained pretrained(float[][] weights, boolean freeze) {
        return new Pretrained(weights, freeze);
    }

    /** Initializes embedding weights with a normal distribution. */
    final class RandomNormal implements EmbeddingInitializer {
        static {
            Loader.load(org.bytedeco.pytorch.presets.torch.class);
        }

        private final float mean;
        private final float std;

        public RandomNormal(float mean, float std) {
            this.mean = mean;
            this.std = std;
        }

        @Override
        public EmbeddingImpl apply(long vocabSize, long embedDim, Long paddingIdx) {
            EmbeddingImpl embed = createEmbedding(vocabSize, embedDim, paddingIdx);
            torch.normal_(embed.weight(), mean, std);
            setPaddingZero(embed, paddingIdx);
            return embed;
        }
    }

    /** Initializes embedding weights with a uniform distribution. */
    final class RandomUniform implements EmbeddingInitializer {
        static {
            Loader.load(org.bytedeco.pytorch.presets.torch.class);
        }

        private final float minval;
        private final float maxval;

        public RandomUniform(float minval, float maxval) {
            this.minval = minval;
            this.maxval = maxval;
        }

        @Override
        public EmbeddingImpl apply(long vocabSize, long embedDim, Long paddingIdx) {
            EmbeddingImpl embed = createEmbedding(vocabSize, embedDim, paddingIdx);
            torch.uniform_(embed.weight(), minval, maxval);
            setPaddingZero(embed, paddingIdx);
            return embed;
        }
    }

    /**
     * Xavier normal initialization.
     * std = gain * sqrt(2 / (fan_in + fan_out))
     */
    final class XavierNormal implements EmbeddingInitializer {
        static {
            Loader.load(org.bytedeco.pytorch.presets.torch.class);
        }

        private final float gain;

        public XavierNormal(float gain) {
            this.gain = gain;
        }

        @Override
        public EmbeddingImpl apply(long vocabSize, long embedDim, Long paddingIdx) {
            EmbeddingImpl embed = createEmbedding(vocabSize, embedDim, paddingIdx);
            float std = gain * (float) Math.sqrt(2.0 / (vocabSize + embedDim));
            torch.normal_(embed.weight(), 0.0f, std);
            setPaddingZero(embed, paddingIdx);
            return embed;
        }
    }

    /**
     * Xavier uniform initialization.
     * bound = gain * sqrt(6 / (fan_in + fan_out))
     */
    final class XavierUniform implements EmbeddingInitializer {
        static {
            Loader.load(org.bytedeco.pytorch.presets.torch.class);
        }

        private final float gain;

        public XavierUniform(float gain) {
            this.gain = gain;
        }

        @Override
        public EmbeddingImpl apply(long vocabSize, long embedDim, Long paddingIdx) {
            EmbeddingImpl embed = createEmbedding(vocabSize, embedDim, paddingIdx);
            float bound = gain * (float) Math.sqrt(6.0 / (vocabSize + embedDim));
            torch.uniform_(embed.weight(), -bound, bound);
            setPaddingZero(embed, paddingIdx);
            return embed;
        }
    }

    /** Creates an embedding layer from pretrained weights. */
    final class Pretrained implements EmbeddingInitializer {
        static {
            Loader.load(org.bytedeco.pytorch.presets.torch.class);
        }

        private final Tensor weightTensor;
        private final boolean freeze;

        public Pretrained(float[][] embeddingWeights, boolean freeze) {
            this.freeze = freeze;
            int rows = embeddingWeights.length;
            int cols = embeddingWeights[0].length;
            float[] flat = new float[rows * cols];
            for (int i = 0; i < rows; i++) {
                System.arraycopy(embeddingWeights[i], 0, flat, i * cols, cols);
            }
            this.weightTensor = torch.tensor(flat).view(rows, cols);
        }

        @Override
        public EmbeddingImpl apply(long vocabSize, long embedDim, Long paddingIdx) {
            if (vocabSize != weightTensor.size(0)) {
                throw new IllegalArgumentException(
                        "vocab_size mismatch: expected " + weightTensor.size(0) + ", got " + vocabSize);
            }
            if (embedDim != weightTensor.size(1)) {
                throw new IllegalArgumentException(
                        "embed_dim mismatch: expected " + weightTensor.size(1) + ", got " + embedDim);
            }

            EmbeddingImpl embedImpl = new EmbeddingImpl(new EmbeddingOptions(vocabSize, embedDim));
            embedImpl.weight().copy_(weightTensor);

            if (freeze) {
                embedImpl.weight().set_requires_grad(false);
            }

            if (paddingIdx != null) {
                Tensor zeroRow = torch.zeros(new long[]{embedDim},
                        new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
                embedImpl.weight().index_copy_(
                        0,
                        torch.tensor(new long[]{paddingIdx},
                                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Long))),
                        zeroRow);
            }

            return embedImpl;
        }
    }
}
