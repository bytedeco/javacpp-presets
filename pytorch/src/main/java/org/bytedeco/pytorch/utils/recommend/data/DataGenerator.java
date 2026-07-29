/*
 * Ported from torch-rechub-scala: torchrec/data/DataGenerator.scala
 *
 * Synthetic dataset factories for ranking / matching / multi-task demos.
 * All returned datasets extend native org.bytedeco.pytorch.data.Dataset.
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DataGenerator {

    private DataGenerator() {}

    public static final class TensorSplit {
        public final TensorDataset train, val, test;
        public TensorSplit(TensorDataset train, TensorDataset val, TensorDataset test) {
            this.train = train; this.val = val; this.test = test;
        }
    }

    public static final class MatchingSplit {
        public final MatchingDataset train, val, test;
        public final int numUsers, numItems;
        public MatchingSplit(MatchingDataset train, MatchingDataset val, MatchingDataset test,
                             int numUsers, int numItems) {
            this.train = train; this.val = val; this.test = test;
            this.numUsers = numUsers; this.numItems = numItems;
        }
    }

    /** MovieLens-style sparse user/item ranking as TensorDataset. */
    public static TensorSplit generateMovieLensData() {
        return generateMovieLensData(100_000, 6040, 3952, 0.8f, 42);
    }

    public static TensorSplit generateMovieLensData(
            int numSamples, int numUsers, int numMovies, float trainRatio, int seed) {
        Random rng = new Random(seed);
        System.out.println("Generating MovieLens-style data (" + numSamples + " samples)...");
        float[] userIds = new float[numSamples];
        float[] movieIds = new float[numSamples];
        float[] ratings = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            userIds[i] = rng.nextInt(numUsers);
            movieIds[i] = rng.nextInt(numMovies);
            ratings[i] = rng.nextFloat() < 0.2f ? 1f : 0f;
        }
        Map<String, Tensor> sparse = new LinkedHashMap<>();
        sparse.put("user_id", RecommendDataset.floatFeature(userIds).toType(ScalarType.Long));
        sparse.put("movie_id", RecommendDataset.floatFeature(movieIds).toType(ScalarType.Long));
        TensorDataset full = new TensorDataset(sparse, Collections.emptyMap(),
                RecommendDataset.floatFeature(ratings));
        return splitTensor(full, trainRatio);
    }

    /** Criteo-style CTR (delegates to CriteoDataset.generateSynthetic). */
    public static CriteoDataset.Split generateCriteoData() {
        return CriteoDataset.generateSynthetic(0.8f, 100_000, 42);
    }

    public static CriteoDataset.Split generateCriteoData(int numSamples, float trainRatio, int seed) {
        return CriteoDataset.generateSynthetic(trainRatio, numSamples, seed);
    }

    /** Census-Income style classification. */
    public static CensusIncomeDataset.Split generateCensusData() {
        return CensusIncomeDataset.generateSynthetic(0.8f, 50_000, 42);
    }

    public static CensusIncomeDataset.Split generateCensusData(int numSamples, float trainRatio, int seed) {
        return CensusIncomeDataset.generateSynthetic(trainRatio, numSamples, seed);
    }

    /** Two-tower matching with negatives. */
    public static MatchingSplit generateMatchingData() {
        return generateMatchingData(50_000, 6040, 3952, 0.8f, 4, 42);
    }

    public static MatchingSplit generateMatchingData(
            int numSamples, int numUsers, int numItems,
            float trainRatio, int negRatio, int seed) {
        MatchingSupport.MatchSplit s = MatchingSupport.synthetic(
                numUsers, numItems, numSamples, trainRatio, negRatio, seed);
        return new MatchingSplit(s.train, s.val, s.test, s.numUsers, s.numItems);
    }

    /** Multi-task click/conversion. */
    public static AliExpressDataset.Split generateMultiTaskData() {
        return AliExpressDataset.generateSynthetic(0.8f, 50_000, 42,
                new String[]{"click", "conversion"});
    }

    public static AliExpressDataset.Split generateMultiTaskData(int numSamples, int seed) {
        return AliExpressDataset.generateSynthetic(0.8f, numSamples, seed,
                new String[]{"click", "conversion"});
    }

    /** Sequence dataset: tokens [B,S] + target + label. */
    public static SequenceDataset generateSequenceData(int numSamples, int seqLen,
                                                       int vocabSize, int seed) {
        Random rng = new Random(seed);
        long[] tokens = new long[numSamples * seqLen];
        long[] targets = new long[numSamples];
        float[] labels = new float[numSamples];
        for (int b = 0; b < numSamples; b++) {
            for (int t = 0; t < seqLen; t++) {
                tokens[b * seqLen + t] = rng.nextInt(vocabSize);
            }
            targets[b] = rng.nextInt(vocabSize);
            labels[b] = rng.nextFloat() < 0.3f ? 1f : 0f;
        }
        Tensor tokenT = RecommendDataset.longFeature(tokens).reshape(numSamples, seqLen);
        Tensor targetT = RecommendDataset.longFeature(targets);
        Tensor labelT = RecommendDataset.floatFeature(labels);
        Map<String, Tensor> seq = new LinkedHashMap<>();
        seq.put("item_seq", tokenT);
        return new SequenceDataset(
                Collections.emptyMap(), seq, labelT, null, null, tokenT, targetT, null);
    }

    private static TensorSplit splitTensor(TensorDataset full, float trainRatio) {
        long n = full.sizeLong();
        long trainSize = (long) (n * trainRatio);
        long valSize = (n - trainSize) / 2;
        long testSize = n - trainSize - valSize;
        return new TensorSplit(
                full.slice(0, trainSize),
                full.slice(trainSize, valSize),
                full.slice(trainSize + valSize, testSize));
    }
}
