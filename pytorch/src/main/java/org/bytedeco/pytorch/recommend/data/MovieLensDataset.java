/*
 * Ported from torch-rechub-scala: torchrec/data/MovieLensDataset.scala
 *
 * MovieLens 1M matching dataset for two-tower retrieval (DSSM).
 * Download ml-1m.zip → parse ratings.dat → vocab remap → negative sampling
 * → MatchingDataset (extends native Dataset).
 */
package org.bytedeco.pytorch.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class MovieLensDataset {

    private static final String DATASET_NAME = "ml-1m";
    private static final String[] MIRRORS = {
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "https://raw.githubusercontent.com/makefu/movielens-1m/master/ml-1m.zip",
    };

    private MovieLensDataset() {}

    public static final class Split {
        public final MatchingDataset train;
        public final MatchingDataset val;
        public final MatchingDataset test;
        public final int numUsers;
        public final int numMovies;

        public Split(MatchingDataset train, MatchingDataset val, MatchingDataset test,
                     int numUsers, int numMovies) {
            this.train = train;
            this.val = val;
            this.test = test;
            this.numUsers = numUsers;
            this.numMovies = numMovies;
        }
    }

    public static Split load() {
        return load(0.8f, 4, -1, 42);
    }

    /**
     * @param trainRatio fraction for train
     * @param negRatio   negatives per positive
     * @param maxSamples cap total ratings (-1 = all)
     * @param seed       RNG seed
     */
    public static Split load(float trainRatio, int negRatio, int maxSamples, int seed) {
        System.out.println("============================================================");
        System.out.println("MovieLens 1M Dataset Loading");
        System.out.println("============================================================");

        File zipOrDir = DatasetDownloader.tryMirrors(MIRRORS, DATASET_NAME);
        File dataDir = zipOrDir.isDirectory() ? zipOrDir : zipOrDir.getParentFile();
        File ratingsFile = findRatingsFile(dataDir);
        if (ratingsFile == null) {
            // try cache root
            ratingsFile = findRatingsFile(DatasetDownloader.cacheDir());
        }
        if (ratingsFile == null) {
            System.out.println("  [Warn] ratings.dat not found. Generating synthetic MovieLens-style data.");
            return generateSynthetic(trainRatio, negRatio, maxSamples > 0 ? maxSamples : 50_000, seed);
        }
        System.out.println("  [Data] ratings: " + ratingsFile.getAbsolutePath());

        System.out.println("  [Parse] Reading ratings...");
        List<Rating> allRatings = parseRatings(ratingsFile, maxSamples);
        System.out.println("  [Data] Total ratings: " + allRatings.size());

        Map<Integer, Integer> userMap = new HashMap<>();
        Map<Integer, Integer> movieMap = new HashMap<>();
        buildVocabularies(allRatings, userMap, movieMap);
        int numUsers = userMap.size();
        int numMovies = movieMap.size();
        System.out.println("  [Data] Unique users: " + numUsers + ", movies: " + numMovies);

        // Remap ratings to contiguous ids
        List<Rating> remapped = new ArrayList<>(allRatings.size());
        for (Rating r : allRatings) {
            Integer u = userMap.get(r.userId);
            Integer m = movieMap.get(r.movieId);
            if (u != null && m != null) {
                remapped.add(new Rating(u, m, r.rating, r.timestamp));
            }
        }

        Map<Integer, Set<Integer>> userPosItems = new HashMap<>();
        for (Rating r : remapped) {
            userPosItems.computeIfAbsent(r.userId, k -> new HashSet<>()).add(r.movieId);
        }

        Random rng = new Random(seed);
        Collections.shuffle(remapped, rng);
        int trainSize = (int) (remapped.size() * trainRatio);
        int valSize = (remapped.size() - trainSize) / 2;
        List<Rating> trainRatings = remapped.subList(0, trainSize);
        List<Rating> valRatings = remapped.subList(trainSize, trainSize + valSize);
        List<Rating> testRatings = remapped.subList(trainSize + valSize, remapped.size());

        System.out.println("  [Split] Train: " + trainRatings.size()
                + ", Val: " + valRatings.size() + ", Test: " + testRatings.size());

        MatchingDataset trainDS = buildMatchingDataset(trainRatings, userPosItems, negRatio, numMovies, seed);
        MatchingDataset valDS = buildMatchingDataset(valRatings, userPosItems, negRatio, numMovies, seed + 1);
        MatchingDataset testDS = buildMatchingDataset(testRatings, userPosItems, negRatio, numMovies, seed + 2);

        System.out.println("  [Dataset] Train size: " + trainDS.sizeLong());
        System.out.println("============================================================");
        return new Split(trainDS, valDS, testDS, numUsers, numMovies);
    }

    private static File findRatingsFile(File dir) {
        if (dir == null) return null;
        File[] candidates = {
                new File(dir, "ml-1m/ratings.dat"),
                new File(dir, "ratings.dat"),
                new File(dir, "ml-1m.zip"),
        };
        for (File f : candidates) {
            if (f.isFile() && f.getName().endsWith("ratings.dat") && f.exists()) return f;
        }
        // walk one level
        File[] kids = dir.listFiles();
        if (kids != null) {
            for (File k : kids) {
                if (k.isDirectory()) {
                    File r = new File(k, "ratings.dat");
                    if (r.exists()) return r;
                } else if (k.getName().equals("ratings.dat")) {
                    return k;
                }
            }
        }
        return null;
    }

    private static final class Rating {
        final int userId;
        final int movieId;
        final float rating;
        final long timestamp;

        Rating(int userId, int movieId, float rating, long timestamp) {
            this.userId = userId;
            this.movieId = movieId;
            this.rating = rating;
            this.timestamp = timestamp;
        }
    }

    private static List<Rating> parseRatings(File file, int maxSamples) {
        List<Rating> out = new ArrayList<>();
        long max = maxSamples > 0 ? maxSamples : Long.MAX_VALUE;
        try (DatasetDownloader.LineIterator lines =
                     DatasetDownloader.readLines(file, "::", false, max)) {
            while (lines.hasNext()) {
                String[] fields = lines.next();
                if (fields.length >= 4) {
                    try {
                        out.add(new Rating(
                                Integer.parseInt(fields[0]),
                                Integer.parseInt(fields[1]),
                                Float.parseFloat(fields[2]),
                                Long.parseLong(fields[3])));
                    } catch (NumberFormatException ignored) {
                    }
                }
                if (out.size() >= max) break;
            }
        } catch (Exception e) {
            System.out.println("  [Parse error] " + e.getMessage());
        }
        return out;
    }

    private static void buildVocabularies(
            List<Rating> ratings,
            Map<Integer, Integer> userMap,
            Map<Integer, Integer> movieMap) {
        TreeSet<Integer> users = new TreeSet<>();
        TreeSet<Integer> movies = new TreeSet<>();
        for (Rating r : ratings) {
            users.add(r.userId);
            movies.add(r.movieId);
        }
        int i = 0;
        for (Integer u : users) userMap.put(u, i++);
        i = 0;
        for (Integer m : movies) movieMap.put(m, i++);
    }

    private static MatchingDataset buildMatchingDataset(
            List<Rating> ratings,
            Map<Integer, Set<Integer>> userPosItems,
            int negRatio,
            int numMovies,
            int seed) {
        Random rng = new Random(seed);
        int capacity = ratings.size() * (1 + Math.max(negRatio, 0));
        float[] userFeatArr = new float[capacity];
        float[] movieFeatArr = new float[capacity];
        float[] labelArr = new float[capacity];
        int idx = 0;

        for (Rating rating : ratings) {
            userFeatArr[idx] = rating.userId;
            movieFeatArr[idx] = rating.movieId;
            labelArr[idx] = 1.0f;
            idx++;

            Set<Integer> posSet = userPosItems.getOrDefault(rating.userId, Collections.emptySet());
            int negCount = 0;
            int attempts = 0;
            while (negCount < negRatio && attempts < numMovies * 2 + 10) {
                int negMovie = rng.nextInt(Math.max(numMovies, 1));
                if (!posSet.contains(negMovie)) {
                    userFeatArr[idx] = rating.userId;
                    movieFeatArr[idx] = negMovie;
                    labelArr[idx] = 0.0f;
                    idx++;
                    negCount++;
                }
                attempts++;
            }
        }

        float[] u = new float[idx];
        float[] m = new float[idx];
        float[] y = new float[idx];
        System.arraycopy(userFeatArr, 0, u, 0, idx);
        System.arraycopy(movieFeatArr, 0, m, 0, idx);
        System.arraycopy(labelArr, 0, y, 0, idx);

        Map<String, Tensor> userFeatures = new LinkedHashMap<>();
        userFeatures.put("user_id", RecommendDataset.floatFeature(u).toType(ScalarType.Long));
        Map<String, Tensor> itemFeatures = new LinkedHashMap<>();
        itemFeatures.put("movie_id", RecommendDataset.floatFeature(m).toType(ScalarType.Long));
        Tensor labels = RecommendDataset.floatFeature(y);
        return new MatchingDataset(userFeatures, itemFeatures, labels);
    }

    /** Synthetic fallback when download/parse fails. */
    public static Split generateSynthetic(float trainRatio, int negRatio, int numSamples, int seed) {
        Random rng = new Random(seed);
        int numUsers = 6040;
        int numMovies = 3952;
        List<Rating> ratings = new ArrayList<>(numSamples);
        Map<Integer, Set<Integer>> userPos = new HashMap<>();
        for (int i = 0; i < numSamples; i++) {
            int u = rng.nextInt(numUsers);
            int m = rng.nextInt(numMovies);
            ratings.add(new Rating(u, m, rng.nextInt(5) + 1f, i));
            userPos.computeIfAbsent(u, k -> new HashSet<>()).add(m);
        }
        Collections.shuffle(ratings, rng);
        int trainSize = (int) (ratings.size() * trainRatio);
        int valSize = (ratings.size() - trainSize) / 2;
        MatchingDataset train = buildMatchingDataset(ratings.subList(0, trainSize), userPos, negRatio, numMovies, seed);
        MatchingDataset val = buildMatchingDataset(ratings.subList(trainSize, trainSize + valSize), userPos, negRatio, numMovies, seed + 1);
        MatchingDataset test = buildMatchingDataset(ratings.subList(trainSize + valSize, ratings.size()), userPos, negRatio, numMovies, seed + 2);
        return new Split(train, val, test, numUsers, numMovies);
    }
}
