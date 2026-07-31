/*
 * Ported from torch-rechub-scala: torchrec/data/NetflixDataset.scala
 *
 * Netflix Prize style matching dataset. Local files preferred; synthetic fallback
 * (official data requires Kaggle manual download into ~/.torchrec-datasets).
 */
package org.bytedeco.pytorch.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class NetflixDataset {

    private NetflixDataset() {}

    public static final class Split {
        public final MatchingDataset train, val, test;
        public final int numUsers, numMovies;
        public Split(MatchingDataset train, MatchingDataset val, MatchingDataset test,
                     int numUsers, int numMovies) {
            this.train = train; this.val = val; this.test = test;
            this.numUsers = numUsers; this.numMovies = numMovies;
        }
    }

    public static Split load() { return load(0.8f, 4, 100_000, 42); }

    public static Split load(float trainRatio, int negRatio, int maxRatings, int seed) {
        System.out.println("============================================================");
        System.out.println("Netflix Prize Dataset Loading");
        System.out.println("============================================================");

        List<int[]> ratings = tryLoadLocal(maxRatings > 0 ? maxRatings : 10_000_000);
        System.out.println("  [Data] Ratings loaded: " + ratings.size());
        if (ratings.isEmpty()) {
            System.out.println("  [Warn] No local Netflix files — synthetic fallback.");
            // Classic Netflix-ish scale (downsampled)
            MatchingSupport.MatchSplit s = MatchingSupport.synthetic(
                    50_000, 10_000, maxRatings > 0 ? maxRatings : 200_000, trainRatio, negRatio, seed);
            return new Split(s.train, s.val, s.test, s.numUsers, s.numItems);
        }

        TreeSet<Integer> users = new TreeSet<>();
        TreeSet<Integer> movies = new TreeSet<>();
        for (int[] r : ratings) { users.add(r[0]); movies.add(r[1]); }
        Map<Integer, Integer> userMap = new HashMap<>();
        Map<Integer, Integer> movieMap = new HashMap<>();
        int i = 0; for (Integer u : users) userMap.put(u, i++);
        i = 0; for (Integer m : movies) movieMap.put(m, i++);

        List<MatchingSupport.Pair> pairs = new ArrayList<>(ratings.size());
        for (int[] r : ratings) {
            pairs.add(new MatchingSupport.Pair(userMap.get(r[0]), movieMap.get(r[1]), 1f));
        }
        System.out.println("  [Data] Users: " + userMap.size() + ", Movies: " + movieMap.size());
        MatchingSupport.MatchSplit s = MatchingSupport.splitAndBuild(
                pairs, userMap.size(), movieMap.size(), trainRatio, negRatio, seed);
        System.out.println("============================================================");
        return new Split(s.train, s.val, s.test, s.numUsers, s.numItems);
    }

    /**
     * Look for probe / combined files under cache:
     *   ~/.torchrec-datasets/netflix/  or ratings with MovieID::CustomerID::Rating::Date
     */
    private static List<int[]> tryLoadLocal(int max) {
        List<int[]> out = new ArrayList<>();
        File[] roots = {
                new File(DatasetDownloader.cacheDir(), "netflix"),
                new File(DatasetDownloader.cacheDir(), "netflix-prize-data"),
                DatasetDownloader.cacheDir(),
        };
        for (File root : roots) {
            if (root == null || !root.exists()) continue;
            collectRatings(root, out, max);
            if (!out.isEmpty()) break;
        }
        return out;
    }

    private static void collectRatings(File dir, List<int[]> out, int max) {
        File[] kids = dir.listFiles();
        if (kids == null) return;
        for (File f : kids) {
            if (out.size() >= max) return;
            if (f.isDirectory()) {
                collectRatings(f, out, max);
            } else if (f.getName().endsWith(".txt") || f.getName().contains("probe")
                    || f.getName().contains("rating")) {
                parseFile(f, out, max);
            }
        }
    }

    private static void parseFile(File file, List<int[]> out, int max) {
        // Support both "MovieID:" headers (Netflix combined_data) and
        // "MovieID::CustomerID::Rating::Date"
        int currentMovie = -1;
        try (DatasetDownloader.LineIterator it =
                     DatasetDownloader.readLines(file, "::", false, max)) {
            // LineIterator always splits — for combined_data use "\n" style single field
        } catch (Exception ignored) {
        }
        try (DatasetDownloader.LineIterator it =
                     DatasetDownloader.readLines(file, "\n", false, max * 2L)) {
            while (it.hasNext() && out.size() < max) {
                String[] fields = it.next();
                if (fields.length == 0) continue;
                String line = fields[0].trim();
                if (line.isEmpty()) continue;
                if (line.endsWith(":")) {
                    try {
                        currentMovie = Integer.parseInt(line.substring(0, line.length() - 1));
                    } catch (NumberFormatException ignored) {
                        currentMovie = -1;
                    }
                    continue;
                }
                if (line.contains("::")) {
                    String[] p = line.split("::");
                    if (p.length >= 3) {
                        try {
                            int movie = Integer.parseInt(p[0]);
                            int user = Integer.parseInt(p[1]);
                            out.add(new int[]{user, movie});
                        } catch (NumberFormatException ignored) {
                        }
                    }
                } else if (currentMovie > 0) {
                    // customerId, rating, date
                    String[] p = line.split(",");
                    if (p.length >= 1) {
                        try {
                            int user = Integer.parseInt(p[0].trim());
                            out.add(new int[]{user, currentMovie});
                        } catch (NumberFormatException ignored) {
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("  [Parse] " + file.getName() + ": " + e.getMessage());
        }
    }
}
