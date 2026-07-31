/*
 * Ported from torch-rechub-scala: torchrec/data/IMDBDataset.scala
 *
 * IMDB reviews as matching dataset (implicit feedback from sentiment/reviews).
 * Download CSV when available; synthetic fallback.
 */
package org.bytedeco.pytorch.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class IMDBDataset {

    private static final String REVIEWS_URL =
            "https://raw.githubusercontent.com/MuhammedBuyukkinaci/IMDB-Dataset/master/IMDB%20Dataset.csv";

    private IMDBDataset() {}

    public static final class Split {
        public final MatchingDataset train, val, test;
        public final int numUsers, numItems;
        public Split(MatchingDataset train, MatchingDataset val, MatchingDataset test,
                     int numUsers, int numItems) {
            this.train = train; this.val = val; this.test = test;
            this.numUsers = numUsers; this.numItems = numItems;
        }
    }

    public static Split load() { return load(0.8f, 4, 50_000, 42); }

    public static Split load(float trainRatio, int negRatio, int maxReviews, int seed) {
        System.out.println("============================================================");
        System.out.println("IMDB Movie Rating Dataset Loading");
        System.out.println("============================================================");

        File dataFile = tryDownload();
        List<MatchingSupport.Pair> pairs = new ArrayList<>();
        int numUsers = 0, numItems = 0;

        if (dataFile != null && dataFile.exists() && dataFile.length() > 1000) {
            System.out.println("  [Parse] Reading IMDB reviews...");
            // Sentiment CSV has no user/item — synthesize pseudo user/item from row hash
            // so the matching API still works for two-tower demos.
            pairs = parseAsImplicit(dataFile, maxReviews > 0 ? maxReviews : 50_000, seed);
            TreeSet<Integer> us = new TreeSet<>();
            TreeSet<Integer> is = new TreeSet<>();
            for (MatchingSupport.Pair p : pairs) { us.add(p.user); is.add(p.item); }
            numUsers = us.size();
            numItems = is.size();
            System.out.println("  [Data] Reviews: " + pairs.size()
                    + "  pseudo-users: " + numUsers + "  pseudo-items: " + numItems);
        }

        if (pairs.isEmpty()) {
            System.out.println("  [Warn] No IMDB rows — synthetic fallback.");
            MatchingSupport.MatchSplit s = MatchingSupport.synthetic(
                    5_000, 2_000, maxReviews > 0 ? maxReviews : 50_000, trainRatio, negRatio, seed);
            return new Split(s.train, s.val, s.test, s.numUsers, s.numItems);
        }

        MatchingSupport.MatchSplit s = MatchingSupport.splitAndBuild(
                pairs, numUsers, numItems, trainRatio, negRatio, seed);
        System.out.println("============================================================");
        return new Split(s.train, s.val, s.test, s.numUsers, s.numItems);
    }

    private static File tryDownload() {
        try {
            System.out.println("  [Try] " + REVIEWS_URL);
            return DatasetDownloader.download(REVIEWS_URL, "imdb_reviews", false);
        } catch (Throwable t) {
            System.out.println("  [Fail] " + t.getMessage());
            return null;
        }
    }

    /**
     * IMDB 50k sentiment CSV: review,sentiment.
     * Map each positive review to a (user, item) pair via deterministic hashes so
     * retrieval trainers have a usable MatchingDataset without external user ids.
     */
    private static List<MatchingSupport.Pair> parseAsImplicit(File file, int max, int seed) {
        List<MatchingSupport.Pair> out = new ArrayList<>();
        Random rng = new Random(seed);
        int userVocab = 5_000;
        int itemVocab = 2_000;
        try (DatasetDownloader.LineIterator it =
                     DatasetDownloader.readLines(file, ",", true, max)) {
            while (it.hasNext() && out.size() < max) {
                String[] f = it.next();
                if (f.length < 2) continue;
                String sentiment = f[f.length - 1].trim().toLowerCase();
                boolean pos = sentiment.contains("pos");
                if (!pos && !sentiment.contains("neg")) {
                    // maybe unquoted; treat non-empty as positive sample generator
                    pos = rng.nextFloat() < 0.5f;
                }
                if (!pos) continue; // only positive interactions for matching
                String review = f[0];
                int user = Math.floorMod(hash(review) ^ seed, userVocab);
                int item = Math.floorMod(hash(review + "|item") * 31 + seed, itemVocab);
                out.add(new MatchingSupport.Pair(user, item, 1f));
            }
        } catch (Exception e) {
            System.out.println("  [Parse error] " + e.getMessage());
        }
        // Remap to contiguous
        TreeSet<Integer> us = new TreeSet<>();
        TreeSet<Integer> is = new TreeSet<>();
        for (MatchingSupport.Pair p : out) { us.add(p.user); is.add(p.item); }
        Map<Integer, Integer> um = new HashMap<>();
        Map<Integer, Integer> im = new HashMap<>();
        int i = 0; for (Integer u : us) um.put(u, i++);
        i = 0; for (Integer it : is) im.put(it, i++);
        List<MatchingSupport.Pair> remapped = new ArrayList<>(out.size());
        for (MatchingSupport.Pair p : out) {
            remapped.add(new MatchingSupport.Pair(um.get(p.user), im.get(p.item), 1f));
        }
        return remapped;
    }

    private static int hash(String s) {
        if (s == null) return 0;
        int h = 0;
        for (int i = 0; i < s.length(); i++) h = 31 * h + s.charAt(i);
        return h;
    }
}
