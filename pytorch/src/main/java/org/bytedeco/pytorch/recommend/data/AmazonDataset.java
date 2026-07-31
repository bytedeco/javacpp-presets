/*
 * Ported from torch-rechub-scala: torchrec/data/AmazonDataset.scala
 *
 * Amazon product reviews → MatchingDataset (native Dataset) with negative sampling.
 * Tries Fine Food Reviews CSV; synthetic fallback.
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
public final class AmazonDataset {

    private static final String FOOD_URL =
            "https://raw.githubusercontent.com/makefu/amazon-fine-food-reviews/master/Reviews.csv";

    private AmazonDataset() {}

    public static final class Split {
        public final MatchingDataset train, val, test;
        public final int numUsers, numItems;
        public Split(MatchingDataset train, MatchingDataset val, MatchingDataset test,
                     int numUsers, int numItems) {
            this.train = train; this.val = val; this.test = test;
            this.numUsers = numUsers; this.numItems = numItems;
        }
    }

    public static Split load() { return load("food", 0.8f, 4, 50_000, 42); }

    public static Split load(String category, float trainRatio, int negRatio,
                             int maxReviews, int seed) {
        System.out.println("============================================================");
        System.out.println("Amazon " + category + " Reviews Dataset Loading");
        System.out.println("============================================================");

        File dataFile = tryDownload(category);
        if (dataFile == null || !dataFile.exists() || dataFile.length() < 1000) {
            System.out.println("  [Warn] Download failed — synthetic Amazon-like data.");
            MatchingSupport.MatchSplit s = MatchingSupport.synthetic(
                    10_000, 5_000, maxReviews > 0 ? maxReviews : 50_000, trainRatio, negRatio, seed);
            return new Split(s.train, s.val, s.test, s.numUsers, s.numItems);
        }

        System.out.println("  [Parse] Reading reviews...");
        List<String[]> raw = parseReviews(dataFile, maxReviews > 0 ? maxReviews : 100_000);
        System.out.println("  [Data] Reviews loaded: " + raw.size());
        if (raw.isEmpty()) {
            MatchingSupport.MatchSplit s = MatchingSupport.synthetic(
                    10_000, 5_000, 50_000, trainRatio, negRatio, seed);
            return new Split(s.train, s.val, s.test, s.numUsers, s.numItems);
        }

        // raw[i] = {user, item}
        TreeSet<String> users = new TreeSet<>();
        TreeSet<String> items = new TreeSet<>();
        for (String[] r : raw) { users.add(r[0]); items.add(r[1]); }
        Map<String, Integer> userMap = new HashMap<>();
        Map<String, Integer> itemMap = new HashMap<>();
        int i = 0; for (String u : users) userMap.put(u, i++);
        i = 0; for (String it : items) itemMap.put(it, i++);

        List<MatchingSupport.Pair> pairs = new ArrayList<>(raw.size());
        for (String[] r : raw) {
            Integer u = userMap.get(r[0]);
            Integer it = itemMap.get(r[1]);
            if (u != null && it != null) pairs.add(new MatchingSupport.Pair(u, it, 1f));
        }
        System.out.println("  [Data] Users: " + userMap.size() + ", Items: " + itemMap.size());
        MatchingSupport.MatchSplit s = MatchingSupport.splitAndBuild(
                pairs, userMap.size(), itemMap.size(), trainRatio, negRatio, seed);
        System.out.println("  [Dataset] Train size: " + s.train.sizeLong());
        System.out.println("============================================================");
        return new Split(s.train, s.val, s.test, s.numUsers, s.numItems);
    }

    private static File tryDownload(String category) {
        try {
            System.out.println("  [Try] " + FOOD_URL);
            return DatasetDownloader.download(FOOD_URL, "amazon_" + category, false);
        } catch (Throwable t) {
            System.out.println("  [Fail] " + t.getMessage());
            return null;
        }
    }

    /** CSV: Id,ProductId,UserId,ProfileName,HelpfulnessNumerator,HelpfulnessDenominator,Score,Time,Summary,Text */
    private static List<String[]> parseReviews(File file, int max) {
        List<String[]> out = new ArrayList<>();
        try (DatasetDownloader.LineIterator it =
                     DatasetDownloader.readLines(file, ",", true, max)) {
            while (it.hasNext() && out.size() < max) {
                String[] f = it.next();
                // Need at least ProductId(1) and UserId(2)
                if (f.length >= 3) {
                    String item = f[1].trim();
                    String user = f[2].trim();
                    if (!user.isEmpty() && !item.isEmpty()) {
                        out.add(new String[]{user, item});
                    }
                }
            }
        } catch (Exception e) {
            System.out.println("  [Parse error] " + e.getMessage());
        }
        return out;
    }
}
