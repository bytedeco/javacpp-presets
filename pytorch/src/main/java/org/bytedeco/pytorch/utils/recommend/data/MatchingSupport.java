/*
 * Shared helpers for matching-style (user, item, label) datasets with negative sampling.
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch.ScalarType;

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

final class MatchingSupport {

    private MatchingSupport() {}

    static final class Pair {
        final int user;
        final int item;
        final float label;
        Pair(int user, int item, float label) {
            this.user = user;
            this.item = item;
            this.label = label;
        }
    }

    static void buildVocab(List<int[]> userItemPairs,
                           Map<String, Integer> userMap,
                           Map<String, Integer> itemMap,
                           List<String> rawUsers,
                           List<String> rawItems) {
        TreeSet<String> users = new TreeSet<>();
        TreeSet<String> items = new TreeSet<>();
        for (String u : rawUsers) users.add(u);
        for (String i : rawItems) items.add(i);
        int i = 0;
        for (String u : users) userMap.put(u, i++);
        i = 0;
        for (String it : items) itemMap.put(it, i++);
    }

    static Map<Integer, Set<Integer>> userPos(List<Pair> pairs) {
        Map<Integer, Set<Integer>> m = new HashMap<>();
        for (Pair p : pairs) {
            if (p.label > 0.5f) {
                m.computeIfAbsent(p.user, k -> new HashSet<>()).add(p.item);
            }
        }
        return m;
    }

    static MatchingDataset buildWithNegatives(
            List<Pair> positives,
            Map<Integer, Set<Integer>> userPosItems,
            int negRatio,
            int numItems,
            int seed) {
        Random rng = new Random(seed);
        int cap = positives.size() * (1 + Math.max(negRatio, 0));
        float[] uArr = new float[cap];
        float[] iArr = new float[cap];
        float[] yArr = new float[cap];
        int idx = 0;
        for (Pair p : positives) {
            uArr[idx] = p.user;
            iArr[idx] = p.item;
            yArr[idx] = 1f;
            idx++;
            Set<Integer> pos = userPosItems.getOrDefault(p.user, Collections.emptySet());
            int neg = 0, attempts = 0;
            while (neg < negRatio && attempts < numItems * 2 + 10) {
                int ni = rng.nextInt(Math.max(numItems, 1));
                if (!pos.contains(ni)) {
                    uArr[idx] = p.user;
                    iArr[idx] = ni;
                    yArr[idx] = 0f;
                    idx++;
                    neg++;
                }
                attempts++;
            }
        }
        float[] u = new float[idx], i = new float[idx], y = new float[idx];
        System.arraycopy(uArr, 0, u, 0, idx);
        System.arraycopy(iArr, 0, i, 0, idx);
        System.arraycopy(yArr, 0, y, 0, idx);
        Map<String, Tensor> userF = new LinkedHashMap<>();
        userF.put("user_id", RecommendDataset.floatFeature(u).toType(ScalarType.Long));
        Map<String, Tensor> itemF = new LinkedHashMap<>();
        itemF.put("item_id", RecommendDataset.floatFeature(i).toType(ScalarType.Long));
        return new MatchingDataset(userF, itemF, RecommendDataset.floatFeature(y));
    }

    /** Split list of positive pairs into train/val/test MatchingDatasets. */
    static class MatchSplit {
        final MatchingDataset train, val, test;
        final int numUsers, numItems;
        MatchSplit(MatchingDataset train, MatchingDataset val, MatchingDataset test,
                   int numUsers, int numItems) {
            this.train = train; this.val = val; this.test = test;
            this.numUsers = numUsers; this.numItems = numItems;
        }
    }

    static MatchSplit splitAndBuild(
            List<Pair> remapped,
            int numUsers, int numItems,
            float trainRatio, int negRatio, int seed) {
        Map<Integer, Set<Integer>> pos = userPos(remapped);
        Random rng = new Random(seed);
        List<Pair> shuffled = new ArrayList<>(remapped);
        Collections.shuffle(shuffled, rng);
        int trainSize = (int) (shuffled.size() * trainRatio);
        int valSize = (shuffled.size() - trainSize) / 2;
        MatchingDataset train = buildWithNegatives(shuffled.subList(0, trainSize), pos, negRatio, numItems, seed);
        MatchingDataset val = buildWithNegatives(shuffled.subList(trainSize, trainSize + valSize), pos, negRatio, numItems, seed + 1);
        MatchingDataset test = buildWithNegatives(shuffled.subList(trainSize + valSize, shuffled.size()), pos, negRatio, numItems, seed + 2);
        return new MatchSplit(train, val, test, numUsers, numItems);
    }

    static MatchSplit synthetic(int numUsers, int numItems, int numSamples,
                                float trainRatio, int negRatio, int seed) {
        Random rng = new Random(seed);
        List<Pair> pairs = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            pairs.add(new Pair(rng.nextInt(numUsers), rng.nextInt(numItems), 1f));
        }
        return splitAndBuild(pairs, numUsers, numItems, trainRatio, negRatio, seed);
    }
}
