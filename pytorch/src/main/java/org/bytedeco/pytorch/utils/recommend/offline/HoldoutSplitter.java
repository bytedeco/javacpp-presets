/*
 * Dataset splitting for offline evaluation without leakage.
 *
 * Industry rules of thumb (Netflix, YouTube, Taobao, TikTok):
 *   1. Prefer time-based split over random — random leaks future interactions.
 *   2. User holdout: leave some users entirely in test (cold-start eval).
 *   3. Temporal global split: train on [T0, T1), test on [T1, T2).
 *   4. Session split: last session per user as test (sequential recsys).
 *   5. Never put the same (user, item, timestamp) event in both sides.
 */
package org.bytedeco.pytorch.utils.recommend.offline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

/** Holdout / split utilities for recommend offline eval. */
public final class HoldoutSplitter {

    private HoldoutSplitter() {}

    /** Minimal event record for splitting. */
    public static final class Event {
        public final String userId;
        public final String itemId;
        public final long timestampMs;
        public final float label;
        public final Map<String, String> features;

        public Event(String userId, String itemId, long timestampMs, float label) {
            this(userId, itemId, timestampMs, label, Collections.emptyMap());
        }

        public Event(
                String userId,
                String itemId,
                long timestampMs,
                float label,
                Map<String, String> features) {
            this.userId = Objects.requireNonNull(userId, "userId");
            this.itemId = Objects.requireNonNull(itemId, "itemId");
            this.timestampMs = timestampMs;
            this.label = label;
            this.features = features == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(features));
        }
    }

    public static final class Split {
        public final List<Event> train;
        public final List<Event> test;
        public final String strategy;
        public final Map<String, String> meta;

        public Split(List<Event> train, List<Event> test, String strategy, Map<String, String> meta) {
            this.train = Collections.unmodifiableList(new ArrayList<>(train));
            this.test = Collections.unmodifiableList(new ArrayList<>(test));
            this.strategy = strategy;
            this.meta = meta == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(meta));
        }

        public int trainSize() {
            return train.size();
        }

        public int testSize() {
            return test.size();
        }

        @Override
        public String toString() {
            return "Split{strategy=" + strategy + ", train=" + train.size()
                    + ", test=" + test.size() + ", meta=" + meta + "}";
        }
    }

    /**
     * Global time-based split: events with timestamp &lt; cutoff go to train,
     * timestamp &gt;= cutoff go to test.
     */
    public static Split byTimestamp(List<Event> events, long cutoffTimestampMs) {
        Objects.requireNonNull(events, "events");
        List<Event> train = new ArrayList<>();
        List<Event> test = new ArrayList<>();
        for (Event e : events) {
            if (e.timestampMs < cutoffTimestampMs) {
                train.add(e);
            } else {
                test.add(e);
            }
        }
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("cutoffTimestampMs", String.valueOf(cutoffTimestampMs));
        return new Split(train, test, "timestamp", meta);
    }

    /**
     * Time-based split by quantile: last {@code testRatio} fraction of timeline is test.
     */
    public static Split byTimeRatio(List<Event> events, double testRatio) {
        if (testRatio <= 0.0 || testRatio >= 1.0) {
            throw new IllegalArgumentException("testRatio must be in (0,1)");
        }
        if (events.isEmpty()) {
            return new Split(Collections.emptyList(), Collections.emptyList(),
                    "time_ratio", Collections.emptyMap());
        }
        List<Event> sorted = new ArrayList<>(events);
        sorted.sort(Comparator.comparingLong(e -> e.timestampMs));
        int cutIndex = (int) Math.floor(sorted.size() * (1.0 - testRatio));
        cutIndex = Math.max(1, Math.min(sorted.size() - 1, cutIndex));
        long cutoff = sorted.get(cutIndex).timestampMs;
        // All events at exactly cutoff go to test to avoid ambiguous boundary.
        List<Event> train = new ArrayList<>();
        List<Event> test = new ArrayList<>();
        for (Event e : sorted) {
            if (e.timestampMs < cutoff) train.add(e);
            else test.add(e);
        }
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("testRatio", String.valueOf(testRatio));
        meta.put("cutoffTimestampMs", String.valueOf(cutoff));
        return new Split(train, test, "time_ratio", meta);
    }

    /**
     * Leave-last-K per user as test (sequential recommendation standard).
     * Users with fewer than K+1 events are dropped from test (kept in train only
     * if {@code keepSparseInTrain}).
     */
    public static Split leaveLastK(List<Event> events, int k, boolean keepSparseInTrain) {
        if (k < 1) throw new IllegalArgumentException("k must be >= 1");
        Map<String, List<Event>> byUser = groupByUser(events);
        List<Event> train = new ArrayList<>();
        List<Event> test = new ArrayList<>();
        for (Map.Entry<String, List<Event>> e : byUser.entrySet()) {
            List<Event> list = e.getValue();
            list.sort(Comparator.comparingLong(ev -> ev.timestampMs));
            if (list.size() <= k) {
                if (keepSparseInTrain) {
                    train.addAll(list);
                }
                continue;
            }
            int splitAt = list.size() - k;
            train.addAll(list.subList(0, splitAt));
            test.addAll(list.subList(splitAt, list.size()));
        }
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("k", String.valueOf(k));
        meta.put("keepSparseInTrain", String.valueOf(keepSparseInTrain));
        return new Split(train, test, "leave_last_k", meta);
    }

    /**
     * User-level holdout: a fraction of users entirely in test (cold-start protocol).
     */
    public static Split byUserHoldout(List<Event> events, double testUserRatio, long seed) {
        if (testUserRatio <= 0.0 || testUserRatio >= 1.0) {
            throw new IllegalArgumentException("testUserRatio must be in (0,1)");
        }
        Map<String, List<Event>> byUser = groupByUser(events);
        List<String> userIds = new ArrayList<>(byUser.keySet());
        Collections.sort(userIds);
        Random rng = new Random(seed);
        Collections.shuffle(userIds, rng);
        int nTestUsers = Math.max(1, (int) Math.round(userIds.size() * testUserRatio));
        Set<String> testUsers = new HashSet<>(userIds.subList(0, nTestUsers));
        List<Event> train = new ArrayList<>();
        List<Event> test = new ArrayList<>();
        for (Map.Entry<String, List<Event>> e : byUser.entrySet()) {
            if (testUsers.contains(e.getKey())) {
                test.addAll(e.getValue());
            } else {
                train.addAll(e.getValue());
            }
        }
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("testUserRatio", String.valueOf(testUserRatio));
        meta.put("testUsers", String.valueOf(testUsers.size()));
        meta.put("trainUsers", String.valueOf(userIds.size() - testUsers.size()));
        meta.put("seed", String.valueOf(seed));
        return new Split(train, test, "user_holdout", meta);
    }

    /**
     * Random event split — ONLY for debugging; leaks temporally. Documented as unsafe.
     */
    public static Split randomUnsafe(List<Event> events, double testRatio, long seed) {
        if (testRatio <= 0.0 || testRatio >= 1.0) {
            throw new IllegalArgumentException("testRatio must be in (0,1)");
        }
        List<Event> shuffled = new ArrayList<>(events);
        Collections.shuffle(shuffled, new Random(seed));
        int nTest = Math.max(1, (int) Math.round(shuffled.size() * testRatio));
        List<Event> test = new ArrayList<>(shuffled.subList(0, nTest));
        List<Event> train = new ArrayList<>(shuffled.subList(nTest, shuffled.size()));
        Map<String, String> meta = new LinkedHashMap<>();
        meta.put("warning", "TEMPORAL_LEAKAGE_RISK");
        meta.put("testRatio", String.valueOf(testRatio));
        meta.put("seed", String.valueOf(seed));
        return new Split(train, test, "random_unsafe", meta);
    }

    /**
     * Extract item catalog from train for candidate generation constraints
     * (test items not in train are cold items).
     */
    public static Set<String> itemCatalog(List<Event> train) {
        Set<String> items = new HashSet<>();
        for (Event e : train) {
            items.add(e.itemId);
        }
        return items;
    }

    public static Set<String> userCatalog(List<Event> events) {
        Set<String> users = new HashSet<>();
        for (Event e : events) {
            users.add(e.userId);
        }
        return users;
    }

    private static Map<String, List<Event>> groupByUser(List<Event> events) {
        Map<String, List<Event>> map = new HashMap<>();
        for (Event e : events) {
            map.computeIfAbsent(e.userId, k -> new ArrayList<>()).add(e);
        }
        return map;
    }
}
