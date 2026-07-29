/*
 * Point-in-time join correctness + performance.
 * Critical: ZERO future leakage (feature_ts > event_ts must never join).
 */
package org.bytedeco.pytorch.utils.feature.benchmarks;

import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureTable;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.ValueType;
import org.bytedeco.pytorch.utils.feature.offline.PointInTimeJoin;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** PIT join adversarial correctness + scale bench. */
public final class PitJoinBenchmark {

    private PitJoinBenchmark() {}

    public static void run(BenchCase.Suite suite, int nEntities) {
        correctness(suite);
        perf(suite, nEntities);
        ttl(suite);
    }

    private static void correctness(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            Entity user = Entity.of("user_id");
            FeatureView view = FeatureView.builder("user_stats")
                    .entities(user)
                    .ttl(Duration.ZERO) // infinite
                    .schema(Field.of("click_7d", ValueType.INT64),
                            Field.of("score", ValueType.FLOAT64))
                    .source(FeatureTable.memory("user_stats"))
                    .build();

            // entity event at t=1000
            List<Map<String, Object>> entities = new ArrayList<>();
            Map<String, Object> e = new LinkedHashMap<>();
            e.put("user_id", 1L);
            e.put("event_timestamp", 1000L);
            e.put("label", 1.0);
            entities.add(e);

            // features: past @500 value=10, future @1500 value=999 (MUST NOT LEAK)
            List<Map<String, Object>> features = new ArrayList<>();
            Map<String, Object> past = new LinkedHashMap<>();
            past.put("user_id", 1L);
            past.put("event_timestamp", 500L);
            past.put("click_7d", 10L);
            past.put("score", 0.5);
            features.add(past);
            Map<String, Object> future = new LinkedHashMap<>();
            future.put("user_id", 1L);
            future.put("event_timestamp", 1500L);
            future.put("click_7d", 999L);
            future.put("score", 9.99);
            features.add(future);
            // another past later @800 should win
            Map<String, Object> past2 = new LinkedHashMap<>();
            past2.put("user_id", 1L);
            past2.put("event_timestamp", 800L);
            past2.put("click_7d", 20L);
            past2.put("score", 0.8);
            features.add(past2);

            PointInTimeJoin.Options opt = new PointInTimeJoin.Options().prefixWithViewName(true);
            PointInTimeJoin.Result result = PointInTimeJoin.joinOne(entities, features, view, opt);

            Map<String, Object> row = result.rows.get(0);
            Object click = row.get("user_stats__click_7d");
            Object score = row.get("user_stats__score");
            boolean ok = Long.valueOf(20L).equals(toLong(click))
                    && Math.abs(toDouble(score) - 0.8) < 1e-9
                    && result.stats.futureRowsRejected >= 1;

            long leaks = PointInTimeJoin.countFutureLeaks(result.rows, features, view, opt);
            if (!ok || leaks != 0) {
                suite.add(BenchCase.fail("pit_correctness",
                        "click=" + click + " score=" + score + " leaks=" + leaks
                                + " stats=" + result.stats,
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("pit_correctness",
                        "latest-as-of ok, futureRejected=" + result.stats.futureRowsRejected,
                        System.nanoTime() - t0));
            }
        } catch (Exception ex) {
            suite.add(BenchCase.fail("pit_correctness", ex.toString(), System.nanoTime() - t0));
        }
    }

    private static void ttl(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            Entity user = Entity.of("user_id");
            FeatureView view = FeatureView.builder("ttl_view")
                    .entities(user)
                    .ttl(Duration.ofMillis(100)) // very short TTL
                    .schema(Field.of("x", ValueType.INT64))
                    .source(FeatureTable.memory("ttl_view"))
                    .build();

            List<Map<String, Object>> entities = List.of(Map.of(
                    "user_id", 1L,
                    "event_timestamp", 1000L));
            // feature at 500 is 500ms old > 100ms TTL → miss
            List<Map<String, Object>> features = List.of(Map.of(
                    "user_id", 1L,
                    "event_timestamp", 500L,
                    "x", 42L));

            PointInTimeJoin.Result result = PointInTimeJoin.joinOne(
                    entities, features, view, new PointInTimeJoin.Options());
            Object x = result.rows.get(0).get("ttl_view__x");
            boolean miss = x == null && result.stats.ttlExpiredRejected >= 1 && result.stats.joinsMiss == 1;
            if (!miss) {
                suite.add(BenchCase.fail("pit_ttl",
                        "x=" + x + " stats=" + result.stats, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("pit_ttl",
                        "ttl expired correctly, rejected=" + result.stats.ttlExpiredRejected,
                        System.nanoTime() - t0));
            }
        } catch (Exception ex) {
            suite.add(BenchCase.fail("pit_ttl", ex.toString(), System.nanoTime() - t0));
        }
    }

    private static void perf(BenchCase.Suite suite, int nEntities) {
        long t0 = System.nanoTime();
        try {
            Entity user = Entity.of("user_id");
            List<FeatureView> views = new ArrayList<>();
            for (int v = 0; v < 5; v++) {
                views.add(FeatureView.builder("v" + v)
                        .entities(user)
                        .schema(Field.of("f", ValueType.FLOAT64))
                        .source(FeatureTable.memory("v" + v))
                        .build());
            }

            List<Map<String, Object>> entities = new ArrayList<>(nEntities);
            for (int i = 0; i < nEntities; i++) {
                Map<String, Object> e = new LinkedHashMap<>();
                e.put("user_id", (long) (i % 1000 + 1));
                e.put("event_timestamp", 1_000_000L + i);
                entities.add(e);
            }

            Map<String, List<Map<String, Object>>> byView = new LinkedHashMap<>();
            for (FeatureView view : views) {
                List<Map<String, Object>> frows = new ArrayList<>(2000);
                for (int i = 0; i < 2000; i++) {
                    Map<String, Object> r = new LinkedHashMap<>();
                    r.put("user_id", (long) (i % 1000 + 1));
                    r.put("event_timestamp", 900_000L + i * 10L);
                    r.put("f", i * 0.01);
                    frows.add(r);
                }
                byView.put(view.name(), frows);
            }

            PointInTimeJoin.Result result = PointInTimeJoin.joinMany(
                    entities, byView, views, new PointInTimeJoin.Options());
            long dt = System.nanoTime() - t0;
            double pEntityUs = (dt / 1000.0) / Math.max(1, nEntities);
            Map<String, Object> metrics = new LinkedHashMap<>();
            metrics.put("entities", nEntities);
            metrics.put("views", views.size());
            metrics.put("us_per_entity", pEntityUs);
            metrics.put("future_rejected", result.stats.futureRowsRejected);
            suite.add(BenchCase.pass("pit_perf",
                    result.stats.toString(), dt, metrics));
        } catch (Exception ex) {
            suite.add(BenchCase.fail("pit_perf", ex.toString(), System.nanoTime() - t0));
        }
    }

    private static long toLong(Object v) {
        if (v instanceof Number) return ((Number) v).longValue();
        return Long.MIN_VALUE;
    }

    private static double toDouble(Object v) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        return Double.NaN;
    }
}
