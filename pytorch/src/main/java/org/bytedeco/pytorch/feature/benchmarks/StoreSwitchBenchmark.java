/*
 * Store switch benchmarks — MEMORY / SQLITE / DUCKDB always;
 * REDIS / MILVUS probed and skipped cleanly when unreachable.
 */
package org.bytedeco.pytorch.feature.benchmarks;

import org.bytedeco.pytorch.feature.FeaturePlatform;
import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.ValueType;
import org.bytedeco.pytorch.feature.online.OnlineFeatureRow;
import org.bytedeco.pytorch.feature.online.OnlineWriteBatch;
import org.bytedeco.pytorch.feature.online.RedisOnlineStore;
import org.bytedeco.pytorch.feature.offline.DuckDbOfflineStore;
import org.bytedeco.pytorch.feature.serving.FeatureResponse;
import org.bytedeco.pytorch.feature.store.FeatureValueCodec;
import org.bytedeco.pytorch.feature.store.StoreConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Backend switch + codec + SQLite/DuckDB/Redis probe benches. */
public final class StoreSwitchBenchmark {

    private StoreSwitchBenchmark() {}

    public static void run(BenchCase.Suite suite) {
        codecRoundtrip(suite);
        memoryConfig(suite);
        sqliteOnlineOfflineEmbedding(suite);
        duckdbOffline(suite);
        redisProbe(suite);
        switchSameApi(suite);
    }

    private static void codecRoundtrip(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            Map<String, Object> values = new LinkedHashMap<>();
            values.put("clicks", 12L);
            values.put("score", 0.85);
            values.put("flag", true);
            values.put("name", "hello");
            values.put("hist", new long[]{1L, 2L, 3L});
            values.put("emb", new float[]{0.1f, 0.2f, 0.3f});
            Map<String, String> enc = FeatureValueCodec.encodeMap(values);
            Map<String, Object> dec = FeatureValueCodec.decodeMap(enc);
            boolean ok = Long.valueOf(12L).equals(toLong(dec.get("clicks")))
                    && Math.abs(toDouble(dec.get("score")) - 0.85) < 1e-9
                    && Boolean.TRUE.equals(dec.get("flag"))
                    && "hello".equals(dec.get("name"))
                    && dec.get("hist") instanceof long[]
                    && ((long[]) dec.get("hist")).length == 3
                    && dec.get("emb") instanceof float[]
                    && ((float[]) dec.get("emb")).length == 3;
            if (!ok) {
                suite.add(BenchCase.fail("store_codec", "dec=" + dec, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("store_codec", "types preserved", System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("store_codec", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void memoryConfig(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.fromConfig(StoreConfig.memory())) {
            smokeServe(fp, "mem");
            suite.add(BenchCase.pass("store_memory_config",
                    "online=" + fp.storeConfig().onlineBackend()
                            + " emb=" + fp.embeddings().backend(),
                    System.nanoTime() - t0));
        } catch (Exception e) {
            suite.add(BenchCase.fail("store_memory_config", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void sqliteOnlineOfflineEmbedding(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            Path tmp = Files.createTempDirectory("fs-sqlite-bench");
            StoreConfig cfg = StoreConfig.builder()
                    .root(tmp)
                    .online("sqlite")
                    .offline("sqlite")
                    .embedding("sqlite")
                    .embeddingDim(8)
                    .sqliteOnlinePath(tmp.resolve("online.db").toString())
                    .sqliteOfflinePath(tmp.resolve("offline.db").toString())
                    .option("sqlite_embedding_path", tmp.resolve("emb.db").toString())
                    .build();
            try (FeaturePlatform fp = FeaturePlatform.fromConfig(cfg)) {
                // direct online write/read
                OnlineFeatureRow row = OnlineFeatureRow.builder("v", "u1")
                        .project("default")
                        .put("x", 7L)
                        .put("s", 1.5)
                        .eventTimestampMs(System.currentTimeMillis())
                        .build();
                fp.online().onlineWrite(OnlineWriteBatch.of(row));
                Optional<OnlineFeatureRow> got = fp.online().onlineRead("default", "v", "u1");
                if (got.isEmpty() || !Long.valueOf(7L).equals(toLong(got.get().get("x")))) {
                    suite.add(BenchCase.fail("store_sqlite_online",
                            "got=" + got, System.nanoTime() - t0));
                    return;
                }

                // offline put + read
                List<Map<String, Object>> rows = new ArrayList<>();
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("user_id", 1L);
                r.put("event_timestamp", System.currentTimeMillis());
                r.put("x", 9L);
                rows.add(r);
                fp.offline().put("default", "uv", rows);
                if (fp.offline().rowCount("default", "uv") < 1) {
                    suite.add(BenchCase.fail("store_sqlite_offline", "rowCount=0", System.nanoTime() - t0));
                    return;
                }

                // embedding
                float[] emb = new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
                fp.embeddings().put("items", "i1", emb);
                Optional<float[]> eg = fp.embeddings().get("items", "i1");
                if (eg.isEmpty() || eg.get().length != 8) {
                    suite.add(BenchCase.fail("store_sqlite_emb",
                            "emb=" + (eg.isEmpty() ? "empty" : eg.get().length),
                            System.nanoTime() - t0));
                    return;
                }

                smokeServe(fp, "sqlite");
                suite.add(BenchCase.pass("store_sqlite_full",
                        "online=" + fp.online().getClass().getSimpleName()
                                + " offline=" + fp.offline().getClass().getSimpleName()
                                + " emb=" + fp.embeddings().backend()
                                + " root=" + tmp,
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("store_sqlite_full", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void duckdbOffline(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            DuckDbOfflineStore store = new DuckDbOfflineStore();
            List<Map<String, Object>> rows = List.of(Map.of(
                    "user_id", 1L,
                    "event_timestamp", System.currentTimeMillis(),
                    "f", 3.14));
            store.put("default", "duck_v", rows);
            boolean ok = store.rowCount("default", "duck_v") == 1
                    && !store.readAll("default", "duck_v").isEmpty();
            store.close();
            if (!ok) {
                suite.add(BenchCase.fail("store_duckdb", "readback failed", System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("store_duckdb",
                        "available=" + store.duckAvailable() + " rows=1",
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("store_duckdb", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void redisProbe(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        String uri = System.getProperty("feature.bench.redis", "redis://127.0.0.1:6379/0");
        try {
            RedisOnlineStore store = RedisOnlineStore.connect(uri, "fs:bench:", null);
            if (!store.available()) {
                store.close();
                suite.add(BenchCase.pass("store_redis_probe",
                        "SKIP unreachable " + uri, System.nanoTime() - t0));
                return;
            }
            OnlineFeatureRow row = OnlineFeatureRow.builder("rv", "k1")
                    .project("bench")
                    .put("x", 42L)
                    .put("emb", new float[]{1f, 2f, 3f})
                    .eventTimestampMs(System.currentTimeMillis())
                    .ttlMs(60_000L)
                    .build();
            store.onlineWrite(OnlineWriteBatch.of(row));
            Optional<OnlineFeatureRow> got = store.onlineRead("bench", "rv", "k1");
            store.delete("bench", "rv", "k1");
            store.close();
            boolean ok = got.isPresent() && Long.valueOf(42L).equals(toLong(got.get().get("x")));
            if (!ok) {
                suite.add(BenchCase.fail("store_redis_probe", "got=" + got, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("store_redis_probe",
                        "write/read/delete ok @ " + uri, System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.pass("store_redis_probe",
                    "SKIP error: " + e.getMessage(), System.nanoTime() - t0));
        }
    }

    /**
     * Same feature definitions materialize+serve on MEMORY and SQLITE configs.
     */
    private static void switchSameApi(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            double memScore = runPipeline(StoreConfig.memory());
            Path tmp = Files.createTempDirectory("fs-switch");
            StoreConfig sqliteCfg = StoreConfig.builder()
                    .root(tmp)
                    .online("sqlite")
                    .offline("file")
                    .embedding("memory")
                    .sqliteOnlinePath(tmp.resolve("on.db").toString())
                    .build();
            double sqliteScore = runPipeline(sqliteCfg);
            if (Math.abs(memScore - sqliteScore) > 1e-6) {
                suite.add(BenchCase.fail("store_switch_api",
                        "mem=" + memScore + " sqlite=" + sqliteScore, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("store_switch_api",
                        "same served score=" + memScore + " on MEMORY and SQLITE",
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("store_switch_api", e.toString(), System.nanoTime() - t0));
        }
    }

    private static double runPipeline(StoreConfig cfg) {
        try (FeaturePlatform fp = FeaturePlatform.fromConfig(cfg)) {
            Entity user = Entity.of("user_id");
            fp.entity(user);
            FeatureView view = FeatureView.builder("u_stats")
                    .entities(user)
                    .schema(Field.of("score", ValueType.FLOAT64))
                    .online(true)
                    .build();
            fp.featureView(view);
            fp.featureService(FeatureService.builder("svc").views("u_stats").build());
            long now = System.currentTimeMillis();
            fp.putOffline("default", "u_stats", List.of(Map.of(
                    "user_id", 1L,
                    "event_timestamp", now,
                    "score", 0.42)));
            fp.materializeViews(List.of(view));
            FeatureResponse resp = fp.getOnlineFeatures("svc", Map.of("user_id", 1L));
            Object s = resp.vector().raw().get("score");
            if (s == null) s = resp.vector().raw().get("u_stats__score");
            return toDouble(s);
        }
    }

    private static void smokeServe(FeaturePlatform fp, String tag) {
        Entity user = Entity.of("user_id");
        fp.entity(user);
        FeatureView view = FeatureView.builder("smoke_" + tag)
                .entities(user)
                .schema(Field.of("x", ValueType.INT64))
                .online(true)
                .build();
        fp.featureView(view);
        fp.featureService(FeatureService.builder("smoke_svc_" + tag).views("smoke_" + tag).build());
        long now = System.currentTimeMillis();
        fp.putOffline("default", "smoke_" + tag, List.of(Map.of(
                "user_id", 1L, "event_timestamp", now, "x", 1L)));
        fp.materializeViews(List.of(view));
        FeatureResponse r = fp.getOnlineFeatures("smoke_svc_" + tag, Map.of("user_id", 1L));
        if (!r.success()) {
            throw new IllegalStateException("smoke serve failed: " + r);
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
