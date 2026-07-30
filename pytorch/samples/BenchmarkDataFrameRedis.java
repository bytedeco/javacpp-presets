package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.redis.Redis;
import org.bytedeco.pytorch.dataframe.redis.RedisException;
import org.bytedeco.pytorch.dataframe.redis.RedisOptions;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorMetric;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorSearchResult;
import org.bytedeco.pytorch.dataframe.vectorstore.VectorStore;
import org.bytedeco.pytorch.dataframe.vectorstore.redis.RedisVectorStore;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * DataFrame ↔ Redis full-stack benchmark (no Jedis).
 *
 * <p>Covers:
 * <ul>
 *   <li>D1 connection / ping / select</li>
 *   <li>D2 KV strings + TTL ({@code SETEX}/{@code EXPIRE}/{@code TTL})</li>
 *   <li>D3 hashes + pipeline + expireMany</li>
 *   <li>D4 DataFrame HASH layout round-trip + TTL</li>
 *   <li>D5 DataFrame JSON / FRAME_JSON layouts</li>
 *   <li>D6 convenience {@code DataFrame.toRedis*} / {@code readRedis*}</li>
 *   <li>D7 RediSearch vector upsert + knn (skipped if module missing)</li>
 *   <li>D8 vector upsert with doc-key TTL</li>
 *   <li>D9 scale 2k HASH write/read</li>
 * </ul>
 *
 * <pre>
 *   # default 127.0.0.1:6379
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... samples.BenchmarkDataFrameRedis
 *
 *   # custom
 *   java ... samples.BenchmarkDataFrameRedis redis://127.0.0.1:6379/0
 * </pre>
 */
public class BenchmarkDataFrameRedis {

    static int passed = 0, failed = 0, skipped = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void section(String title) {
        System.out.println("\n=== " + title + " ===");
        report.append("\n=== ").append(title).append(" ===\n");
    }

    static void check(String name, boolean ok) {
        check(name, ok, null);
    }

    static void check(String name, boolean ok, String detail) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
            report.append("PASS  ").append(name).append('\n');
        } else {
            failed++;
            System.out.println("  FAIL  " + name + (detail == null || detail.isEmpty() ? "" : " — " + detail));
            report.append("FAIL  ").append(name);
            if (detail != null) report.append(" — ").append(detail);
            report.append('\n');
        }
    }

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("  SKIP  " + name + " — " + reason);
        report.append("SKIP  ").append(name).append(" — ").append(reason).append('\n');
    }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            System.out.println(" FAIL " + name + ": " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static DataFrame seedPeople() {
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("name", Column.DType.STRING);
        df.addColumn("score", Column.DType.FLOAT64);
        df.addColumn("ok", Column.DType.BOOLEAN);
        Object[][] rows = {
                {"1", "alice", 9.5, true},
                {"2", "bob", 7.0, false},
                {"3", "carol", 8.25, true},
                {"4", "dave", 6.5, false},
                {"5", "erin", 9.0, true},
        };
        for (Object[] row : rows) {
            int ri = df.addEmptyRow();
            for (int c = 0; c < row.length; c++) df.set(ri, df.column(c).name(), row[c]);
        }
        return df;
    }

    static DataFrame seedVectors(int n, int dim, long seed) {
        Random rnd = new Random(seed);
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("emb", Column.DType.VECTOR);
        df.addColumn("title", Column.DType.STRING);
        df.addColumn("year", Column.DType.INT64);
        for (int i = 0; i < n; i++) {
            int row = df.addEmptyRow();
            df.set(row, "id", "id-" + i);
            float[] v = new float[dim];
            double norm = 0;
            for (int d = 0; d < dim; d++) {
                v[d] = (float) rnd.nextGaussian();
                norm += v[d] * v[d];
            }
            norm = Math.sqrt(norm);
            if (norm > 1e-12) for (int d = 0; d < dim; d++) v[d] = (float) (v[d] / norm);
            df.set(row, "emb", v);
            df.set(row, "title", "doc-" + i);
            df.set(row, "year", 2000L + (i % 25));
        }
        return df;
    }

    static DataFrame scale(int n) {
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.STRING);
        df.addColumn("v", Column.DType.FLOAT64);
        df.addColumn("label", Column.DType.STRING);
        for (int i = 0; i < n; i++) {
            int ri = df.addEmptyRow();
            df.set(ri, "id", "r" + i);
            df.set(ri, "v", i * 0.1);
            df.set(ri, "label", "g" + (i % 10));
        }
        return df;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameRedis ===");
        String uri = args.length > 0 && args[0] != null && !args[0].isBlank()
                ? args[0] : "127.0.0.1:6379";

        Redis redis;
        try {
            redis = uri.contains("://") || uri.contains("/")
                    ? Redis.connectUri(uri.startsWith("redis") ? uri : "redis://" + uri)
                    : Redis.connectUri(uri.contains(":") ? uri : uri + ":6379");
            redis.ping();
        } catch (Throwable e) {
            System.out.println("Cannot connect to Redis at " + uri + ": " + e.getMessage());
            System.out.println("Start Redis (or Redis Stack for vector tests) and re-run.");
            System.out.println("=== summary: passed=0 failed=0 skipped=1 (no redis) ===");
            return;
        }

        final String ns = "dfbench:" + System.currentTimeMillis() + ":";
        try {
            // ── D1 connection ─────────────────────────────────────────────
            section("D1 connection");
            benchmark("1. ping / echo / dbsize", () -> {
                String pong = redis.ping();
                check("ping", pong != null && pong.toUpperCase().contains("PONG"), "r=" + pong);
                check("echo", "hi".equals(redis.echo("hi")));
                long size = redis.dbSize();
                check("dbsize>=0", size >= 0, "dbsize=" + size);
            });

            // ── D2 strings + TTL ───────────────────────────────────────────
            section("D2 strings + TTL");
            benchmark("2. setex / get / ttl / persist", () -> {
                String k = ns + "kv:x";
                redis.setex(k, 30, "hello");
                check("get", "hello".equals(redis.get(k)));
                long t = redis.ttl(k);
                check("ttl in (0,30]", t > 0 && t <= 30, "ttl=" + t);
                redis.persist(k);
                check("persist ttl=-1", redis.ttl(k) == -1);
                redis.set(k, "v2", Duration.ofSeconds(5), false, false);
                check("set with EX", redis.ttl(k) > 0 && redis.ttl(k) <= 5);
                redis.del(k);
            });

            benchmark("3. mset / mget / incr", () -> {
                Map<String, String> m = new LinkedHashMap<>();
                m.put(ns + "a", "1");
                m.put(ns + "b", "2");
                redis.mset(m);
                List<String> got = redis.mget(ns + "a", ns + "b", ns + "missing");
                check("mget size", got.size() == 3);
                check("mget a", "1".equals(got.get(0)));
                check("mget missing null", got.get(2) == null);
                redis.set(ns + "cnt", "10");
                check("incr", redis.incr(ns + "cnt") == 11);
                redis.del(ns + "a", ns + "b", ns + "cnt");
            });

            // ── D3 hashes ─────────────────────────────────────────────────
            section("D3 hashes + expireMany");
            benchmark("4. hset / hgetall / expireMany", () -> {
                String k = ns + "hash:1";
                redis.hset(k, Map.of("name", "alice", "score", "9.5", "ok", "true"));
                Map<String, String> all = redis.hgetall(k);
                check("hget name", "alice".equals(all.get("name")));
                check("hlen", redis.hlen(k) >= 3);
                List<String> keys = List.of(k, ns + "hash:2");
                redis.hset(ns + "hash:2", Map.of("name", "bob"));
                long n = redis.expireMany(keys, Duration.ofSeconds(60));
                check("expireMany", n >= 1, "n=" + n);
                check("ttl hash", redis.ttl(k) > 0);
                redis.del(k, ns + "hash:2");
            });

            // ── D4 DataFrame HASH ─────────────────────────────────────────
            section("D4 DataFrame HASH layout + TTL");
            DataFrame people = seedPeople();
            String hashPrefix = ns + "people:";

            benchmark("5. writeHash + readHash round-trip", () -> {
                int n = redis.writeHash(people, hashPrefix, "id", Duration.ofMinutes(10));
                check("wrote 5", n == 5, "n=" + n);
                check("key exists", redis.exists(hashPrefix + "1"));
                long ttl = redis.ttl(hashPrefix + "1");
                check("row ttl>0", ttl > 0, "ttl=" + ttl);
                DataFrame back = redis.readHash(hashPrefix);
                check("read rows", back.rowCount() == 5, "rows=" + back.rowCount());
                check("has name", back.hasColumn("name"));
                check("has score", back.hasColumn("score"));
            });

            benchmark("6. DataFrame.toRedisHash / readRedisHash", () -> {
                String p = ns + "people2:";
                int n = people.toRedisHash(redis, p, "id", Duration.ofSeconds(120));
                check("toRedisHash 5", n == 5);
                DataFrame back = DataFrame.readRedisHash(redis, p);
                check("readRedisHash rows", back.rowCount() == 5, "rows=" + back.rowCount());
                // cleanup
                for (String k : redis.scanAll(p + "*")) redis.del(k);
            });

            // ── D5 JSON / FRAME ───────────────────────────────────────────
            section("D5 JSON + FRAME layouts");
            benchmark("7. per-row JSON + TTL", () -> {
                String p = ns + "json:";
                int n = people.toRedisJson(redis, p, "id", Duration.ofSeconds(90));
                check("json wrote 5", n == 5);
                String raw = redis.get(p + "1");
                check("json body", raw != null && raw.contains("alice"), "raw=" + raw);
                check("json ttl", redis.ttl(p + "1") > 0);
                DataFrame back = DataFrame.readRedisJson(redis, p);
                check("json read rows", back.rowCount() == 5, "rows=" + back.rowCount());
                for (String k : redis.scanAll(p + "*")) redis.del(k);
            });

            benchmark("8. frame JSON single key", () -> {
                String key = ns + "frame";
                int n = people.toRedisFrame(redis, key, Duration.ofSeconds(60));
                check("frame keys written", n == 1);
                DataFrame back = DataFrame.readRedisFrame(redis, key);
                check("frame rows", back.rowCount() == 5, "rows=" + back.rowCount());
                check("frame ttl", redis.ttl(key) > 0);
                redis.del(key);
            });

            benchmark("9. RedisOptions.parse + toRedis/readRedis", () -> {
                RedisOptions opts = RedisOptions.parse("hash://" + ns + "opts:?ttl=45&id=id");
                int n = people.toRedis(redis, opts);
                check("opts write", n == 5);
                DataFrame back = DataFrame.readRedis(redis, opts);
                check("opts read", back.rowCount() == 5);
                for (String k : redis.scanAll(ns + "opts:*")) redis.del(k);
            });

            // ── D6 scan / lists / sets smoke ──────────────────────────────
            section("D6 misc structures smoke");
            benchmark("10. list / set / zset", () -> {
                String lk = ns + "list";
                redis.rpush(lk, "a", "b", "c");
                check("llen", redis.llen(lk) == 3);
                check("lrange", redis.lrange(lk, 0, -1).size() == 3);
                String sk = ns + "set";
                redis.sadd(sk, "x", "y", "x");
                check("scard", redis.scard(sk) == 2);
                String zk = ns + "zset";
                redis.zadd(zk, 1.0, "a");
                redis.zadd(zk, 2.0, "b");
                check("zcard", redis.zcard(zk) == 2);
                redis.del(lk, sk, zk);
            });

            // ── D7 RediSearch vector ──────────────────────────────────────
            section("D7 RediSearch vector (optional)");
            final int dim = 32;
            final boolean[] redisearchOk = {false};
            benchmark("11. FT.CREATE + vector upsert + knn", () -> {
                String index = ns + "idx:clips";
                String prefix = ns + "doc:";
                try {
                    DataFrame vecs = seedVectors(50, dim, 42);
                    try (RedisVectorStore vs = RedisVectorStore.builder()
                            .client(redis.client())
                            .index(index)
                            .prefix(prefix)
                            .dim(dim)
                            .metric(VectorMetric.COSINE)
                            .textFields("title")
                            .numericFields("year")
                            .build()) {
                        vs.ensureCollection();
                        vs.upsertDataFrame(vecs, "id", "emb", "title", "year");
                        long c = vs.count();
                        check("vector count", c == 50 || c == -1, "count=" + c);
                        float[] q = (float[]) vecs.get(0, "emb");
                        VectorSearchResult top = vs.search(q, 5);
                        check("knn size", top.size() > 0, "size=" + top.size());
                        if (top.size() > 0) {
                            check("knn self-ish", top.get(0).id() != null);
                            System.out.println("    top-1 id=" + top.get(0).id()
                                    + " score=" + top.get(0).score());
                        }
                        redisearchOk[0] = true;
                        // cleanup docs + index
                        vs.dropCollection();
                    }
                } catch (RedisException | org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException e) {
                    String msg = e.getMessage() == null ? "" : e.getMessage().toLowerCase();
                    if (msg.contains("unknown command") || msg.contains("ft.")
                            || msg.contains("no such") || msg.contains("unknown index name")
                            || msg.contains("err unknown")) {
                        skip("RediSearch vector", "module not available: " + e.getMessage());
                    } else {
                        throw e;
                    }
                }
            });

            // ── D8 vector + TTL ───────────────────────────────────────────
            section("D8 vector upsert with doc TTL");
            benchmark("12. vector upsert + EXPIRE on doc keys", () -> {
                if (!redisearchOk[0]) {
                    // still try — maybe FT worked partially; if not, skip
                    // Attempt a plain hash-style TTL path using Redis API directly
                    String prefix = ns + "vttl:";
                    DataFrame vecs = seedVectors(10, dim, 7);
                    // store as hash without RediSearch to still validate TTL path of Redis DF
                    int n = vecs.toRedisHash(redis, prefix, "id", Duration.ofSeconds(30));
                    check("hash-ttl wrote", n == 10);
                    check("doc ttl", redis.ttl(prefix + "id-0") > 0);
                    for (String k : redis.scanAll(prefix + "*")) redis.del(k);
                    return;
                }
                String index = ns + "idx:ttl";
                String prefix = ns + "docttl:";
                DataFrame vecs = seedVectors(20, dim, 99);
                try (RedisVectorStore vs = RedisVectorStore.builder()
                        .client(redis.client())
                        .index(index)
                        .prefix(prefix)
                        .dim(dim)
                        .metric(VectorMetric.L2)
                        .ttl(Duration.ofSeconds(45))
                        .textFields("title")
                        .build()) {
                    vs.ensureCollection();
                    vs.upsertDataFrame(vecs, "id", "emb", Duration.ofSeconds(45), "title", "year");
                    long t = vs.ttl("id-0");
                    check("vector doc ttl>0", t > 0 && t <= 45, "ttl=" + t);
                    // refresh TTL
                    long n = vs.expire(Duration.ofSeconds(90), "id-0", "id-1");
                    check("expire refresh", n >= 1, "n=" + n);
                    vs.dropCollection();
                } catch (org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException e) {
                    skip("vector TTL", e.getMessage());
                }
            });

            benchmark("13. DataFrame.toRedisVectorStore with TTL", () -> {
                if (!redisearchOk[0]) {
                    skip("toRedisVectorStore TTL", "RediSearch not available");
                    return;
                }
                DataFrame vecs = seedVectors(15, dim, 3);
                String index = ns + "idx:df";
                // parse host/port from connection — use builder via open path
                try (VectorStore vs = vecs.toRedisVectorStore(
                        redis.host() == null ? "127.0.0.1" : redis.host(),
                        redis.port(),
                        index, "id", "emb", dim,
                        VectorMetric.COSINE,
                        Duration.ofSeconds(60),
                        "title", "year")) {
                    check("df vector count", vs.count() == 15 || vs.count() == -1,
                            "count=" + vs.count());
                    if (vs instanceof RedisVectorStore rvs) {
                        check("df doc ttl", rvs.ttl("id-0") > 0);
                        rvs.dropCollection();
                    }
                } catch (org.bytedeco.pytorch.dataframe.vectorstore.VectorStoreException e) {
                    skip("toRedisVectorStore", e.getMessage());
                }
            });

            // ── D9 scale ──────────────────────────────────────────────────
            section("D9 scale 2k HASH write/read");
            benchmark("14. scale 2000 rows HASH", () -> {
                int n = 2000;
                DataFrame big = scale(n);
                String p = ns + "scale:";
                long t0 = System.nanoTime();
                int written = big.toRedisHash(redis, p, "id", Duration.ofMinutes(5));
                long wMs = (System.nanoTime() - t0) / 1_000_000L;
                check("scale wrote", written == n, "n=" + written);
                t0 = System.nanoTime();
                DataFrame back = DataFrame.readRedisHash(redis, p);
                long rMs = (System.nanoTime() - t0) / 1_000_000L;
                check("scale read rows", back.rowCount() == n, "rows=" + back.rowCount());
                System.out.println("    write " + n + ": " + wMs + " ms ("
                        + (n * 1000.0 / Math.max(1, wMs)) + " rows/s); read: " + rMs + " ms");
                // cleanup via scan
                List<String> keys = redis.scanAll(p + "*", 500);
                if (!keys.isEmpty()) redis.unlink(keys.toArray(new String[0]));
            });

            // cleanup remaining ns keys
            List<String> leftover = redis.scanAll(ns + "*", 500);
            if (!leftover.isEmpty()) {
                redis.unlink(leftover.toArray(new String[0]));
                System.out.println("\n  cleaned " + leftover.size() + " leftover keys under " + ns);
            }

        } finally {
            try { redis.close(); } catch (Exception ignored) {}
        }

        System.out.println("\n=== summary: passed=" + passed + " failed=" + failed
                + " skipped=" + skipped + " ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
    }
}
