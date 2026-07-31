/*
 * Redis online feature store — Feast RedisOnlineStore / Alibaba Tair / ByteDance
 * feature KV pattern.
 *
 * Key layout:  {prefix}{project}#{view}#{entityKey}  → Redis HASH
 *   fields: feature columns (FeatureValueCodec) + _event_ts, _written_at, _ttl
 *
 * Uses existing pure-RESP {@link org.bytedeco.pytorch.dataframe.redis.Redis}
 * (no Jedis/Lettuce). Connection failures surface as IllegalStateException;
 * use {@link #available()} to probe before production traffic.
 */
package org.bytedeco.pytorch.feature.online;

import org.bytedeco.pytorch.dataframe.redis.Redis;
import org.bytedeco.pytorch.feature.store.FeatureValueCodec;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Redis HASH-backed {@link OnlineStore}. */
public final class RedisOnlineStore implements OnlineStore {

    private final Redis redis;
    private final boolean ownsRedis;
    private final String keyPrefix;
    private final Duration defaultTtl;
    private final boolean evictExpiredOnRead;

    public RedisOnlineStore(Redis redis, String keyPrefix, Duration defaultTtl, boolean ownsRedis) {
        this.redis = Objects.requireNonNull(redis, "redis");
        this.keyPrefix = keyPrefix != null ? keyPrefix : "fs:";
        this.defaultTtl = defaultTtl;
        this.ownsRedis = ownsRedis;
        this.evictExpiredOnRead = true;
    }

    public static RedisOnlineStore connect(String uri) {
        return connect(uri, "fs:", null);
    }

    public static RedisOnlineStore connect(String uri, String keyPrefix, Duration ttl) {
        Redis r = Redis.connectUri(uri != null ? uri : "redis://127.0.0.1:6379/0");
        return new RedisOnlineStore(r, keyPrefix, ttl, true);
    }

    public static RedisOnlineStore wrap(Redis redis, String keyPrefix, Duration ttl) {
        return new RedisOnlineStore(redis, keyPrefix, ttl, false);
    }

    public Redis redis() {
        return redis;
    }

    public String keyPrefix() {
        return keyPrefix;
    }

    /** Lightweight liveness probe. */
    public boolean available() {
        try {
            String pong = redis.ping();
            return pong != null && pong.toUpperCase().contains("PONG");
        } catch (Exception e) {
            return false;
        }
    }

    private String key(String project, String viewName, String entityKey) {
        String p = project == null || project.isEmpty() ? "default" : project;
        return keyPrefix + p + "#" + viewName + "#" + entityKey;
    }

    private String viewPattern(String project, String viewName) {
        String p = project == null || project.isEmpty() ? "default" : project;
        return keyPrefix + p + "#" + viewName + "#*";
    }

    @Override
    public void onlineWrite(OnlineWriteBatch batch) {
        if (batch == null || batch.size() == 0) return;
        try {
            // pipeline-friendly sequential writes (Redis client has pipeline APIs)
            List<Object[]> commands = new ArrayList<>();
            for (OnlineFeatureRow row : batch.rows()) {
                String k = key(row.project(), row.viewName(), row.entityKey());
                Map<String, String> fields = new LinkedHashMap<>();
                fields.putAll(FeatureValueCodec.encodeMap(row.values()));
                fields.put("_event_ts", FeatureValueCodec.encode(row.eventTimestampMs()));
                fields.put("_written_at", FeatureValueCodec.encode(row.writtenAtMs()));
                long ttlMs = row.ttlMs() > 0 ? row.ttlMs()
                        : (defaultTtl != null ? defaultTtl.toMillis() : 0L);
                fields.put("_ttl", FeatureValueCodec.encode(ttlMs));

                // HSET key f1 v1 f2 v2 ...
                List<Object> args = new ArrayList<>();
                args.add("HSET");
                args.add(k);
                for (Map.Entry<String, String> e : fields.entrySet()) {
                    args.add(e.getKey());
                    args.add(e.getValue());
                }
                commands.add(args.toArray());

                if (ttlMs > 0) {
                    // expire roughly at event_ts + ttl, or from now if event_ts old
                    long now = System.currentTimeMillis();
                    long remain = row.eventTimestampMs() > 0
                            ? (row.eventTimestampMs() + ttlMs) - now
                            : ttlMs;
                    if (remain > 0) {
                        commands.add(new Object[]{"PEXPIRE", k, String.valueOf(remain)});
                    }
                } else if (defaultTtl != null && !defaultTtl.isZero() && !defaultTtl.isNegative()) {
                    commands.add(new Object[]{"PEXPIRE", k, String.valueOf(defaultTtl.toMillis())});
                }
            }
            redis.pipelineVoid(commands);
        } catch (RuntimeException e) {
            throw new IllegalStateException("RedisOnlineStore.write failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Optional<OnlineFeatureRow> onlineRead(String project, String viewName, String entityKey) {
        try {
            String k = key(project, viewName, entityKey);
            Map<String, String> all = redis.hgetall(k);
            if (all == null || all.isEmpty()) return Optional.empty();
            OnlineFeatureRow row = fromHash(project, viewName, entityKey, all);
            if (evictExpiredOnRead && row.isExpired(System.currentTimeMillis())) {
                redis.del(k);
                return Optional.empty();
            }
            return Optional.of(row);
        } catch (RuntimeException e) {
            throw new IllegalStateException("RedisOnlineStore.read failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Map<String, OnlineFeatureRow> onlineReadBatch(String project, String viewName,
                                                         Collection<String> entityKeys) {
        Map<String, OnlineFeatureRow> out = new LinkedHashMap<>();
        if (entityKeys == null || entityKeys.isEmpty()) return out;
        // Pipeline HGETALL for each key
        try {
            List<String> keys = new ArrayList<>();
            List<Object[]> commands = new ArrayList<>();
            for (String ek : entityKeys) {
                String k = key(project, viewName, ek);
                keys.add(ek);
                commands.add(new Object[]{"HGETALL", k});
            }
            List<Object> replies = redis.pipeline(commands);
            long now = System.currentTimeMillis();
            for (int i = 0; i < keys.size(); i++) {
                Object rep = i < replies.size() ? replies.get(i) : null;
                Map<String, String> hash = toStringMap(rep);
                if (hash == null || hash.isEmpty()) continue;
                OnlineFeatureRow row = fromHash(project, viewName, keys.get(i), hash);
                if (evictExpiredOnRead && row.isExpired(now)) {
                    redis.del(key(project, viewName, keys.get(i)));
                    continue;
                }
                out.put(keys.get(i), row);
            }
            return out;
        } catch (RuntimeException e) {
            // fallback sequential
            for (String ek : entityKeys) {
                onlineRead(project, viewName, ek).ifPresent(r -> out.put(ek, r));
            }
            return out;
        }
    }

    @Override
    public long size(String project, String viewName) {
        try {
            List<String> keys = redis.scanAll(viewPattern(project, viewName));
            return keys == null ? 0L : keys.size();
        } catch (RuntimeException e) {
            return -1L;
        }
    }

    @Override
    public void delete(String project, String viewName, String entityKey) {
        try {
            redis.del(key(project, viewName, entityKey));
        } catch (RuntimeException e) {
            throw new IllegalStateException("RedisOnlineStore.delete failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void clearView(String project, String viewName) {
        try {
            List<String> keys = redis.scanAll(viewPattern(project, viewName));
            if (keys == null || keys.isEmpty()) return;
            // delete in chunks
            int batch = 256;
            for (int i = 0; i < keys.size(); i += batch) {
                List<String> chunk = keys.subList(i, Math.min(i + batch, keys.size()));
                redis.del(chunk.toArray(new String[0]));
            }
        } catch (RuntimeException e) {
            throw new IllegalStateException("RedisOnlineStore.clearView failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() {
        if (ownsRedis) {
            try {
                redis.close();
            } catch (Exception ignored) {
            }
        }
    }

    private static OnlineFeatureRow fromHash(String project, String viewName, String entityKey,
                                             Map<String, String> all) {
        Map<String, Object> values = new LinkedHashMap<>();
        long eventTs = 0L;
        long writtenAt = 0L;
        long ttl = 0L;
        for (Map.Entry<String, String> e : all.entrySet()) {
            String f = e.getKey();
            if ("_event_ts".equals(f)) {
                Object v = FeatureValueCodec.decode(e.getValue());
                eventTs = v instanceof Number ? ((Number) v).longValue() : 0L;
            } else if ("_written_at".equals(f)) {
                Object v = FeatureValueCodec.decode(e.getValue());
                writtenAt = v instanceof Number ? ((Number) v).longValue() : 0L;
            } else if ("_ttl".equals(f)) {
                Object v = FeatureValueCodec.decode(e.getValue());
                ttl = v instanceof Number ? ((Number) v).longValue() : 0L;
            } else if (!f.startsWith("_")) {
                values.put(f, FeatureValueCodec.decode(e.getValue()));
            }
        }
        return OnlineFeatureRow.builder(viewName, entityKey)
                .project(project == null || project.isEmpty() ? "default" : project)
                .values(values)
                .eventTimestampMs(eventTs)
                .writtenAtMs(writtenAt > 0 ? writtenAt : System.currentTimeMillis())
                .ttlMs(ttl)
                .build();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> toStringMap(Object rep) {
        if (rep == null) return Map.of();
        if (rep instanceof Map) {
            Map<String, String> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) rep).entrySet()) {
                out.put(String.valueOf(e.getKey()), e.getValue() == null ? null : String.valueOf(e.getValue()));
            }
            return out;
        }
        if (rep instanceof List) {
            // Redis HGETALL RESP array: [k1,v1,k2,v2,...]
            List<?> list = (List<?>) rep;
            Map<String, String> out = new LinkedHashMap<>();
            for (int i = 0; i + 1 < list.size(); i += 2) {
                out.put(String.valueOf(list.get(i)),
                        list.get(i + 1) == null ? null : String.valueOf(list.get(i + 1)));
            }
            return out;
        }
        return Map.of();
    }
}
