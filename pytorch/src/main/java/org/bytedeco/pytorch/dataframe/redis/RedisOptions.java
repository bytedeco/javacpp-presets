package org.bytedeco.pytorch.dataframe.redis;

import org.bytedeco.pytorch.dataframe.Column;

import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Options for DataFrame ↔ Redis I/O (hash / JSON / string layouts).
 *
 * <pre>{@code
 * RedisOptions opts = RedisOptions.builder()
 *     .prefix("df:people:")
 *     .idColumn("id")
 *     .ttl(Duration.ofHours(1))
 *     .layout(RedisOptions.Layout.HASH)
 *     .build();
 * df.toRedis(redis, opts);
 * }</pre>
 */
public final class RedisOptions {

    /** How each DataFrame row is stored under Redis. */
    public enum Layout {
        /**
         * One Redis Hash per row: {@code HSET prefix{id} field value}.
         * Best for field-level access and RediSearch ON HASH.
         */
        HASH,
        /**
         * One JSON string key per row: {@code SET prefix{id} json}.
         * Uses the built-in minimal JSON encoder (no RedisJSON module required).
         */
        JSON,
        /**
         * Single key holding the entire frame as a JSON array of records
         * (or JSON object if only one logical blob). Key = {@code prefix} or
         * {@code key} override.
         */
        FRAME_JSON,
        /**
         * Single key holding JSONL (one record per line).
         */
        FRAME_JSONL
    }

    public enum IfExists {
        /** Overwrite existing keys (default). */
        REPLACE,
        /** Skip keys that already exist ({@code NX} semantics where applicable). */
        SKIP,
        /** Fail if any target key already exists. */
        FAIL
    }

    private final String prefix;
    private final String key;
    private final String idColumn;
    private final Duration ttl;
    private final Layout layout;
    private final IfExists ifExists;
    private final int pipelineBatch;
    private final boolean scanMatchPrefix;
    private final int scanCount;
    private final Map<String, Column.DType> dtype;
    private final boolean includeNulls;
    private final boolean binaryVectorsAsBase64;

    private RedisOptions(Builder b) {
        this.prefix = b.prefix == null ? "df:" : b.prefix;
        this.key = b.key;
        this.idColumn = b.idColumn;
        this.ttl = b.ttl;
        this.layout = b.layout == null ? Layout.HASH : b.layout;
        this.ifExists = b.ifExists == null ? IfExists.REPLACE : b.ifExists;
        this.pipelineBatch = Math.max(1, b.pipelineBatch);
        this.scanMatchPrefix = b.scanMatchPrefix;
        this.scanCount = Math.max(1, b.scanCount);
        this.dtype = b.dtype == null
                ? null
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.dtype));
        this.includeNulls = b.includeNulls;
        this.binaryVectorsAsBase64 = b.binaryVectorsAsBase64;
    }

    public static Builder builder() { return new Builder(); }

    public static RedisOptions defaults() { return builder().build(); }

    public static RedisOptions hash(String prefix) {
        return builder().prefix(prefix).layout(Layout.HASH).build();
    }

    public static RedisOptions hash(String prefix, Duration ttl) {
        return builder().prefix(prefix).layout(Layout.HASH).ttl(ttl).build();
    }

    public static RedisOptions json(String prefix) {
        return builder().prefix(prefix).layout(Layout.JSON).build();
    }

    public static RedisOptions json(String prefix, Duration ttl) {
        return builder().prefix(prefix).layout(Layout.JSON).ttl(ttl).build();
    }

    public static RedisOptions frame(String key) {
        return builder().key(key).layout(Layout.FRAME_JSON).build();
    }

    public static RedisOptions frame(String key, Duration ttl) {
        return builder().key(key).layout(Layout.FRAME_JSON).ttl(ttl).build();
    }

    public String prefix() { return prefix; }
    public String key() { return key; }
    public String idColumn() { return idColumn; }
    public Duration ttl() { return ttl; }
    public Layout layout() { return layout; }
    public IfExists ifExists() { return ifExists; }
    public int pipelineBatch() { return pipelineBatch; }
    public boolean scanMatchPrefix() { return scanMatchPrefix; }
    public int scanCount() { return scanCount; }
    public Map<String, Column.DType> dtype() { return dtype; }
    public boolean includeNulls() { return includeNulls; }
    public boolean binaryVectorsAsBase64() { return binaryVectorsAsBase64; }

    /** Effective TTL seconds, or {@code -1} if none. */
    public long ttlSeconds() {
        if (ttl == null || ttl.isZero() || ttl.isNegative()) return -1L;
        long s = ttl.getSeconds();
        // round up sub-second durations to 1s so EXPIRE is meaningful
        if (s == 0 && !ttl.isZero()) return 1L;
        return s;
    }

    /** Effective TTL milliseconds, or {@code -1} if none. */
    public long ttlMillis() {
        if (ttl == null || ttl.isZero() || ttl.isNegative()) return -1L;
        long ms = ttl.toMillis();
        return ms <= 0 ? 1L : ms;
    }

    public boolean hasTtl() {
        return ttlSeconds() > 0;
    }

    public static final class Builder {
        private String prefix = "df:";
        private String key;
        private String idColumn;
        private Duration ttl;
        private Layout layout = Layout.HASH;
        private IfExists ifExists = IfExists.REPLACE;
        private int pipelineBatch = 256;
        private boolean scanMatchPrefix = true;
        private int scanCount = 200;
        private Map<String, Column.DType> dtype;
        private boolean includeNulls = false;
        private boolean binaryVectorsAsBase64 = true;

        public Builder prefix(String v) {
            this.prefix = v == null ? "df:" : v;
            return this;
        }

        /** Single-key override for {@link Layout#FRAME_JSON} / {@link Layout#FRAME_JSONL}. */
        public Builder key(String v) {
            this.key = v;
            return this;
        }

        public Builder idColumn(String v) {
            this.idColumn = v;
            return this;
        }

        public Builder ttl(Duration v) {
            this.ttl = v;
            return this;
        }

        public Builder ttlSeconds(long seconds) {
            this.ttl = seconds <= 0 ? null : Duration.ofSeconds(seconds);
            return this;
        }

        public Builder ttlMillis(long millis) {
            this.ttl = millis <= 0 ? null : Duration.ofMillis(millis);
            return this;
        }

        public Builder layout(Layout v) {
            this.layout = v == null ? Layout.HASH : v;
            return this;
        }

        public Builder ifExists(IfExists v) {
            this.ifExists = v == null ? IfExists.REPLACE : v;
            return this;
        }

        public Builder pipelineBatch(int v) {
            this.pipelineBatch = Math.max(1, v);
            return this;
        }

        public Builder scanMatchPrefix(boolean v) {
            this.scanMatchPrefix = v;
            return this;
        }

        public Builder scanCount(int v) {
            this.scanCount = Math.max(1, v);
            return this;
        }

        public Builder dtype(Map<String, Column.DType> v) {
            this.dtype = v;
            return this;
        }

        public Builder includeNulls(boolean v) {
            this.includeNulls = v;
            return this;
        }

        public Builder binaryVectorsAsBase64(boolean v) {
            this.binaryVectorsAsBase64 = v;
            return this;
        }

        public RedisOptions build() {
            if ((layout == Layout.FRAME_JSON || layout == Layout.FRAME_JSONL)
                    && (key == null || key.isBlank())
                    && (prefix == null || prefix.isBlank())) {
                throw new IllegalArgumentException("FRAME_* layout requires key or prefix");
            }
            return new RedisOptions(this);
        }
    }

    /** Resolve the Redis key for a row id. */
    public String keyFor(Object id) {
        String idStr = id == null ? "null" : String.valueOf(id);
        if (layout == Layout.FRAME_JSON || layout == Layout.FRAME_JSONL) {
            if (key != null && !key.isBlank()) return key;
            return prefix.endsWith(":") || prefix.isEmpty() ? prefix + "frame" : prefix;
        }
        return prefix + idStr;
    }

    public String frameKey() {
        if (key != null && !key.isBlank()) return key;
        if (prefix == null || prefix.isEmpty()) return "df:frame";
        if (prefix.endsWith(":")) return prefix + "frame";
        return prefix;
    }

    @Override
    public String toString() {
        return "RedisOptions{layout=" + layout
                + ", prefix='" + prefix + '\''
                + ", key='" + key + '\''
                + ", idColumn='" + idColumn + '\''
                + ", ttl=" + ttl
                + ", ifExists=" + ifExists
                + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof RedisOptions that)) return false;
        return pipelineBatch == that.pipelineBatch
                && scanMatchPrefix == that.scanMatchPrefix
                && scanCount == that.scanCount
                && includeNulls == that.includeNulls
                && binaryVectorsAsBase64 == that.binaryVectorsAsBase64
                && Objects.equals(prefix, that.prefix)
                && Objects.equals(key, that.key)
                && Objects.equals(idColumn, that.idColumn)
                && Objects.equals(ttl, that.ttl)
                && layout == that.layout
                && ifExists == that.ifExists
                && Objects.equals(dtype, that.dtype);
    }

    @Override
    public int hashCode() {
        return Objects.hash(prefix, key, idColumn, ttl, layout, ifExists,
                pipelineBatch, scanMatchPrefix, scanCount, dtype, includeNulls, binaryVectorsAsBase64);
    }

    /** Parse a soft URI fragment like {@code hash://df:people:?ttl=3600&id=id}. */
    public static RedisOptions parse(String spec) {
        if (spec == null || spec.isBlank()) return defaults();
        Builder b = builder();
        String s = spec.trim();
        int schemeEnd = s.indexOf("://");
        String rest = s;
        if (schemeEnd > 0) {
            String scheme = s.substring(0, schemeEnd).toLowerCase(Locale.ROOT);
            rest = s.substring(schemeEnd + 3);
            switch (scheme) {
                case "hash", "hset" -> b.layout(Layout.HASH);
                case "json", "string" -> b.layout(Layout.JSON);
                case "frame", "framejson" -> b.layout(Layout.FRAME_JSON);
                case "jsonl", "framejsonl" -> b.layout(Layout.FRAME_JSONL);
                default -> { /* keep HASH */ }
            }
        }
        String path = rest;
        String query = null;
        int q = rest.indexOf('?');
        if (q >= 0) {
            path = rest.substring(0, q);
            query = rest.substring(q + 1);
        }
        if (!path.isBlank()) {
            if (path.contains("{") || path.endsWith(":") || path.contains("/")) {
                b.prefix(path.endsWith(":") || path.endsWith("/") ? path : path + ":");
            } else {
                b.key(path).prefix(path.endsWith(":") ? path : path + ":");
            }
        }
        if (query != null) {
            for (String pair : query.split("&")) {
                if (pair.isEmpty()) continue;
                int eq = pair.indexOf('=');
                String k = eq < 0 ? pair : pair.substring(0, eq);
                String v = eq < 0 ? "" : pair.substring(eq + 1);
                switch (k.toLowerCase(Locale.ROOT)) {
                    case "ttl", "ttl_s", "ttlseconds" -> {
                        try { b.ttlSeconds(Long.parseLong(v)); } catch (NumberFormatException ignored) {}
                    }
                    case "ttl_ms", "ttlmillis" -> {
                        try { b.ttlMillis(Long.parseLong(v)); } catch (NumberFormatException ignored) {}
                    }
                    case "id", "idcol", "idcolumn" -> b.idColumn(v);
                    case "layout" -> {
                        try { b.layout(Layout.valueOf(v.toUpperCase(Locale.ROOT))); }
                        catch (Exception ignored) {}
                    }
                    case "batch", "pipeline" -> {
                        try { b.pipelineBatch(Integer.parseInt(v)); } catch (NumberFormatException ignored) {}
                    }
                    case "prefix" -> b.prefix(v);
                    case "key" -> b.key(v);
                    default -> { }
                }
            }
        }
        return b.build();
    }
}
