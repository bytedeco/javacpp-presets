/*
 * Unified store configuration for online / offline / embedding backends.
 * Switch production vs local by changing backends — FeaturePlatform.builder().stores(cfg).
 */
package org.bytedeco.pytorch.utils.feature.store;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Immutable multi-backend store configuration. */
public final class StoreConfig {

    private final StoreBackend onlineBackend;
    private final StoreBackend offlineBackend;
    private final StoreBackend embeddingBackend;

    private final Path root;                 // FILE / SQLITE / LANCE base
    private final String redisUri;           // redis://host:port/db
    private final String redisKeyPrefix;
    private final Duration redisTtl;

    private final String milvusUrl;
    private final String milvusToken;
    private final String milvusCollection;
    private final int embeddingDim;
    private final String embeddingMetric;    // COSINE / L2 / IP

    private final String sqliteOnlinePath;   // null → in-memory or root/online.db
    private final String sqliteOfflinePath;
    private final String duckdbPath;         // null → in-memory duck

    private final Map<String, String> options;

    private StoreConfig(Builder b) {
        this.onlineBackend = b.onlineBackend != null ? b.onlineBackend : StoreBackend.MEMORY;
        this.offlineBackend = b.offlineBackend != null ? b.offlineBackend : StoreBackend.MEMORY;
        this.embeddingBackend = b.embeddingBackend != null ? b.embeddingBackend : StoreBackend.MEMORY;
        this.root = b.root;
        this.redisUri = b.redisUri != null ? b.redisUri : "redis://127.0.0.1:6379/0";
        this.redisKeyPrefix = b.redisKeyPrefix != null ? b.redisKeyPrefix : "fs:";
        this.redisTtl = b.redisTtl;
        this.milvusUrl = b.milvusUrl != null ? b.milvusUrl : "http://127.0.0.1:9091";
        this.milvusToken = b.milvusToken != null ? b.milvusToken : "";
        this.milvusCollection = b.milvusCollection != null ? b.milvusCollection : "feature_embeddings";
        this.embeddingDim = b.embeddingDim > 0 ? b.embeddingDim : 64;
        this.embeddingMetric = b.embeddingMetric != null ? b.embeddingMetric : "COSINE";
        this.sqliteOnlinePath = b.sqliteOnlinePath;
        this.sqliteOfflinePath = b.sqliteOfflinePath;
        this.duckdbPath = b.duckdbPath;
        this.options = Collections.unmodifiableMap(new LinkedHashMap<>(b.options));
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Dev defaults: all in-memory. */
    public static StoreConfig memory() {
        return builder()
                .online(StoreBackend.MEMORY)
                .offline(StoreBackend.MEMORY)
                .embedding(StoreBackend.MEMORY)
                .build();
    }

    /** Local durable: file offline + sqlite online + sqlite embedding. */
    public static StoreConfig localDurable(Path root) {
        Objects.requireNonNull(root, "root");
        return builder()
                .root(root)
                .online(StoreBackend.SQLITE)
                .offline(StoreBackend.FILE)
                .embedding(StoreBackend.SQLITE)
                .sqliteOnlinePath(root.resolve("online.db").toString())
                .sqliteOfflinePath(root.resolve("offline.db").toString())
                .build();
    }

    /** Analytics offline on DuckDB + Redis online (common Feast prod shape). */
    public static StoreConfig duckdbRedis(Path duckRoot, String redisUri) {
        return builder()
                .root(duckRoot)
                .offline(StoreBackend.DUCKDB)
                .online(StoreBackend.REDIS)
                .embedding(StoreBackend.MEMORY)
                .redisUri(redisUri)
                .build();
    }

    /** Multimodal: Lance offline + Redis online + Milvus embeddings. */
    public static StoreConfig lanceRedisMilvus(Path lanceRoot, String redisUri, String milvusUrl, int dim) {
        return builder()
                .root(lanceRoot)
                .offline(StoreBackend.LANCE)
                .online(StoreBackend.REDIS)
                .embedding(StoreBackend.MILVUS)
                .redisUri(redisUri)
                .milvusUrl(milvusUrl)
                .embeddingDim(dim)
                .build();
    }

    public StoreBackend onlineBackend() { return onlineBackend; }
    public StoreBackend offlineBackend() { return offlineBackend; }
    public StoreBackend embeddingBackend() { return embeddingBackend; }
    public Path root() { return root; }
    public String redisUri() { return redisUri; }
    public String redisKeyPrefix() { return redisKeyPrefix; }
    public Duration redisTtl() { return redisTtl; }
    public String milvusUrl() { return milvusUrl; }
    public String milvusToken() { return milvusToken; }
    public String milvusCollection() { return milvusCollection; }
    public int embeddingDim() { return embeddingDim; }
    public String embeddingMetric() { return embeddingMetric; }
    public String sqliteOnlinePath() { return sqliteOnlinePath; }
    public String sqliteOfflinePath() { return sqliteOfflinePath; }
    public String duckdbPath() { return duckdbPath; }
    public Map<String, String> options() { return options; }

    public String option(String key, String dflt) {
        return options.getOrDefault(key, dflt);
    }

    public Path resolveRoot(String child) {
        if (root == null) return Path.of(child);
        return root.resolve(child);
    }

    @Override
    public String toString() {
        return "StoreConfig{online=" + onlineBackend
                + ", offline=" + offlineBackend
                + ", embedding=" + embeddingBackend
                + ", root=" + root
                + "}";
    }

    public static final class Builder {
        private StoreBackend onlineBackend = StoreBackend.MEMORY;
        private StoreBackend offlineBackend = StoreBackend.MEMORY;
        private StoreBackend embeddingBackend = StoreBackend.MEMORY;
        private Path root;
        private String redisUri;
        private String redisKeyPrefix;
        private Duration redisTtl;
        private String milvusUrl;
        private String milvusToken;
        private String milvusCollection;
        private int embeddingDim = 64;
        private String embeddingMetric = "COSINE";
        private String sqliteOnlinePath;
        private String sqliteOfflinePath;
        private String duckdbPath;
        private final Map<String, String> options = new LinkedHashMap<>();

        public Builder online(StoreBackend b) { this.onlineBackend = b; return this; }
        public Builder offline(StoreBackend b) { this.offlineBackend = b; return this; }
        public Builder embedding(StoreBackend b) { this.embeddingBackend = b; return this; }
        public Builder online(String b) { return online(StoreBackend.parse(b)); }
        public Builder offline(String b) { return offline(StoreBackend.parse(b)); }
        public Builder embedding(String b) { return embedding(StoreBackend.parse(b)); }

        public Builder root(Path root) { this.root = root; return this; }
        public Builder redisUri(String redisUri) { this.redisUri = redisUri; return this; }
        public Builder redisKeyPrefix(String redisKeyPrefix) { this.redisKeyPrefix = redisKeyPrefix; return this; }
        public Builder redisTtl(Duration redisTtl) { this.redisTtl = redisTtl; return this; }
        public Builder milvusUrl(String milvusUrl) { this.milvusUrl = milvusUrl; return this; }
        public Builder milvusToken(String milvusToken) { this.milvusToken = milvusToken; return this; }
        public Builder milvusCollection(String milvusCollection) { this.milvusCollection = milvusCollection; return this; }
        public Builder embeddingDim(int embeddingDim) { this.embeddingDim = embeddingDim; return this; }
        public Builder embeddingMetric(String embeddingMetric) { this.embeddingMetric = embeddingMetric; return this; }
        public Builder sqliteOnlinePath(String sqliteOnlinePath) { this.sqliteOnlinePath = sqliteOnlinePath; return this; }
        public Builder sqliteOfflinePath(String sqliteOfflinePath) { this.sqliteOfflinePath = sqliteOfflinePath; return this; }
        public Builder duckdbPath(String duckdbPath) { this.duckdbPath = duckdbPath; return this; }
        public Builder option(String k, String v) {
            if (k != null && v != null) options.put(k, v);
            return this;
        }

        public StoreConfig build() {
            return new StoreConfig(this);
        }
    }
}
