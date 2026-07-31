/*
 * StoreFactory — create OnlineStore / OfflineStore / EmbeddingStore from StoreConfig.
 *
 * Switch production backends without changing feature definitions:
 * <pre>{@code
 * StoreConfig cfg = StoreConfig.builder()
 *     .online("redis").redisUri("redis://feat-redis:6379/0")
 *     .offline("duckdb").root(Path.of("/data/features"))
 *     .embedding("milvus").milvusUrl("http://milvus:9091").embeddingDim(64)
 *     .build();
 * try (FeaturePlatform fp = FeaturePlatform.builder().stores(cfg).build()) { ... }
 * }</pre>
 *
 * Remote backends that are unreachable fall back to in-memory with a warning
 * only when {@code options.fallback_memory=true} (default false for online/embedding
 * so misconfig fails loudly in prod). Offline duckdb/lance always have local fallbacks.
 */
package org.bytedeco.pytorch.feature.store;

import org.bytedeco.pytorch.feature.offline.DuckDbOfflineStore;
import org.bytedeco.pytorch.feature.offline.FileOfflineStore;
import org.bytedeco.pytorch.feature.offline.LanceOfflineStore;
import org.bytedeco.pytorch.feature.offline.OfflineStore;
import org.bytedeco.pytorch.feature.offline.SqliteOfflineStore;
import org.bytedeco.pytorch.feature.online.FileOnlineStore;
import org.bytedeco.pytorch.feature.online.InMemoryOnlineStore;
import org.bytedeco.pytorch.feature.online.OnlineStore;
import org.bytedeco.pytorch.feature.online.RedisOnlineStore;
import org.bytedeco.pytorch.feature.online.SqliteOnlineStore;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;

/** Factory for pluggable feature-platform stores. */
public final class StoreFactory {

    public static final class Bundle implements AutoCloseable {
        public final OnlineStore online;
        public final OfflineStore offline;
        public final EmbeddingStore embedding;
        public final StoreConfig config;
        public final List<String> notes;

        Bundle(OnlineStore online, OfflineStore offline, EmbeddingStore embedding,
               StoreConfig config, List<String> notes) {
            this.online = online;
            this.offline = offline;
            this.embedding = embedding;
            this.config = config;
            this.notes = List.copyOf(notes);
        }

        @Override
        public void close() {
            closeQuietly(online);
            closeQuietly(offline);
            closeQuietly(embedding);
        }

        private static void closeQuietly(AutoCloseable c) {
            if (c == null) return;
            try {
                c.close();
            } catch (Exception ignored) {
            }
        }
    }

    private StoreFactory() {}

    public static Bundle open(StoreConfig config) {
        return open(config, null);
    }

    public static Bundle open(StoreConfig config, Consumer<String> log) {
        Objects.requireNonNull(config, "config");
        List<String> notes = new ArrayList<>();
        Consumer<String> logger = log != null ? log : notes::add;

        OfflineStore offline = createOffline(config, logger);
        OnlineStore online = createOnline(config, logger);
        EmbeddingStore embedding = createEmbedding(config, logger);

        logger.accept("stores ready: online=" + config.onlineBackend()
                + " offline=" + config.offlineBackend()
                + " embedding=" + config.embeddingBackend());
        return new Bundle(online, offline, embedding, config, notes);
    }

    public static OnlineStore createOnline(StoreConfig config) {
        return createOnline(config, s -> {});
    }

    public static OnlineStore createOnline(StoreConfig config, Consumer<String> log) {
        Objects.requireNonNull(config, "config");
        boolean fallback = truthy(config.option("fallback_memory", "false"));
        StoreBackend b = config.onlineBackend();
        try {
            switch (b) {
                case MEMORY:
                    return new InMemoryOnlineStore();
                case FILE: {
                    Path root = config.root() != null
                            ? config.root().resolve("online")
                            : Path.of("feature-data", "online");
                    return new FileOnlineStore(root);
                }
                case SQLITE: {
                    String path = config.sqliteOnlinePath();
                    if (path == null || path.isBlank()) {
                        if (config.root() != null) {
                            path = config.root().resolve("online.db").toString();
                        }
                    }
                    if (path == null || path.isBlank() || ":memory:".equals(path)) {
                        return SqliteOnlineStore.inMemory();
                    }
                    return SqliteOnlineStore.open(Path.of(path));
                }
                case REDIS: {
                    RedisOnlineStore store = RedisOnlineStore.connect(
                            config.redisUri(), config.redisKeyPrefix(), config.redisTtl());
                    if (!store.available()) {
                        store.close();
                        if (fallback) {
                            log.accept("WARN: Redis unavailable at " + config.redisUri()
                                    + " — falling back to MEMORY online store");
                            return new InMemoryOnlineStore();
                        }
                        throw new IllegalStateException("Redis not available at " + config.redisUri()
                                + " (set option fallback_memory=true to degrade)");
                    }
                    log.accept("Redis online store connected: " + config.redisUri());
                    return store;
                }
                default:
                    log.accept("WARN: online backend " + b + " not online-capable; using MEMORY");
                    return new InMemoryOnlineStore();
            }
        } catch (RuntimeException e) {
            if (fallback && b != StoreBackend.MEMORY) {
                log.accept("WARN: online " + b + " failed (" + e.getMessage()
                        + ") — fallback MEMORY");
                return new InMemoryOnlineStore();
            }
            throw e;
        }
    }

    public static OfflineStore createOffline(StoreConfig config) {
        return createOffline(config, s -> {});
    }

    public static OfflineStore createOffline(StoreConfig config, Consumer<String> log) {
        Objects.requireNonNull(config, "config");
        StoreBackend b = config.offlineBackend();
        switch (b) {
            case MEMORY:
                return FileOfflineStore.inMemory();
            case FILE: {
                Path root = config.root() != null
                        ? config.root().resolve("offline")
                        : Path.of("feature-data", "offline");
                return new FileOfflineStore(root);
            }
            case DUCKDB: {
                Path root = config.root() != null ? config.root().resolve("duckdb") : null;
                DuckDbOfflineStore store = root != null
                        ? new DuckDbOfflineStore(root)
                        : new DuckDbOfflineStore();
                log.accept("DuckDB offline store (available=" + store.duckAvailable() + ")");
                return store;
            }
            case LANCE: {
                Path root = config.root() != null
                        ? config.root().resolve("lance")
                        : Path.of("feature-data", "lance");
                return new LanceOfflineStore(root);
            }
            case SQLITE: {
                String path = config.sqliteOfflinePath();
                if (path == null || path.isBlank()) {
                    if (config.root() != null) {
                        path = config.root().resolve("offline.db").toString();
                    }
                }
                if (path == null || path.isBlank() || ":memory:".equals(path)) {
                    return SqliteOfflineStore.inMemory();
                }
                return SqliteOfflineStore.open(Path.of(path));
            }
            default:
                log.accept("WARN: offline backend " + b + " not offline-capable; using MEMORY");
                return FileOfflineStore.inMemory();
        }
    }

    public static EmbeddingStore createEmbedding(StoreConfig config) {
        return createEmbedding(config, s -> {});
    }

    public static EmbeddingStore createEmbedding(StoreConfig config, Consumer<String> log) {
        Objects.requireNonNull(config, "config");
        boolean fallback = truthy(config.option("fallback_memory", "false"));
        StoreBackend b = config.embeddingBackend();
        int dim = config.embeddingDim();
        try {
            switch (b) {
                case MEMORY:
                    return new MemoryEmbeddingStore(dim);
                case SQLITE: {
                    String path = config.option("sqlite_embedding_path", null);
                    if (path == null && config.root() != null) {
                        path = config.root().resolve("embeddings.db").toString();
                    }
                    if (path == null || path.isBlank() || ":memory:".equals(path)) {
                        return SqliteEmbeddingStoreAdapter.inMemory(dim);
                    }
                    return SqliteEmbeddingStoreAdapter.open(Path.of(path), dim);
                }
                case MILVUS: {
                    MilvusEmbeddingStore store = MilvusEmbeddingStore.connect(config);
                    if (!store.available()) {
                        store.close();
                        if (fallback) {
                            log.accept("WARN: Milvus unavailable at " + config.milvusUrl()
                                    + " — falling back to MEMORY embedding store");
                            return new MemoryEmbeddingStore(dim);
                        }
                        throw new IllegalStateException("Milvus not available at " + config.milvusUrl()
                                + " (set option fallback_memory=true to degrade)");
                    }
                    log.accept("Milvus embedding store connected: " + config.milvusUrl());
                    return store;
                }
                case REDIS_VECTOR: {
                    RedisVectorEmbeddingStore store = RedisVectorEmbeddingStore.connect(config);
                    if (!store.available()) {
                        store.close();
                        if (fallback) {
                            log.accept("WARN: Redis vector unavailable — fallback MEMORY embeddings");
                            return new MemoryEmbeddingStore(dim);
                        }
                        throw new IllegalStateException("Redis vector not available at " + config.redisUri());
                    }
                    log.accept("Redis vector embedding store connected: " + config.redisUri());
                    return store;
                }
                case LANCE:
                    // Lance path reuses memory index + file offline for vectors in this round
                    log.accept("NOTE: LANCE embedding backend uses memory ANN + lance offline tables");
                    return new MemoryEmbeddingStore(dim);
                default:
                    return new MemoryEmbeddingStore(dim);
            }
        } catch (RuntimeException e) {
            if (fallback && b != StoreBackend.MEMORY) {
                log.accept("WARN: embedding " + b + " failed (" + e.getMessage()
                        + ") — fallback MEMORY");
                return new MemoryEmbeddingStore(dim);
            }
            throw e;
        }
    }

    private static boolean truthy(String v) {
        if (v == null) return false;
        String s = v.trim().toLowerCase();
        return s.equals("1") || s.equals("true") || s.equals("yes") || s.equals("on");
    }
}
