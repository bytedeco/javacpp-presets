/*
 * Pluggable storage backends for FeaturePlatform online / offline / embedding.
 *
 * Production switch matrix (Feast / Tecton / Databricks / Alibaba / ByteDance):
 *   ONLINE:    MEMORY | FILE | REDIS | SQLITE
 *   OFFLINE:   MEMORY | FILE | DUCKDB | LANCE | SQLITE
 *   EMBEDDING: MEMORY | SQLITE | MILVUS | REDIS_VECTOR | LANCE
 */
package org.bytedeco.pytorch.feature.store;

import java.util.Locale;

/** Named storage backend kinds. */
public enum StoreBackend {
    /** Process-local ConcurrentHashMap (dev / unit test). */
    MEMORY,
    /** JSONL / file snapshot under a root path. */
    FILE,
    /** Redis hash/string KV online store (Feast RedisOnlineStore). */
    REDIS,
    /** SQLite WAL online/offline / embedding sidecar. */
    SQLITE,
    /** DuckDB OLAP offline (parquet scans, ASOF). */
    DUCKDB,
    /** Lance multimodal / vector offline tables. */
    LANCE,
    /** Milvus vector collection for embedding features. */
    MILVUS,
    /** Redis Stack / RediSearch vector index. */
    REDIS_VECTOR;

    public boolean isOnlineCapable() {
        switch (this) {
            case MEMORY:
            case FILE:
            case REDIS:
            case SQLITE:
                return true;
            default:
                return false;
        }
    }

    public boolean isOfflineCapable() {
        switch (this) {
            case MEMORY:
            case FILE:
            case DUCKDB:
            case LANCE:
            case SQLITE:
                return true;
            default:
                return false;
        }
    }

    public boolean isEmbeddingCapable() {
        switch (this) {
            case MEMORY:
            case SQLITE:
            case MILVUS:
            case REDIS_VECTOR:
            case LANCE:
                return true;
            default:
                return false;
        }
    }

    public static StoreBackend parse(String raw) {
        if (raw == null || raw.isBlank()) return MEMORY;
        String s = raw.trim().toUpperCase(Locale.ROOT).replace('-', '_');
        switch (s) {
            case "MEM":
            case "INMEMORY":
            case "IN_MEMORY":
                return MEMORY;
            case "REDISSTACK":
            case "REDIS_STACK":
            case "REDSEARCH":
            case "REDISEARCH":
                return REDIS_VECTOR;
            default:
                return StoreBackend.valueOf(s);
        }
    }
}
