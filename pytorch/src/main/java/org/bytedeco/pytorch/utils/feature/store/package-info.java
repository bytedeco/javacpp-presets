/**
 * Pluggable storage backends for the feature platform.
 *
 * <h2>Switch matrix</h2>
 * <table>
 *   <tr><th>Role</th><th>Backends</th></tr>
 *   <tr><td>Online KV</td><td>{@code MEMORY}, {@code FILE}, {@code REDIS}, {@code SQLITE}</td></tr>
 *   <tr><td>Offline</td><td>{@code MEMORY}, {@code FILE}, {@code DUCKDB}, {@code LANCE}, {@code SQLITE}</td></tr>
 *   <tr><td>Embedding / ANN</td><td>{@code MEMORY}, {@code SQLITE}, {@code MILVUS}, {@code REDIS_VECTOR}</td></tr>
 * </table>
 *
 * <pre>{@code
 * // local durable
 * FeaturePlatform fp = FeaturePlatform.fromConfig(StoreConfig.localDurable(Path.of("/data/fs")));
 *
 * // prod-like: DuckDB offline + Redis online
 * FeaturePlatform fp2 = FeaturePlatform.duckdbRedis(Path.of("/data/duck"), "redis://feat:6379/0");
 *
 * // multimodal retrieval: Lance + Redis + Milvus
 * FeaturePlatform fp3 = FeaturePlatform.lanceRedisMilvus(
 *     Path.of("/data/lance"), "redis://feat:6379/0", "http://milvus:9091", 64);
 *
 * // fine-grained
 * StoreConfig cfg = StoreConfig.builder()
 *     .online("sqlite").sqliteOnlinePath("/data/online.db")
 *     .offline("duckdb").root(Path.of("/data"))
 *     .embedding("sqlite")
 *     .option("fallback_memory", "true")  // degrade if remote down
 *     .build();
 * FeaturePlatform fp4 = FeaturePlatform.builder().stores(cfg).build();
 * }</pre>
 *
 * <p>Remote clients reuse existing zero-SDK adapters:
 * {@link org.bytedeco.pytorch.dataframe.redis.Redis},
 * {@link org.bytedeco.pytorch.dataframe.milvus.Milvus},
 * {@link org.bytedeco.pytorch.utils.sqlite.SQLite}.
 *
 * @see StoreConfig
 * @see StoreFactory
 * @see EmbeddingStore
 */
package org.bytedeco.pytorch.utils.feature.store;
