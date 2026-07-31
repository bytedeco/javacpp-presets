/*
 * FeaturePlatform — top-level façade wiring registry, offline/online stores,
 * materialization, and feature provider (Feast FeatureStore / Databricks FS).
 *
 * <pre>{@code
 * try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
 *     fp.registry().registerEntity(Entity.of("user_id"));
 *     fp.registry().registerFeatureView(userStats);
 *     fp.offline().put("default", "user_stats", rows);
 *     fp.materialize().materializeViews(List.of(userStats), Instant.EPOCH, Instant.now());
 *     FeatureResponse r = fp.provider().getOnlineFeatures("ranker_v1", Map.of("user_id", 1L));
 * }
 * }</pre>
 */
package org.bytedeco.pytorch.feature;

import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.feature.core.Project;
import org.bytedeco.pytorch.feature.core.StreamFeatureView;
import org.bytedeco.pytorch.feature.materialize.IncrementalCursor;
import org.bytedeco.pytorch.feature.materialize.MaterializationEngine;
import org.bytedeco.pytorch.feature.materialize.MaterializationResult;
import org.bytedeco.pytorch.feature.metrics.FeaturePlatformMetrics;
import org.bytedeco.pytorch.feature.offline.FileOfflineStore;
import org.bytedeco.pytorch.feature.offline.OfflineStore;
import org.bytedeco.pytorch.feature.online.InMemoryOnlineStore;
import org.bytedeco.pytorch.feature.online.OnlineStore;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;
import org.bytedeco.pytorch.feature.registry.FileRegistryStore;
import org.bytedeco.pytorch.feature.registry.InMemoryRegistryStore;
import org.bytedeco.pytorch.feature.registry.RegistryStore;
import org.bytedeco.pytorch.feature.serving.FeatureProvider;
import org.bytedeco.pytorch.feature.serving.FeatureRequest;
import org.bytedeco.pytorch.feature.serving.FeatureResponse;
import org.bytedeco.pytorch.feature.store.EmbeddingStore;
import org.bytedeco.pytorch.feature.store.MemoryEmbeddingStore;
import org.bytedeco.pytorch.feature.store.StoreConfig;
import org.bytedeco.pytorch.feature.store.StoreFactory;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Enterprise feature platform entry point. */
public final class FeaturePlatform implements AutoCloseable {

    private final FeatureRegistry registry;
    private final OfflineStore offlineStore;
    private final OnlineStore onlineStore;
    private final EmbeddingStore embeddingStore;
    private final StoreConfig storeConfig;
    private final MaterializationEngine materializationEngine;
    private final FeatureProvider provider;
    private final FeaturePlatformMetrics metrics;
    private final IncrementalCursor cursor;

    private FeaturePlatform(FeatureRegistry registry,
                            OfflineStore offlineStore,
                            OnlineStore onlineStore,
                            EmbeddingStore embeddingStore,
                            StoreConfig storeConfig,
                            FeaturePlatformMetrics metrics) {
        this.registry = Objects.requireNonNull(registry, "registry");
        this.offlineStore = Objects.requireNonNull(offlineStore, "offlineStore");
        this.onlineStore = Objects.requireNonNull(onlineStore, "onlineStore");
        this.embeddingStore = embeddingStore != null ? embeddingStore : new MemoryEmbeddingStore();
        this.storeConfig = storeConfig != null ? storeConfig : StoreConfig.memory();
        this.metrics = metrics != null ? metrics : new FeaturePlatformMetrics();
        this.cursor = new IncrementalCursor();
        this.materializationEngine = new MaterializationEngine(registry, offlineStore, onlineStore, cursor);
        this.provider = new FeatureProvider(registry, onlineStore, offlineStore);
    }

    public static FeaturePlatform inMemory() {
        return new FeaturePlatform(
                new FeatureRegistry(new InMemoryRegistryStore()),
                FileOfflineStore.inMemory(),
                new InMemoryOnlineStore(),
                new MemoryEmbeddingStore(),
                StoreConfig.memory(),
                new FeaturePlatformMetrics());
    }

    public static FeaturePlatform fileBacked(Path root) {
        Objects.requireNonNull(root, "root");
        return fromConfig(StoreConfig.localDurable(root));
    }

    /**
     * Build platform from a {@link StoreConfig} — switch Redis / SQLite / DuckDB /
     * Lance / Milvus without changing feature definitions.
     */
    public static FeaturePlatform fromConfig(StoreConfig config) {
        Objects.requireNonNull(config, "config");
        StoreFactory.Bundle bundle = StoreFactory.open(config);
        RegistryStore regStore;
        if (config.root() != null) {
            regStore = new FileRegistryStore(config.root().resolve("registry"));
        } else {
            regStore = new InMemoryRegistryStore();
        }
        return new FeaturePlatform(
                new FeatureRegistry(regStore),
                bundle.offline,
                bundle.online,
                bundle.embedding,
                config,
                new FeaturePlatformMetrics());
    }

    /** Convenience: DuckDB offline + Redis online (Feast-like prod shape). */
    public static FeaturePlatform duckdbRedis(Path duckRoot, String redisUri) {
        return fromConfig(StoreConfig.duckdbRedis(duckRoot, redisUri));
    }

    /** Convenience: Lance offline + Redis online + Milvus embeddings. */
    public static FeaturePlatform lanceRedisMilvus(Path lanceRoot, String redisUri,
                                                   String milvusUrl, int dim) {
        return fromConfig(StoreConfig.lanceRedisMilvus(lanceRoot, redisUri, milvusUrl, dim));
    }

    public static Builder builder() {
        return new Builder();
    }

    public FeatureRegistry registry() {
        return registry;
    }

    public OfflineStore offline() {
        return offlineStore;
    }

    public OnlineStore online() {
        return onlineStore;
    }

    /** Multimodal / tower embedding store (MEMORY / SQLITE / MILVUS / REDIS_VECTOR). */
    public EmbeddingStore embeddings() {
        return embeddingStore;
    }

    public StoreConfig storeConfig() {
        return storeConfig;
    }

    public MaterializationEngine materialize() {
        return materializationEngine;
    }

    public FeatureProvider provider() {
        return provider;
    }

    public FeaturePlatformMetrics metrics() {
        return metrics;
    }

    public IncrementalCursor cursor() {
        return cursor;
    }

    // ── convenience registration ────────────────────────────────────────────

    public Project project(String name) {
        return registry.registerProject(Project.of(name));
    }

    public Entity entity(Entity entity) {
        metrics.inc(FeaturePlatformMetrics.REGISTRY_REGISTER);
        return registry.registerEntity(entity);
    }

    public FeatureView featureView(FeatureView view) {
        metrics.inc(FeaturePlatformMetrics.REGISTRY_REGISTER);
        return registry.registerFeatureView(view);
    }

    public OnDemandFeatureView onDemand(OnDemandFeatureView view) {
        metrics.inc(FeaturePlatformMetrics.REGISTRY_REGISTER);
        return registry.registerOnDemandFeatureView(view);
    }

    public StreamFeatureView streamView(StreamFeatureView view) {
        metrics.inc(FeaturePlatformMetrics.REGISTRY_REGISTER);
        return registry.registerStreamFeatureView(view);
    }

    public FeatureService featureService(FeatureService service) {
        metrics.inc(FeaturePlatformMetrics.REGISTRY_REGISTER);
        return registry.registerFeatureService(service);
    }

    public void putOffline(String project, String viewName, List<Map<String, Object>> rows) {
        offlineStore.put(project, viewName, rows);
    }

    public MaterializationResult materializeAll(String project) {
        long t0 = System.nanoTime();
        MaterializationResult r = materializationEngine.materializeProject(project, Instant.EPOCH, Instant.now());
        metrics.record(FeaturePlatformMetrics.MATERIALIZE_LATENCY, System.nanoTime() - t0);
        metrics.counter(FeaturePlatformMetrics.MATERIALIZE_ROWS).add(r.rowsWritten());
        return r;
    }

    public MaterializationResult materializeViews(List<FeatureView> views) {
        long t0 = System.nanoTime();
        MaterializationResult r = materializationEngine.materializeViews(views, Instant.EPOCH, Instant.now());
        metrics.record(FeaturePlatformMetrics.MATERIALIZE_LATENCY, System.nanoTime() - t0);
        metrics.counter(FeaturePlatformMetrics.MATERIALIZE_ROWS).add(r.rowsWritten());
        return r;
    }

    public FeatureResponse getOnlineFeatures(FeatureRequest request) {
        long t0 = System.nanoTime();
        FeatureResponse r = provider.getOnlineFeatures(request);
        long dt = System.nanoTime() - t0;
        metrics.record(FeaturePlatformMetrics.ONLINE_LATENCY, dt);
        metrics.inc(FeaturePlatformMetrics.ONLINE_GET);
        if (r.viewsMiss() > 0) {
            metrics.counter(FeaturePlatformMetrics.ONLINE_MISS).add(r.viewsMiss());
        }
        return r;
    }

    public FeatureResponse getOnlineFeatures(String featureService, Map<String, Object> entities) {
        return getOnlineFeatures(FeatureRequest.builder()
                .featureService(featureService)
                .entities(entities)
                .build());
    }

    @Override
    public void close() {
        try {
            registry.close();
        } catch (Exception ignored) {
        }
        try {
            offlineStore.close();
        } catch (Exception ignored) {
        }
        try {
            onlineStore.close();
        } catch (Exception ignored) {
        }
        try {
            embeddingStore.close();
        } catch (Exception ignored) {
        }
    }

    public static final class Builder {
        private RegistryStore registryStore;
        private OfflineStore offlineStore;
        private OnlineStore onlineStore;
        private EmbeddingStore embeddingStore;
        private StoreConfig storeConfig;
        private FeaturePlatformMetrics metrics;

        public Builder registryStore(RegistryStore registryStore) {
            this.registryStore = registryStore;
            return this;
        }

        public Builder offlineStore(OfflineStore offlineStore) {
            this.offlineStore = offlineStore;
            return this;
        }

        public Builder onlineStore(OnlineStore onlineStore) {
            this.onlineStore = onlineStore;
            return this;
        }

        public Builder embeddingStore(EmbeddingStore embeddingStore) {
            this.embeddingStore = embeddingStore;
            return this;
        }

        /** Wire all three stores from a single {@link StoreConfig} (overridable per-slot). */
        public Builder stores(StoreConfig storeConfig) {
            this.storeConfig = storeConfig;
            return this;
        }

        public Builder metrics(FeaturePlatformMetrics metrics) {
            this.metrics = metrics;
            return this;
        }

        public FeaturePlatform build() {
            StoreConfig cfg = storeConfig != null ? storeConfig : StoreConfig.memory();
            OfflineStore off = offlineStore;
            OnlineStore on = onlineStore;
            EmbeddingStore emb = embeddingStore;
            if (off == null || on == null || emb == null) {
                StoreFactory.Bundle bundle = StoreFactory.open(cfg);
                if (off == null) off = bundle.offline;
                if (on == null) on = bundle.online;
                if (emb == null) emb = bundle.embedding;
            }
            RegistryStore regStore = registryStore;
            if (regStore == null) {
                regStore = cfg.root() != null
                        ? new FileRegistryStore(cfg.root().resolve("registry"))
                        : new InMemoryRegistryStore();
            }
            FeatureRegistry reg = new FeatureRegistry(regStore);
            return new FeaturePlatform(reg, off, on, emb, cfg,
                    metrics != null ? metrics : new FeaturePlatformMetrics());
        }
    }
}
