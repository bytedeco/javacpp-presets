/*
 * Feature Provider — Databricks-style unified online/offline consumption API.
 */
package org.bytedeco.pytorch.feature.serving;

import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.offline.OfflineStore;
import org.bytedeco.pytorch.feature.offline.TrainingDataset;
import org.bytedeco.pytorch.feature.online.OnlineStore;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Unified feature consumption façade. */
public final class FeatureProvider {

    private final FeatureRegistry registry;
    private final OnlineFeatureService online;
    private final BatchFeatureService batch;

    public FeatureProvider(FeatureRegistry registry, OnlineStore onlineStore, OfflineStore offlineStore) {
        this.registry = Objects.requireNonNull(registry, "registry");
        this.online = new OnlineFeatureService(registry, onlineStore);
        this.batch = new BatchFeatureService(registry, offlineStore);
    }

    public FeatureRegistry registry() {
        return registry;
    }

    public OnlineFeatureService online() {
        return online;
    }

    public BatchFeatureService batch() {
        return batch;
    }

    public FeatureResponse getOnlineFeatures(FeatureRequest request) {
        return online.getOnlineFeatures(request);
    }

    public FeatureResponse getOnlineFeatures(String featureService, Map<String, Object> entities) {
        return online.getOnlineFeatures(FeatureRequest.builder()
                .featureService(featureService)
                .entities(entities)
                .build());
    }

    public FeatureResponse getOnlineFeatures(String project, String featureService,
                                             Map<String, Object> entities,
                                             Map<String, Object> requestContext) {
        return online.getOnlineFeatures(FeatureRequest.builder()
                .project(project)
                .featureService(featureService)
                .entities(entities)
                .requestContext(requestContext)
                .build());
    }

    /** Ranking fanout: one shared user map + N item maps merged per candidate. */
    public FeatureResponse getOnlineFeaturesBatch(String project, String featureService,
                                                  List<Map<String, Object>> entityRows,
                                                  Map<String, Object> requestContext) {
        return online.getOnlineFeatures(FeatureRequest.builder()
                .project(project)
                .featureService(featureService)
                .entityRows(entityRows)
                .requestContext(requestContext)
                .build());
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 String project,
                                                 String featureService,
                                                 String labelColumn) {
        return batch.getHistoricalFeatures(entityRows, project, featureService, labelColumn);
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 FeatureService service,
                                                 String labelColumn) {
        return batch.getHistoricalFeatures(entityRows, service, labelColumn);
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 List<FeatureView> views,
                                                 String labelColumn) {
        return batch.getHistoricalFeatures(entityRows, views, labelColumn);
    }
}
