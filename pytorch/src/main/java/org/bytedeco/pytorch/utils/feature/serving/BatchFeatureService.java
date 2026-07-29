/*
 * Batch feature service — historical (PIT) retrieval wrapper for training / backfill.
 */
package org.bytedeco.pytorch.utils.feature.serving;

import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.offline.HistoricalRetrievalJob;
import org.bytedeco.pytorch.utils.feature.offline.OfflineStore;
import org.bytedeco.pytorch.utils.feature.offline.TrainingDataset;
import org.bytedeco.pytorch.utils.feature.registry.FeatureRegistry;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Offline / batch serving path. */
public final class BatchFeatureService {

    private final FeatureRegistry registry;
    private final HistoricalRetrievalJob retrievalJob;

    public BatchFeatureService(FeatureRegistry registry, OfflineStore offlineStore) {
        this.registry = Objects.requireNonNull(registry, "registry");
        this.retrievalJob = new HistoricalRetrievalJob(registry, offlineStore);
    }

    public BatchFeatureService(FeatureRegistry registry, HistoricalRetrievalJob retrievalJob) {
        this.registry = Objects.requireNonNull(registry, "registry");
        this.retrievalJob = Objects.requireNonNull(retrievalJob, "retrievalJob");
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 String project,
                                                 String featureServiceName,
                                                 String labelColumn) {
        FeatureService svc = registry.requireFeatureService(project, featureServiceName);
        return retrievalJob.getHistoricalFeatures(entityRows, svc, labelColumn, null);
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 FeatureService service,
                                                 String labelColumn) {
        return retrievalJob.getHistoricalFeatures(entityRows, service, labelColumn, null);
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 List<FeatureView> views,
                                                 String labelColumn) {
        return retrievalJob.getHistoricalFeatures(entityRows, views, labelColumn, null);
    }

    /**
     * Convert a TrainingDataset row set into FeatureVectors (one per row).
     */
    public FeatureResponse toFeatureResponse(TrainingDataset dataset, String project, String service) {
        Objects.requireNonNull(dataset, "dataset");
        long t0 = System.nanoTime();
        List<FeatureVector> vectors = new ArrayList<>(dataset.size());
        for (Map<String, Object> row : dataset.rows()) {
            FeatureVector.Builder vb = FeatureVector.builder();
            for (String ek : dataset.entityKeys()) {
                if (row.containsKey(ek)) vb.entity(ek, row.get(ek));
            }
            for (String fc : dataset.featureColumns()) {
                FeatureVector.putTyped(vb, fc, row.get(fc));
            }
            if (dataset.labelColumn() != null && !dataset.labelColumn().isEmpty()) {
                vb.meta("label", String.valueOf(row.get(dataset.labelColumn())));
            }
            vb.meta("schema", dataset.schemaVersion());
            vectors.add(vb.build());
        }
        return FeatureResponse.builder()
                .project(project)
                .featureService(service)
                .vectors(vectors)
                .viewsHit(dataset.featureColumns().size())
                .elapsedNanos(System.nanoTime() - t0)
                .meta("schema", dataset.schemaVersion())
                .success(true)
                .build();
    }

    public HistoricalRetrievalJob retrievalJob() {
        return retrievalJob;
    }
}
