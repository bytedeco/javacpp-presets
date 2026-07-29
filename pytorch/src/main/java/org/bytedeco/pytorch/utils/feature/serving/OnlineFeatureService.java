/*
 * Online feature service — low-latency get_online_features path.
 *
 * Reads OnlineStore by entity key per FeatureView, then applies OnDemandFeatureViews.
 * Aligns with Feast OnlineFeatureServer / Databricks Feature Serving.
 */
package org.bytedeco.pytorch.utils.feature.serving;

import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.utils.feature.offline.PointInTimeJoin;
import org.bytedeco.pytorch.utils.feature.online.OnlineFeatureRow;
import org.bytedeco.pytorch.utils.feature.online.OnlineStore;
import org.bytedeco.pytorch.utils.feature.registry.FeatureRegistry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Online serving path. */
public final class OnlineFeatureService {

    private final FeatureRegistry registry;
    private final OnlineStore onlineStore;

    public OnlineFeatureService(FeatureRegistry registry, OnlineStore onlineStore) {
        this.registry = Objects.requireNonNull(registry, "registry");
        this.onlineStore = Objects.requireNonNull(onlineStore, "onlineStore");
    }

    public FeatureResponse getOnlineFeatures(FeatureRequest request) {
        Objects.requireNonNull(request, "request");
        long t0 = System.nanoTime();
        FeatureResponse.Builder resp = FeatureResponse.builder()
                .project(request.project())
                .featureService(request.featureService());

        try {
            List<FeatureView> views = resolveViews(request);
            List<OnDemandFeatureView> onDemand = request.includeOnDemand()
                    ? resolveOnDemand(request)
                    : List.of();

            List<Map<String, Object>> entityRows = request.effectiveEntityRows();
            if (entityRows.isEmpty()) {
                return resp.error("no entity keys in request").elapsedNanos(System.nanoTime() - t0).build();
            }

            int hit = 0;
            int miss = 0;
            int od = 0;
            List<FeatureVector> vectors = new ArrayList<>(entityRows.size());

            for (Map<String, Object> entity : entityRows) {
                FeatureVector.Builder vb = FeatureVector.builder().entities(entity);
                Map<String, Map<String, Object>> sourcesByView = new LinkedHashMap<>();

                for (FeatureView view : views) {
                    List<String> joinKeys = view.joinKeys().isEmpty() ? view.entityNames() : view.joinKeys();
                    String ek = PointInTimeJoin.entityKey(entity, joinKeys);
                    Optional<OnlineFeatureRow> row = onlineStore.onlineRead(view.project(), view.name(), ek);
                    if (row.isPresent()) {
                        hit++;
                        Map<String, Object> vals = row.get().values();
                        sourcesByView.put(view.name(), vals);
                        for (Map.Entry<String, Object> e : vals.entrySet()) {
                            String col = view.name() + "__" + e.getKey();
                            FeatureVector.putTyped(vb, col, e.getValue());
                            FeatureVector.putTyped(vb, e.getKey(), e.getValue());
                        }
                        vb.meta("ts:" + view.name(), String.valueOf(row.get().eventTimestampMs()));
                    } else {
                        miss++;
                    }
                }

                if (!onDemand.isEmpty()) {
                    for (OnDemandFeatureView odv : onDemand) {
                        Map<String, Object> computed = odv.apply(request.requestContext(), sourcesByView);
                        od++;
                        for (Map.Entry<String, Object> e : computed.entrySet()) {
                            String col = odv.name() + "__" + e.getKey();
                            FeatureVector.putTyped(vb, col, e.getValue());
                            FeatureVector.putTyped(vb, e.getKey(), e.getValue());
                        }
                    }
                }

                vectors.add(vb.meta("project", request.project()).build());
            }

            return resp
                    .vectors(vectors)
                    .viewsHit(hit)
                    .viewsMiss(miss)
                    .onDemandComputed(od)
                    .elapsedNanos(System.nanoTime() - t0)
                    .success(true)
                    .build();
        } catch (RuntimeException ex) {
            return resp
                    .error(ex.getMessage() != null ? ex.getMessage() : ex.getClass().getSimpleName())
                    .elapsedNanos(System.nanoTime() - t0)
                    .build();
        }
    }

    private List<FeatureView> resolveViews(FeatureRequest request) {
        if (request.featureService() != null && !request.featureService().isEmpty()) {
            FeatureService svc = registry.requireFeatureService(request.project(), request.featureService());
            return registry.resolveViews(svc);
        }
        List<FeatureView> views = new ArrayList<>();
        for (String vn : request.viewNames()) {
            registry.getFeatureView(request.project(), vn).ifPresent(views::add);
            registry.getStreamFeatureView(request.project(), vn).ifPresent(sfv -> views.add(sfv.asBatchView()));
        }
        return views;
    }

    private List<OnDemandFeatureView> resolveOnDemand(FeatureRequest request) {
        if (request.featureService() != null && !request.featureService().isEmpty()) {
            FeatureService svc = registry.requireFeatureService(request.project(), request.featureService());
            return registry.resolveOnDemand(svc);
        }
        return List.of();
    }
}
