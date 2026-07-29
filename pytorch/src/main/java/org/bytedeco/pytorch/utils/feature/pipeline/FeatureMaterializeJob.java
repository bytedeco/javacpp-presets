/*
 * Feature materialize job — offline warehouse → online store, driven from
 * FeatureIngest results or explicit FeatureViews / FeatureService.
 *
 * Completes the FE → ingest → materialize chain so ranking services can
 * getOnlineFeatures immediately after DataFrame feature engineering.
 */
package org.bytedeco.pytorch.utils.feature.pipeline;

import org.bytedeco.pytorch.utils.feature.FeaturePlatform;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.lifecycle.FreshnessMonitor;
import org.bytedeco.pytorch.utils.feature.materialize.MaterializationJob;
import org.bytedeco.pytorch.utils.feature.materialize.MaterializationResult;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Fluent materialization after DataFrame ingest.
 *
 * <pre>{@code
 * MaterializationResult r = FeatureMaterializeJob.on(fp)
 *     .fromIngest(ingestResult)
 *     .incremental(false)
 *     .observeFreshness(freshnessMonitor)
 *     .run();
 * }</pre>
 */
public final class FeatureMaterializeJob {

    private final FeaturePlatform platform;
    private final List<FeatureView> views = new ArrayList<>();
    private String project = "default";
    private Instant start = Instant.EPOCH;
    private Instant end;
    private boolean incremental;
    private FreshnessMonitor freshnessMonitor;

    private FeatureMaterializeJob(FeaturePlatform platform) {
        this.platform = Objects.requireNonNull(platform, "platform");
    }

    public static FeatureMaterializeJob on(FeaturePlatform platform) {
        return new FeatureMaterializeJob(platform);
    }

    public FeatureMaterializeJob project(String project) {
        this.project = project != null ? project : "default";
        return this;
    }

    public FeatureMaterializeJob views(FeatureView... vs) {
        if (vs != null) {
            for (FeatureView v : vs) {
                if (v != null) views.add(v);
            }
        }
        return this;
    }

    public FeatureMaterializeJob views(List<FeatureView> vs) {
        if (vs != null) views.addAll(vs);
        return this;
    }

    public FeatureMaterializeJob viewNames(String... names) {
        if (names != null) {
            for (String n : names) {
                platform.registry().getFeatureView(project, n).ifPresent(views::add);
                platform.registry().getStreamFeatureView(project, n)
                        .ifPresent(sfv -> views.add(sfv.asBatchView()));
            }
        }
        return this;
    }

    public FeatureMaterializeJob featureService(String serviceName) {
        FeatureService svc = platform.registry().requireFeatureService(project, serviceName);
        views.addAll(platform.registry().resolveViews(svc));
        return this;
    }

    public FeatureMaterializeJob fromIngest(FeatureIngest.Result ingest) {
        Objects.requireNonNull(ingest, "ingest");
        this.project = ingest.project;
        if (ingest.view != null) {
            views.add(ingest.view);
        } else {
            platform.registry().getFeatureView(ingest.project, ingest.viewName).ifPresent(views::add);
        }
        return this;
    }

    public FeatureMaterializeJob fromIngest(List<FeatureIngest.Result> ingests) {
        if (ingests != null) {
            for (FeatureIngest.Result r : ingests) fromIngest(r);
        }
        return this;
    }

    public FeatureMaterializeJob start(Instant start) {
        this.start = start != null ? start : Instant.EPOCH;
        return this;
    }

    public FeatureMaterializeJob end(Instant end) {
        this.end = end;
        return this;
    }

    public FeatureMaterializeJob incremental(boolean incremental) {
        this.incremental = incremental;
        return this;
    }

    public FeatureMaterializeJob observeFreshness(FreshnessMonitor monitor) {
        this.freshnessMonitor = monitor;
        return this;
    }

    public MaterializationResult run() {
        if (views.isEmpty()) {
            // materialize all online views in project
            return platform.materialize().materializeProject(
                    project, start, end != null ? end : Instant.now());
        }
        Instant e = end != null ? end : Instant.now();
        MaterializationJob job = MaterializationJob.builder()
                .project(project)
                .views(views)
                .start(start)
                .end(e)
                .incremental(incremental)
                .build();
        MaterializationResult result = platform.materialize().materialize(job);

        if (freshnessMonitor != null && result.success()) {
            long wm = result.watermarkMs() > 0 ? result.watermarkMs() : System.currentTimeMillis();
            for (FeatureView v : views) {
                freshnessMonitor.observe(v.project(), v.name(), wm);
            }
        }
        return result;
    }
}
