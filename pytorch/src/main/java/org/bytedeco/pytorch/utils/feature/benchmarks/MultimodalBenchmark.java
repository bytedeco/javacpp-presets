/*
 * Multimodal embedding write/read integrity + bridge + drift/lifecycle benches.
 */
package org.bytedeco.pytorch.utils.feature.benchmarks;

import org.bytedeco.pytorch.utils.feature.FeaturePlatform;
import org.bytedeco.pytorch.utils.feature.bridge.RecommendFeatureBridge;
import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.ValueType;
import org.bytedeco.pytorch.utils.feature.lifecycle.AccessPolicy;
import org.bytedeco.pytorch.utils.feature.lifecycle.FeatureDriftMonitor;
import org.bytedeco.pytorch.utils.feature.lifecycle.FeatureQualityReport;
import org.bytedeco.pytorch.utils.feature.lifecycle.FeatureValidator;
import org.bytedeco.pytorch.utils.feature.lifecycle.FreshnessMonitor;
import org.bytedeco.pytorch.utils.feature.lifecycle.SchemaEvolution;
import org.bytedeco.pytorch.utils.feature.multimodal.EmbeddingFeatureSpec;
import org.bytedeco.pytorch.utils.feature.multimodal.MultimodalFeatureView;
import org.bytedeco.pytorch.utils.feature.serving.FeatureResponse;
import org.bytedeco.pytorch.utils.feature.serving.FeatureVector;
import org.bytedeco.pytorch.utils.recommend.basic.features.Feature;
import org.bytedeco.pytorch.utils.recommend.modelops.FeatureStoreSnapshot;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Multimodal / bridge / lifecycle benchmark cases. */
public final class MultimodalBenchmark {

    private MultimodalBenchmark() {}

    public static void run(BenchCase.Suite suite) {
        embeddingIntegrity(suite);
        bridge(suite);
        driftFreshness(suite);
        schemaLifecycle(suite);
    }

    private static void embeddingIntegrity(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            Entity item = Entity.builder("item_id").joinKey("item_id").build();
            fp.entity(item);
            MultimodalFeatureView mm = MultimodalFeatureView.builder("item_mm")
                    .entities(item)
                    .image("cover", 32)
                    .text("title", 16)
                    .embedding("tower", 32)
                    .online(true)
                    .build();
            FeatureView view = mm.toFeatureView();
            fp.featureView(view);
            fp.featureService(FeatureService.builder("mm_svc").views("item_mm").build());

            long now = System.currentTimeMillis();
            float[] cover = new float[32];
            float[] title = new float[16];
            float[] tower = new float[32];
            double l2 = 0;
            for (int i = 0; i < 32; i++) {
                cover[i] = (float) Math.sin(i * 0.1);
                tower[i] = (float) Math.cos(i * 0.1);
                l2 += cover[i] * cover[i];
            }
            for (int i = 0; i < 16; i++) title[i] = (float) (i * 0.01);
            l2 = Math.sqrt(l2);

            Map<String, Object> row = new LinkedHashMap<>();
            row.put("item_id", 7L);
            row.put("event_timestamp", now);
            row.put("cover_uri", "s3://c/7.jpg");
            row.put("cover_emb", cover);
            row.put("title", "hello");
            row.put("title_emb", title);
            row.put("tower", tower);
            fp.putOffline("default", "item_mm", List.of(row));
            fp.materializeViews(List.of(view));

            FeatureResponse resp = fp.getOnlineFeatures("mm_svc", Map.of("item_id", 7L));
            float[] got = resp.vector().embeddings().get("cover_emb");
            if (got == null) got = resp.vector().embeddings().get("item_mm__cover_emb");
            // also check raw
            if (got == null) {
                Object raw = resp.vector().raw().get("cover_emb");
                if (raw instanceof float[]) got = (float[]) raw;
            }

            EmbeddingFeatureSpec spec = EmbeddingFeatureSpec.of("cover_emb", 32);
            boolean ok = got != null && spec.validate(got) && got.length == 32
                    && Math.abs(spec.l2Norm(got) - l2) < 1e-4;
            if (!ok) {
                suite.add(BenchCase.fail("multimodal_emb",
                        "gotLen=" + (got == null ? -1 : got.length) + " resp=" + resp.vector(),
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("multimodal_emb",
                        "dim=32 l2=" + String.format("%.4f", spec.l2Norm(got)),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("multimodal_emb", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void bridge(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            FeatureView view = FeatureView.builder("bridge_v")
                    .entities(Entity.of("user_id"))
                    .schema(
                            Field.builder("uid").valueType(ValueType.INT64).tag("vocab_size", "1000").tag("embed_dim", "8").build(),
                            Field.of("age", ValueType.FLOAT64),
                            Field.of("hist", ValueType.INT64_LIST),
                            Field.embedding("emb", 4))
                    .build();
            List<Feature> feats = RecommendFeatureBridge.toRecommendFeatures(view);
            if (feats.size() != 4) {
                suite.add(BenchCase.fail("bridge_features", "size=" + feats.size(), System.nanoTime() - t0));
                return;
            }

            FeatureVector.Builder vb = FeatureVector.builder()
                    .entity("user_id", 42L)
                    .sparse("uid", 7L)
                    .dense("age", 25.5)
                    .sequence("hist", new long[]{1, 2, 3})
                    .embedding("emb", new float[]{0.1f, 0.2f, 0.3f, 0.4f})
                    .meta("event_timestamp", String.valueOf(System.currentTimeMillis()));
            FeatureVector vec = vb.build();
            FeatureStoreSnapshot snap = RecommendFeatureBridge.toSnapshot(vec, "snap-1", "v1");
            Map<String, Double> skew = RecommendFeatureBridge.identitySkew(vec);
            boolean ok = snap.denseFeatures().containsKey("age")
                    && snap.sparseFeatures().containsKey("uid")
                    && skew.isEmpty();
            if (!ok) {
                suite.add(BenchCase.fail("bridge_snapshot",
                        "skew=" + skew + " snap=" + snap, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("bridge_snapshot",
                        "feats=" + feats.size() + " skewEmpty=true", System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("bridge_snapshot", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void driftFreshness(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            // baseline ~ N(0,1), current shifted ~ N(3,1) → high PSI
            double[] base = new double[500];
            double[] cur = new double[500];
            for (int i = 0; i < 500; i++) {
                base[i] = Math.sin(i) * 0.5; // tight around 0
                cur[i] = 5.0 + Math.sin(i) * 0.5; // shifted
            }
            FeatureDriftMonitor drift = new FeatureDriftMonitor(10, 0.2);
            FeatureDriftMonitor.PsiResult psi = drift.psi("score", base, cur);
            if (!psi.alert || !(psi.psi > 0.2)) {
                suite.add(BenchCase.fail("drift_psi", psi.toString(), System.nanoTime() - t0));
                return;
            }

            FreshnessMonitor fresh = new FreshnessMonitor();
            fresh.setSlo("default", "v1", Duration.ofMinutes(5));
            long now = System.currentTimeMillis();
            fresh.observe("default", "v1", now - Duration.ofHours(2).toMillis());
            FreshnessMonitor.Status st = fresh.check("default", "v1", now);
            if (!st.alert) {
                suite.add(BenchCase.fail("freshness", st.toString(), System.nanoTime() - t0));
                return;
            }

            FeatureView view = FeatureView.builder("qv")
                    .entities(Entity.of("user_id"))
                    .schema(Field.of("x", ValueType.FLOAT64))
                    .build();
            List<Map<String, Object>> rows = new ArrayList<>();
            for (int i = 0; i < 20; i++) {
                rows.add(Map.of("user_id", (long) i, "event_timestamp", now, "x", (double) i));
            }
            FeatureValidator.Report val = new FeatureValidator().validate(view, rows);
            FeatureQualityReport qr = FeatureQualityReport.builder("v1")
                    .validation(val)
                    .freshness(st)
                    .drift(List.of(psi))
                    .build();
            // healthy should be false due to freshness+drift
            if (qr.healthy()) {
                suite.add(BenchCase.fail("quality_report", "expected unhealthy: " + qr, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("drift_freshness_quality",
                        "psi=" + String.format("%.3f", psi.psi) + " lagMs=" + st.lagMs,
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("drift_freshness_quality", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void schemaLifecycle(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            FeatureView v1 = FeatureView.builder("s")
                    .entities(Entity.of("user_id"))
                    .schema(Field.of("a", ValueType.INT64))
                    .build();
            FeatureView v2add = FeatureView.builder("s")
                    .entities(Entity.of("user_id"))
                    .schema(Field.of("a", ValueType.INT64), Field.of("b", ValueType.FLOAT64))
                    .build();
            FeatureView v2break = FeatureView.builder("s")
                    .entities(Entity.of("user_id"))
                    .schema(Field.of("a", ValueType.FLOAT64))
                    .build();

            SchemaEvolution.Diff add = SchemaEvolution.diff(v1, v2add);
            SchemaEvolution.Diff brk = SchemaEvolution.diff(v1, v2break);
            if (add.breaking() || add.type != SchemaEvolution.ChangeType.ADDITIVE) {
                suite.add(BenchCase.fail("schema_additive", add.toString(), System.nanoTime() - t0));
                return;
            }
            if (!brk.breaking()) {
                suite.add(BenchCase.fail("schema_breaking", brk.toString(), System.nanoTime() - t0));
                return;
            }
            boolean threw = false;
            try {
                SchemaEvolution.requireCompatible(v1, v2break);
            } catch (IllegalStateException e) {
                threw = true;
            }
            AccessPolicy acl = new AccessPolicy();
            acl.grant("p1", "alice", AccessPolicy.Role.READ);
            acl.grant("p1", "bob", AccessPolicy.Role.ADMIN);
            boolean aliceWriteDenied = !acl.can("p1", "alice", AccessPolicy.Role.WRITE);
            boolean bobOk = acl.can("p1", "bob", AccessPolicy.Role.WRITE);
            if (!threw || !aliceWriteDenied || !bobOk) {
                suite.add(BenchCase.fail("schema_acl",
                        "threw=" + threw + " aliceWriteDenied=" + aliceWriteDenied + " bobOk=" + bobOk,
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("schema_lifecycle_acl",
                        "additive+breaking+acl ok", System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("schema_lifecycle_acl", e.toString(), System.nanoTime() - t0));
        }
    }
}
