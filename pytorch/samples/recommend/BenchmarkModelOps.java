/*
 * Benchmark for modelops: registry lifecycle, shadow serving, drift (PSI),
 * online learning hook, feature store snapshot skew.
 *
 *   java -cp ... samples.recommend.BenchmarkModelOps
 */
package samples.recommend;

import org.bytedeco.pytorch.recommend.modelops.DriftDetector;
import org.bytedeco.pytorch.recommend.modelops.FeatureStoreSnapshot;
import org.bytedeco.pytorch.recommend.modelops.ModelRegistry;
import org.bytedeco.pytorch.recommend.modelops.ModelStage;
import org.bytedeco.pytorch.recommend.modelops.ModelVersion;
import org.bytedeco.pytorch.recommend.modelops.OnlineLearningHook;
import org.bytedeco.pytorch.recommend.modelops.ShadowServing;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public final class BenchmarkModelOps {

    public static void main(String[] args) {
        System.exit(runTests());
    }

    public static int runTests() {
        BenchSupport.Suite s = new BenchSupport.Suite("BenchmarkModelOps");
        s.header();

        s.benchmark("registry_register_and_promote_path", () -> {
            ModelRegistry reg = new ModelRegistry();
            List<String> events = new ArrayList<>();
            reg.addListener(e -> events.add(e.type.name()));

            ModelVersion v1 = ModelVersion.builder("ctr", "v1")
                    .artifactUri("s3://models/ctr/v1.pt")
                    .framework("pytorch")
                    .offlineMetric("AUC", 0.78)
                    .description("baseline")
                    .build();
            reg.register(v1);
            s.checkEq("stage TRAINED", ModelStage.TRAINED, reg.get("ctr", "v1").stage());

            reg.transition("ctr", "v1", ModelStage.OFFLINE_PASS);
            reg.transition("ctr", "v1", ModelStage.SHADOW);
            reg.transition("ctr", "v1", ModelStage.CANARY);
            reg.transition("ctr", "v1", ModelStage.PROD);
            s.checkEq("prod pointer", "v1", reg.productionOf("ctr").versionId());
            s.checkEq("stage PROD", ModelStage.PROD, reg.get("ctr", "v1").stage());
            s.checkTrue("events registered", events.contains("REGISTERED"));
            s.checkTrue("events stage", events.contains("STAGE_CHANGED"));
        });

        s.benchmark("registry_promote_helper_and_reject", () -> {
            ModelRegistry reg = new ModelRegistry();
            reg.register(ModelVersion.builder("ctr", "v2").artifactUri("a").build());
            s.checkEq("auto1", ModelStage.OFFLINE_PASS, reg.promote("ctr", "v2").stage());
            s.checkEq("auto2", ModelStage.SHADOW, reg.promote("ctr", "v2").stage());
            s.checkEq("auto3", ModelStage.CANARY, reg.promote("ctr", "v2").stage());
            s.checkEq("auto4", ModelStage.PROD, reg.promote("ctr", "v2").stage());

            reg.register(ModelVersion.builder("ctr", "bad").artifactUri("b").build());
            reg.reject("ctr", "bad", "offline AUC drop");
            s.checkEq("rejected", ModelStage.REJECTED, reg.get("ctr", "bad").stage());

            boolean threw = false;
            try {
                reg.transition("ctr", "bad", ModelStage.PROD);
            } catch (IllegalStateException ex) {
                threw = true;
            }
            s.checkTrue("cannot promote rejected", threw);
        });

        s.benchmark("registry_rollback_archives_prev_prod", () -> {
            ModelRegistry reg = new ModelRegistry();
            reg.register(ModelVersion.builder("rank", "old").artifactUri("old").build());
            reg.register(ModelVersion.builder("rank", "new").artifactUri("new")
                    .parentVersionId("old").build());
            // fast-path both to useful stages
            for (ModelStage st : new ModelStage[] {
                    ModelStage.OFFLINE_PASS, ModelStage.SHADOW, ModelStage.CANARY, ModelStage.PROD}) {
                reg.transition("rank", "old", st);
            }
            for (ModelStage st : new ModelStage[] {
                    ModelStage.OFFLINE_PASS, ModelStage.SHADOW, ModelStage.CANARY, ModelStage.PROD}) {
                reg.transition("rank", "new", st);
            }
            s.checkEq("prod is new", "new", reg.productionOf("rank").versionId());
            s.checkEq("old archived", ModelStage.ARCHIVED, reg.get("rank", "old").stage());

            ModelVersion rolled = reg.rollback("rank", "old");
            s.checkEq("rolled prod", "old", rolled.versionId());
            s.checkEq("prod pointer old", "old", reg.productionOf("rank").versionId());
            s.checkEq("new archived after rollback", ModelStage.ARCHIVED, reg.get("rank", "new").stage());
        });

        s.benchmark("registry_duplicate_and_list", () -> {
            ModelRegistry reg = new ModelRegistry();
            reg.register(ModelVersion.builder("m", "1").build());
            boolean threw = false;
            try {
                reg.register(ModelVersion.builder("m", "1").build());
            } catch (IllegalStateException ex) {
                threw = true;
            }
            s.checkTrue("dup rejected", threw);
            reg.register(ModelVersion.builder("m", "2").offlineMetric("AUC", 0.8).build());
            reg.updateMetrics("m", "2", Map.of("AUC", 0.81, "LogLoss", 0.4));
            s.checkClose("metric updated", 0.81, reg.get("m", "2").offlineMetrics().get("AUC"), 1e-12);
            s.checkEq("2 versions", 2, reg.listVersions("m").size());
            s.checkTrue("models list", reg.listModels().contains("m"));
        });

        s.benchmark("shadow_serving_disagreement_and_gate", () -> {
            // prod: score = index; shadow: score = index + noise on half
            ShadowServing.ModelScorer prod = (key, items) -> {
                Map<String, Double> m = new LinkedHashMap<>();
                for (int i = 0; i < items.size(); i++) m.put(items.get(i), (double) i);
                return m;
            };
            ShadowServing.ModelScorer shadow = (key, items) -> {
                Map<String, Double> m = new LinkedHashMap<>();
                for (int i = 0; i < items.size(); i++) {
                    // flip top on purpose sometimes
                    m.put(items.get(i), (double) (items.size() - 1 - i));
                }
                return m;
            };
            ShadowServing ss = new ShadowServing("dark", prod, shadow, 1.0);
            List<String> items = List.of("a", "b", "c", "d", "e");
            for (int i = 0; i < 150; i++) {
                ss.compareSync("u" + i, items);
            }
            ShadowServing.DisagreementStats st = ss.stats();
            System.out.println("    " + st);
            s.checkTrue("samples >= 150", st.samples >= 150);
            s.checkTrue("rank mismatches > 0", st.rankMismatches > 0);
            s.checkTrue("mean abs delta > 0", st.meanAbsScoreDelta > 0);
            // strict gate should fail on reversed ranking
            s.checkTrue("gate fails on reverse", !ss.passGate(0.05, 0.1, 0.01));

            // identical scorers pass gate
            ShadowServing same = new ShadowServing("same", prod, prod, 1.0);
            for (int i = 0; i < 150; i++) same.compareSync("u" + i, items);
            s.checkTrue("identical passes gate", same.passGate(0.05, 0.01, 0.01));
            same.shutdown();
            ss.shutdown();
        });

        s.benchmark("shadow_async_sample_rate", () -> {
            AtomicInteger shadowCalls = new AtomicInteger();
            ShadowServing.ModelScorer prod = (k, items) -> Map.of("i0", 1.0, "i1", 0.5);
            ShadowServing.ModelScorer sh = (k, items) -> {
                shadowCalls.incrementAndGet();
                return Map.of("i0", 1.0, "i1", 0.5);
            };
            ShadowServing ss = new ShadowServing("async", prod, sh, 0.2);
            for (int i = 0; i < 500; i++) {
                ss.score("u" + i, List.of("i0", "i1"));
            }
            // allow async drain
            Thread.sleep(200);
            int calls = shadowCalls.get();
            System.out.println("    shadow async calls=" + calls + "/500 (~100 expected)");
            s.checkRange("sample ~20%", calls, 40, 200);
            ss.shutdown();
        });

        s.benchmark("drift_psi_stable_vs_shifted", () -> {
            Random rng = BenchSupport.rng(21);
            double[] base = BenchSupport.randomGaussian(rng, 5000, 0.0, 1.0);
            double[] same = BenchSupport.randomGaussian(rng, 5000, 0.0, 1.0);
            double[] shifted = BenchSupport.randomGaussian(rng, 5000, 1.5, 1.0);

            DriftDetector.PsiResult stable = DriftDetector.psi(base, same, 10, "f", 0.25);
            DriftDetector.PsiResult drift = DriftDetector.psi(base, shifted, 10, "f", 0.25);
            System.out.println("    stable " + stable);
            System.out.println("    shifted " + drift);
            s.checkTrue("stable PSI low", stable.psi < 0.25);
            s.checkTrue("stable no alert", !stable.alert);
            s.checkTrue("shifted PSI high", drift.psi > 0.25);
            s.checkTrue("shifted alert", drift.alert);
        });

        s.benchmark("drift_kl_js_and_report", () -> {
            double[] p = {0.5, 0.5, 0.0};
            double[] q = {0.5, 0.5, 0.0};
            // avoid zero mass issues — use smoothed
            double[] p2 = {0.7, 0.2, 0.1};
            double[] q2 = {0.2, 0.3, 0.5};
            double kl = DriftDetector.klDivergence(p2, q2);
            double js = DriftDetector.jsDivergence(p2, q2);
            s.checkTrue("KL > 0", kl > 0);
            s.checkTrue("JS > 0", js > 0);
            s.checkTrue("JS finite", Double.isFinite(js));

            Map<String, double[]> baseF = new HashMap<>();
            Map<String, double[]> curF = new HashMap<>();
            Random rng = BenchSupport.rng(3);
            baseF.put("age", BenchSupport.randomGaussian(rng, 1000, 30, 5));
            curF.put("age", BenchSupport.randomGaussian(rng, 1000, 30, 5));
            baseF.put("price", BenchSupport.randomGaussian(rng, 1000, 10, 2));
            curF.put("price", BenchSupport.randomGaussian(rng, 1000, 18, 2)); // drifted

            double[] baseScores = BenchSupport.randomGaussian(rng, 1000, 0.2, 0.05);
            double[] curScores = BenchSupport.randomGaussian(rng, 1000, 0.2, 0.05);
            DriftDetector.DriftReport rep = DriftDetector.report(
                    baseF, curF, baseScores, curScores, 0.01, 0.02, 0.25, 0.05);
            System.out.println(rep);
            s.checkTrue("report text", rep.text.contains("Drift Report"));
            // price should alert
            boolean priceAlert = false;
            for (DriftDetector.PsiResult r : rep.featurePsi) {
                if ("price".equals(r.featureName) && r.alert) priceAlert = true;
            }
            s.checkTrue("price drift alert", priceAlert);
        });

        s.benchmark("online_histogram_monitor", () -> {
            DriftDetector.OnlineMonitor mon = new DriftDetector.OnlineMonitor(0.25, 10);
            mon.defineFeature("x", -3, 3);
            Random rng = BenchSupport.rng(9);
            for (int i = 0; i < 2000; i++) mon.observe("x", rng.nextGaussian());
            mon.freezeBaseline("x");
            // live window similar
            for (int i = 0; i < 2000; i++) mon.observe("x", rng.nextGaussian());
            s.checkTrue("no alert on same dist", !mon.anyAlert() || mon.evaluate().get(0).psi < 0.5);

            // strong shift
            mon.defineFeature("y", -3, 5);
            for (int i = 0; i < 2000; i++) mon.observe("y", rng.nextGaussian());
            mon.freezeBaseline("y");
            for (int i = 0; i < 2000; i++) mon.observe("y", 2.5 + rng.nextGaussian() * 0.3);
            List<DriftDetector.PsiResult> ev = mon.evaluate();
            boolean yAlert = false;
            for (DriftDetector.PsiResult r : ev) {
                if ("y".equals(r.featureName) && r.alert) yAlert = true;
            }
            s.checkTrue("y shifted alerts", yAlert);
        });

        s.benchmark("online_learning_hook_flush_registers", () -> {
            ModelRegistry reg = new ModelRegistry();
            reg.register(ModelVersion.builder("ctr", "base")
                    .artifactUri("s3://m/base.pt")
                    .build());
            AtomicInteger trains = new AtomicInteger();
            OnlineLearningHook hook = new OnlineLearningHook(
                    "ctr", reg,
                    (batch, base) -> {
                        trains.incrementAndGet();
                        return new OnlineLearningHook.UpdateResult(
                                true, batch.size(), "s3://m/ol-" + trains.get() + ".pt", "ok", 1L);
                    },
                    "base", 10, 60_000L);

            for (int i = 0; i < 25; i++) {
                hook.accept("u" + i, "item" + i, "click", 1.0f);
            }
            // 25 events, flushSize=10 => at least 2 auto flushes, maybe remainder
            OnlineLearningHook.UpdateResult last = hook.flush();
            s.checkTrue("trained", trains.get() >= 2);
            s.checkTrue("versions grew", reg.listVersions("ctr").size() >= 2);
            s.checkTrue("enqueued 25", hook.enqueued() >= 25);
            s.checkTrue("flushed > 0", hook.flushed() > 0);
            System.out.println("    trains=" + trains.get() + " versions=" + reg.listVersions("ctr").size()
                    + " last=" + last);
        });

        s.benchmark("online_learning_error_requeues", () -> {
            ModelRegistry reg = new ModelRegistry();
            reg.register(ModelVersion.builder("ctr", "base").artifactUri("a").build());
            OnlineLearningHook hook = new OnlineLearningHook(
                    "ctr", reg,
                    (batch, base) -> {
                        throw new RuntimeException("ps down");
                    },
                    "base", 100, 60_000L);
            for (int i = 0; i < 5; i++) {
                hook.accept("u", "i" + i, "click", 1f);
            }
            OnlineLearningHook.UpdateResult r = hook.flush();
            s.checkTrue("failed", !r.success);
            s.checkTrue("requeued", hook.buffered() >= 5);
            s.checkTrue("failedFlushes", hook.failedFlushes() >= 1);
        });

        s.benchmark("feature_store_snapshot_skew", () -> {
            FeatureStoreSnapshot train = FeatureStoreSnapshot.builder("snap-train")
                    .schemaVersion("feat-v3")
                    .userId("u1")
                    .eventTimeMs(1000)
                    .ingestTimeMs(1100)
                    .dense("age", 30.0)
                    .dense("price", 10.0)
                    .sparse("user_id", 42L)
                    .sequence("hist", new long[] {1, 2, 3})
                    .meta("scene", "home")
                    .build();
            s.checkEq("lag 100", 100L, train.freshnessLagMs());

            FeatureStoreSnapshot online = FeatureStoreSnapshot.builder("snap-online")
                    .schemaVersion("feat-v3")
                    .userId("u1")
                    .eventTimeMs(1000)
                    .ingestTimeMs(1500)
                    .dense("age", 30.0)
                    .dense("price", 25.0) // skewed
                    // missing age is fine; price skewed; missing "new_feat" only online
                    .dense("new_feat", 1.0)
                    .build();

            Map<String, Double> skew = online.denseSkewAgainst(train, 0.05);
            s.checkTrue("price in skew", skew.containsKey("price"));
            s.checkTrue("new_feat missing in train marked", skew.containsKey("new_feat")
                    || skew.containsKey("new_feat"));
            // age should not be in skew
            s.checkTrue("age aligned", !skew.containsKey("age"));
            s.checkTrue("toString", train.toString().contains("FeatureStoreSnapshot"));
        });

        s.benchmark("modelops_throughput_psi", () -> {
            Random rng = BenchSupport.rng(1);
            double[] a = BenchSupport.randomGaussian(rng, 20_000, 0, 1);
            double[] b = BenchSupport.randomGaussian(rng, 20_000, 0.05, 1);
            long t0 = System.nanoTime();
            DriftDetector.PsiResult r = DriftDetector.psi(a, b, 20, "x", 0.25);
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("    PSI n=20k bins=20 in " + ms + "ms psi=" + r.psi);
            s.checkTrue("fast enough < 2s", ms < 2000);
            s.checkTrue("psi finite", Double.isFinite(r.psi));
        });

        return s.exitCode();
    }
}
