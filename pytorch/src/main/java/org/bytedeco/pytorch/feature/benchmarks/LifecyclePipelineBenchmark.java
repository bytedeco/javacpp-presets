/*
 * Full-lifecycle pipeline benchmark:
 *   raw DataFrame → feature engineering → FeatureIngest → materialize
 *   → online serve → PIT training export → DeepFM train steps → quality
 *
 * Also covers multi-angle stage assertions (each lifecycle stage must PASS).
 */
package org.bytedeco.pytorch.feature.benchmarks;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.feature.FeaturePlatform;
import org.bytedeco.pytorch.feature.pipeline.LifecyclePipeline;
import org.bytedeco.pytorch.feature.store.StoreConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** End-to-end feature + model lifecycle bench. */
public final class LifecyclePipelineBenchmark {

    private LifecyclePipelineBenchmark() {}

    public static void run(BenchCase.Suite suite) {
        lifecycleMemory(suite);
        lifecycleWithTrain(suite);
        lifecycleSqliteBackend(suite);
        lifecycleStageCoverage(suite);
    }

    private static void lifecycleMemory(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame raw = DataFrameFeatureStoreBenchmark.syntheticUsers(64, true);
            LifecyclePipeline.Result r = LifecyclePipeline.on(fp)
                    .project("life")
                    .view("user_life")
                    .featureService("life_svc")
                    .entities("user_id")
                    .raw(raw)
                    .featureEngineering(fe -> fe
                            .impute("mean", "age")
                            .standardScale("age", "score")
                            .build())
                    .labelColumn("label")
                    .trainDeepFM(false)
                    .run();
            boolean ok = r.success
                    && r.ingest != null && r.ingest.ok()
                    && r.materialize != null && r.materialize.success()
                    && r.trainingExport != null && r.trainingExport.size() > 0
                    && r.onlineSmoke != null && r.onlineSmoke.success();
            if (!ok) {
                suite.add(BenchCase.fail("lifecycle_memory", r.toString(), System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("lifecycle_memory",
                        "stages=" + r.stages.size()
                                + " exportN=" + r.trainingExport.size()
                                + " totalMs=" + String.format("%.1f", r.totalMs()),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("lifecycle_memory", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void lifecycleWithTrain(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame raw = DataFrameFeatureStoreBenchmark.syntheticUsers(128, false);
            int steps = intProp("feature.bench.lifecycle_steps", 12);
            LifecyclePipeline.Result r = LifecyclePipeline.on(fp)
                    .project("train")
                    .view("user_train")
                    .featureService("train_svc")
                    .entities("user_id")
                    .raw(raw)
                    .featureEngineering(fe -> fe
                            .standardScale("age", "score")
                            .build())
                    .labelColumn("label")
                    .trainDeepFM(true)
                    .trainSteps(steps)
                    .batchSize(32)
                    .embedDim(8)
                    .learningRate(1e-2f)
                    .run();

            boolean trainOk = r.finalTrainLoss != null && Double.isFinite(r.finalTrainLoss);
            boolean ok = r.success && trainOk && r.recommendFeatures.size() > 0;
            // Allow train stage soft-fail only if loss is finite from a partial path —
            // require explicit train stage pass
            boolean trainStagePass = false;
            for (LifecyclePipeline.StageTiming s : r.stages) {
                if ("train_deepfm".equals(s.stage) && s.ok) trainStagePass = true;
            }
            if (!ok || !trainStagePass) {
                suite.add(BenchCase.fail("lifecycle_train_deepfm",
                        r.toString(), System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("lifecycle_train_deepfm",
                        "steps=" + steps
                                + " loss=" + String.format("%.4f", r.finalTrainLoss)
                                + " feats=" + r.recommendFeatures.size()
                                + " exportN=" + r.trainingExport.size(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("lifecycle_train_deepfm", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void lifecycleSqliteBackend(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            Path tmp = Files.createTempDirectory("life-sqlite");
            StoreConfig cfg = StoreConfig.builder()
                    .root(tmp)
                    .online("sqlite")
                    .offline("sqlite")
                    .embedding("memory")
                    .sqliteOnlinePath(tmp.resolve("online.db").toString())
                    .sqliteOfflinePath(tmp.resolve("offline.db").toString())
                    .build();
            try (FeaturePlatform fp = FeaturePlatform.fromConfig(cfg)) {
                DataFrame raw = DataFrameFeatureStoreBenchmark.syntheticUsers(48, true);
                LifecyclePipeline.Result r = LifecyclePipeline.on(fp)
                        .project("sqlife")
                        .view("user_sql")
                        .featureService("sql_svc")
                        .entities("user_id")
                        .raw(raw)
                        .featureEngineering(fe -> fe
                                .impute("mean", "age")
                                .standardScale("age", "score")
                                .build())
                        .trainDeepFM(false)
                        .run();
                boolean ok = r.success
                        && fp.storeConfig().onlineBackend().name().equals("SQLITE")
                        && r.trainingExport != null
                        && r.trainingExport.size() > 0;
                if (!ok) {
                    suite.add(BenchCase.fail("lifecycle_sqlite",
                            r.toString() + " cfg=" + fp.storeConfig(),
                            System.nanoTime() - t0));
                } else {
                    suite.add(BenchCase.pass("lifecycle_sqlite",
                            "sqlite online+offline exportN=" + r.trainingExport.size()
                                    + " root=" + tmp,
                            System.nanoTime() - t0));
                }
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("lifecycle_sqlite", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void lifecycleStageCoverage(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame raw = DataFrameFeatureStoreBenchmark.syntheticUsers(32, false);
            LifecyclePipeline.Result r = LifecyclePipeline.on(fp)
                    .project("cov")
                    .view("user_cov")
                    .featureService("cov_svc")
                    .entities("user_id")
                    .raw(raw)
                    .featureEngineering(fe -> fe.standardScale("score").build())
                    .trainDeepFM(true)
                    .trainSteps(6)
                    .run();

            List<String> required = List.of(
                    "feature_eng", "ingest", "register_svc", "materialize",
                    "online_serve", "train_export", "train_deepfm", "quality");
            List<String> missing = new ArrayList<>();
            for (String req : required) {
                boolean found = false;
                for (LifecyclePipeline.StageTiming s : r.stages) {
                    if (req.equals(s.stage)) {
                        found = true;
                        if (!s.ok) missing.add(req + "(fail)");
                        break;
                    }
                }
                if (!found) missing.add(req + "(absent)");
            }
            if (!missing.isEmpty()) {
                suite.add(BenchCase.fail("lifecycle_stage_coverage",
                        "missing/fail=" + missing + "\n" + r,
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("lifecycle_stage_coverage",
                        "all " + required.size() + " stages PASS, totalMs="
                                + String.format("%.1f", r.totalMs()),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("lifecycle_stage_coverage", e.toString(), System.nanoTime() - t0));
        }
    }

    private static int intProp(String key, int dflt) {
        String v = System.getProperty(key);
        if (v == null || v.isBlank()) return dflt;
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            return dflt;
        }
    }
}
