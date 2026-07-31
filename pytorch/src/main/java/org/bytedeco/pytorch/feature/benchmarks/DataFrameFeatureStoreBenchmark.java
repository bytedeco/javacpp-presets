/*
 * Multi-angle DataFrame ↔ Feature Store integration benchmark.
 *
 * Dimensions:
 *   1) bridge_roundtrip     — rows ↔ DataFrame ↔ rows schema/value fidelity
 *   2) select_project       — selectColumns / ensureEventTimestamp / dense matrix
 *   3) fe_then_ingest       — df.feature() impute+scale → FeatureIngest auto-register
 *   4) ingest_materialize   — offline put → materialize → online get consistency
 *   5) training_export_pit  — entity df + service → PIT TrainingDataset / DataFrame
 *   6) batch_build          — TrainingExport → recommend Batch tensors
 *   7) dual_view_service    — two views ingested, one FeatureService, online+historical
 *   8) replace_vs_append    — replace semantics clear prior offline rows
 */
package org.bytedeco.pytorch.feature.benchmarks;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.feature.FeaturePlatform;
import org.bytedeco.pytorch.feature.bridge.DataFrameBridge;
import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.materialize.MaterializationResult;
import org.bytedeco.pytorch.feature.pipeline.FeatureIngest;
import org.bytedeco.pytorch.feature.pipeline.FeatureMaterializeJob;
import org.bytedeco.pytorch.feature.pipeline.FeatureTrainingExport;
import org.bytedeco.pytorch.feature.serving.FeatureRequest;
import org.bytedeco.pytorch.feature.serving.FeatureResponse;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.Features;
import org.bytedeco.pytorch.recommend.data.Batch;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** DataFrame feature-engineering ↔ feature-store integration suite. */
public final class DataFrameFeatureStoreBenchmark {

    private DataFrameFeatureStoreBenchmark() {}

    public static void run(BenchCase.Suite suite) {
        bridgeRoundtrip(suite);
        selectAndMatrix(suite);
        feThenIngest(suite);
        ingestMaterializeOnline(suite);
        trainingExportPit(suite);
        batchBuild(suite);
        dualViewService(suite);
        replaceVsAppend(suite);
    }

    private static void bridgeRoundtrip(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            List<Map<String, Object>> rows = new ArrayList<>();
            for (int i = 0; i < 20; i++) {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("user_id", (long) i);
                r.put("score", i * 0.1);
                r.put("flag", i % 2 == 0);
                r.put("hist", new long[]{i, i + 1});
                rows.add(r);
            }
            DataFrame df = DataFrameBridge.fromRows(rows);
            List<Map<String, Object>> back = DataFrameBridge.toRows(df);
            boolean ok = df.rowCount() == 20
                    && back.size() == 20
                    && Long.valueOf(5L).equals(toLong(back.get(5).get("user_id")))
                    && Math.abs(toDouble(back.get(5).get("score")) - 0.5) < 1e-9
                    && DataFrameBridge.inferSchema(df).containsKey("user_id");
            if (!ok) {
                suite.add(BenchCase.fail("df_bridge_roundtrip",
                        "rows=" + back.size() + " schema=" + DataFrameBridge.inferSchema(df),
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_bridge_roundtrip",
                        "n=20 schema=" + DataFrameBridge.inferSchema(df).size(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_bridge_roundtrip", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void selectAndMatrix(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try {
            DataFrame df = DataFrame.create();
            df.addColumn("user_id", Column.DType.INT64);
            df.addColumn("age", Column.DType.FLOAT64);
            df.addColumn("noise", Column.DType.STRING);
            for (int i = 0; i < 10; i++) {
                int idx = df.addRow();
                df.set(idx, "user_id", (long) i);
                df.set(idx, "age", 20.0 + i);
                df.set(idx, "noise", "x" + i);
            }
            DataFrameBridge.ensureEventTimestamp(df, "event_timestamp", 1_700_000_000_000L);
            DataFrame proj = DataFrameBridge.selectColumns(df, List.of("user_id", "age", "event_timestamp"));
            double[][] m = DataFrameBridge.toDenseMatrix(proj, List.of("age"));
            boolean ok = proj.columnCount() == 3
                    && proj.hasColumn("event_timestamp")
                    && !proj.hasColumn("noise")
                    && m.length == 10
                    && Math.abs(m[3][0] - 23.0) < 1e-9;
            if (!ok) {
                suite.add(BenchCase.fail("df_select_matrix",
                        "cols=" + proj.getColumnNames() + " m00=" + (m.length > 0 ? m[0][0] : -1),
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_select_matrix",
                        "projected cols=3 dense[3][0]=23", System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_select_matrix", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void feThenIngest(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame raw = syntheticUsers(50, /*withNullAge*/ true);
            FeatureIngest.Result r = FeatureIngest.into(fp)
                    .project("demo")
                    .view("user_fe")
                    .entities("user_id")
                    .timestampColumn("event_timestamp")
                    .ttlDays(7)
                    .autoRegister(true)
                    .validate(true)
                    .exclude("label")
                    .featureEngineering(fe -> fe
                            .impute("mean", "age")
                            .standardScale("age", "score")
                            .build())
                    .from(raw)
                    .run();
            boolean noLabel = !r.featureColumns.contains("label");
            boolean ok = r.ok()
                    && r.registered
                    && r.rowsWritten == 50
                    && noLabel
                    && fp.registry().getFeatureView("demo", "user_fe").isPresent()
                    && fp.offline().rowCount("demo", "user_fe") == 50;
            if (!ok) {
                suite.add(BenchCase.fail("df_fe_ingest", r.toString(), System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_fe_ingest",
                        "registered view features=" + r.featureColumns + " written=" + r.rowsWritten
                                + " excludeLabel=" + noLabel,
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_fe_ingest", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void ingestMaterializeOnline(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame raw = syntheticUsers(30, false);
            FeatureIngest.Result ing = FeatureIngest.into(fp)
                    .project("demo")
                    .view("u_online")
                    .entities("user_id")
                    .features("age", "score", "city")
                    .from(raw)
                    .run();
            fp.featureService(FeatureService.builder("svc_online")
                    .project("demo").view("u_online").build());
            MaterializationResult mat = FeatureMaterializeJob.on(fp)
                    .fromIngest(ing)
                    .run();
            FeatureResponse resp = fp.getOnlineFeatures(FeatureRequest.builder()
                    .project("demo")
                    .featureService("svc_online")
                    .entity("user_id", 1L)
                    .build());
            Object age = resp.vector().raw().get("age");
            if (age == null) age = resp.vector().raw().get("u_online__age");
            boolean ok = ing.ok() && mat.success() && mat.rowsWritten() >= 1
                    && resp.success() && age instanceof Number;
            if (!ok) {
                suite.add(BenchCase.fail("df_ingest_materialize_online",
                        "ing=" + ing + " mat=" + mat + " resp=" + resp + " age=" + age,
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_ingest_materialize_online",
                        "written=" + mat.rowsWritten() + " onlineAge=" + age,
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_ingest_materialize_online", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void trainingExportPit(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            // Feature rows at t=1000 and t=2000; entity events at t=1500 must see only t=1000
            long tPast = 1_000L;
            long tFuture = 2_000L;
            long tEvent = 1_500L;
            List<Map<String, Object>> featRows = new ArrayList<>();
            for (int u = 1; u <= 10; u++) {
                Map<String, Object> past = new LinkedHashMap<>();
                past.put("user_id", (long) u);
                past.put("event_timestamp", tPast);
                past.put("f", 10.0 + u);
                featRows.add(past);
                Map<String, Object> fut = new LinkedHashMap<>();
                fut.put("user_id", (long) u);
                fut.put("event_timestamp", tFuture);
                fut.put("f", 999.0); // must not leak
                featRows.add(fut);
            }
            FeatureIngest.Result ing = FeatureIngest.into(fp)
                    .project("pit")
                    .view("uf")
                    .entities("user_id")
                    .features("f")
                    .fromRows(featRows)
                    .run();
            fp.featureService(FeatureService.builder("pit_svc").project("pit").view("uf").build());

            DataFrame entity = DataFrame.create();
            entity.addColumn("user_id", Column.DType.INT64);
            entity.addColumn("event_timestamp", Column.DType.INT64);
            entity.addColumn("label", Column.DType.FLOAT64);
            for (int u = 1; u <= 10; u++) {
                int idx = entity.addRow();
                entity.set(idx, "user_id", (long) u);
                entity.set(idx, "event_timestamp", tEvent);
                entity.set(idx, "label", u % 2 == 0 ? 1.0 : 0.0);
            }

            FeatureTrainingExport.Result exp = FeatureTrainingExport.from(fp)
                    .project("pit")
                    .featureService("pit_svc")
                    .entityDataFrame(entity)
                    .labelColumn("label")
                    .run();

            boolean noLeak = true;
            for (Map<String, Object> row : exp.dataset.rows()) {
                Object f = row.get("uf__f");
                if (f == null) f = row.get("f");
                double v = toDouble(f);
                if (v >= 900) {
                    noLeak = false;
                    break;
                }
            }
            // Offline readRange is bounded by entity event_ts, so future rows may never
            // enter the join (futureRowsRejected can be 0). Correctness = no leak
            // + as-of value from the past row (user 1 → f=11.0 at t=1000).
            Object sampleF = exp.size() > 0 ? exp.dataset.row(0).get("uf__f") : null;
            if (sampleF == null && exp.size() > 0) sampleF = exp.dataset.row(0).get("f");
            boolean valueOk = Math.abs(toDouble(sampleF) - 11.0) < 1e-9;
            boolean ok = ing.ok() && exp.size() == 10 && noLeak && valueOk
                    && exp.dataFrame.rowCount() == 10
                    && exp.joinStats != null
                    && exp.joinStats.joinsHit == 10;
            if (!ok) {
                suite.add(BenchCase.fail("df_train_export_pit",
                        "size=" + exp.size() + " noLeak=" + noLeak + " valueOk=" + valueOk
                                + " stats=" + exp.joinStats
                                + " sample=" + (exp.size() > 0 ? exp.dataset.row(0) : null),
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_train_export_pit",
                        "n=10 zero-leak hit=" + exp.joinStats.joinsHit
                                + " asof f=" + sampleF
                                + " futureRejected=" + exp.joinStats.futureRowsRejected,
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_train_export_pit", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void batchBuild(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame raw = syntheticUsers(40, false);
            FeatureIngest.into(fp)
                    .project("demo").view("ub").entities("user_id")
                    .features("city", "age", "score")
                    .from(raw).run();
            fp.featureService(FeatureService.builder("b_svc").project("demo").view("ub").build());

            DataFrame entity = DataFrameBridge.selectColumns(raw,
                    List.of("user_id", "event_timestamp", "label"));
            FeatureTrainingExport.Result exp = FeatureTrainingExport.from(fp)
                    .project("demo").featureService("b_svc")
                    .entityDataFrame(entity).labelColumn("label").run();

            List<Feature> feats = List.of(
                    Features.sparse("user_id", 1000, 8),
                    Features.sparse("city", 100, 8),
                    Features.dense("age", 1),
                    Features.dense("score", 1));
            // Map export column names: may be prefixed
            List<Feature> aligned = new ArrayList<>();
            for (Feature f : feats) {
                aligned.add(f);
            }
            List<Batch> batches = exp.toBatches(aligned, 16);
            boolean ok = !batches.isEmpty()
                    && batches.get(0).labels != null
                    && batches.get(0).labels.size(0) > 0;
            // close tensors lightly — benchmark process short-lived
            if (!ok) {
                suite.add(BenchCase.fail("df_batch_build",
                        "batches=" + batches.size(), System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_batch_build",
                        "batches=" + batches.size()
                                + " firstBatch=" + batches.get(0).labels.size(0)
                                + " sparseKeys=" + batches.get(0).sparseFeatures.keySet(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_batch_build", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void dualViewService(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame users = syntheticUsers(20, false);
            DataFrame items = syntheticItems(15);
            FeatureIngest.Result u = FeatureIngest.into(fp)
                    .project("rec").view("user_v").entities("user_id")
                    .features("age", "score", "city").from(users).run();
            FeatureIngest.Result it = FeatureIngest.into(fp)
                    .project("rec").view("item_v").entities("item_id")
                    .features("price", "cate").from(items).run();
            fp.featureService(FeatureService.builder("rank")
                    .project("rec").views("user_v", "item_v").build());
            MaterializationResult mat = FeatureMaterializeJob.on(fp)
                    .project("rec").fromIngest(List.of(u, it)).run();

            // entity pairs
            DataFrame ent = DataFrame.create();
            ent.addColumn("user_id", Column.DType.INT64);
            ent.addColumn("item_id", Column.DType.INT64);
            ent.addColumn("event_timestamp", Column.DType.INT64);
            ent.addColumn("label", Column.DType.FLOAT64);
            long now = System.currentTimeMillis();
            for (int i = 0; i < 20; i++) {
                int idx = ent.addRow();
                ent.set(idx, "user_id", (long) (i % 20));
                ent.set(idx, "item_id", (long) (i % 15));
                ent.set(idx, "event_timestamp", now);
                ent.set(idx, "label", i % 2 == 0 ? 1.0 : 0.0);
            }
            FeatureTrainingExport.Result exp = FeatureTrainingExport.from(fp)
                    .project("rec").featureService("rank")
                    .entityDataFrame(ent).labelColumn("label").run();

            FeatureResponse online = fp.getOnlineFeatures(FeatureRequest.builder()
                    .project("rec").featureService("rank")
                    .entity("user_id", 1L).entity("item_id", 2L)
                    .build());

            boolean ok = u.ok() && it.ok() && mat.success()
                    && exp.size() == 20
                    && online.success()
                    && online.vector().size() > 0;
            if (!ok) {
                suite.add(BenchCase.fail("df_dual_view_service",
                        "mat=" + mat + " exp=" + exp + " online=" + online,
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_dual_view_service",
                        "expN=20 onlineFeats=" + online.vector().size()
                                + " matWritten=" + mat.rowsWritten(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_dual_view_service", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void replaceVsAppend(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            DataFrame a = syntheticUsers(10, false);
            FeatureIngest.into(fp).project("demo").view("rep")
                    .entities("user_id").from(a).replace(false).run();
            long n1 = fp.offline().rowCount("demo", "rep");
            FeatureIngest.into(fp).project("demo").view("rep")
                    .entities("user_id").from(a).replace(false).run();
            long n2 = fp.offline().rowCount("demo", "rep");
            FeatureIngest.into(fp).project("demo").view("rep")
                    .entities("user_id").from(a).replace(true).run();
            long n3 = fp.offline().rowCount("demo", "rep");
            boolean ok = n1 == 10 && n2 == 20 && n3 == 10;
            if (!ok) {
                suite.add(BenchCase.fail("df_replace_append",
                        "n1=" + n1 + " n2=" + n2 + " n3=" + n3, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("df_replace_append",
                        "append 10→20, replace→10", System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("df_replace_append", e.toString(), System.nanoTime() - t0));
        }
    }

    // ── synthetic data ──────────────────────────────────────────────────────

    static DataFrame syntheticUsers(int n, boolean withNullAge) {
        DataFrame df = DataFrame.create();
        df.addColumn("user_id", Column.DType.INT64);
        df.addColumn("age", Column.DType.FLOAT64);
        df.addColumn("score", Column.DType.FLOAT64);
        df.addColumn("city", Column.DType.INT64);
        df.addColumn("event_timestamp", Column.DType.INT64);
        df.addColumn("label", Column.DType.FLOAT64);
        long now = System.currentTimeMillis();
        for (int i = 0; i < n; i++) {
            int idx = df.addRow();
            df.set(idx, "user_id", (long) i);
            if (withNullAge && i % 5 == 0) {
                df.set(idx, "age", null);
            } else {
                df.set(idx, "age", 18.0 + (i % 40));
            }
            df.set(idx, "score", 0.1 * (i % 10));
            df.set(idx, "city", (long) (i % 7));
            df.set(idx, "event_timestamp", now - i * 1000L);
            df.set(idx, "label", i % 3 == 0 ? 1.0 : 0.0);
        }
        return df;
    }

    static DataFrame syntheticItems(int n) {
        DataFrame df = DataFrame.create();
        df.addColumn("item_id", Column.DType.INT64);
        df.addColumn("price", Column.DType.FLOAT64);
        df.addColumn("cate", Column.DType.INT64);
        df.addColumn("event_timestamp", Column.DType.INT64);
        long now = System.currentTimeMillis();
        for (int i = 0; i < n; i++) {
            int idx = df.addRow();
            df.set(idx, "item_id", (long) i);
            df.set(idx, "price", 9.9 + i);
            df.set(idx, "cate", (long) (i % 5));
            df.set(idx, "event_timestamp", now - i * 2000L);
        }
        return df;
    }

    private static long toLong(Object v) {
        return v instanceof Number ? ((Number) v).longValue() : Long.MIN_VALUE;
    }

    private static double toDouble(Object v) {
        return v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
    }
}
