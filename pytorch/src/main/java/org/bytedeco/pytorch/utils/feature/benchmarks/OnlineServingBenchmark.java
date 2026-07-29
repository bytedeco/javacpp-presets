/*
 * Online serving + materialize consistency benchmarks.
 */
package org.bytedeco.pytorch.utils.feature.benchmarks;

import org.bytedeco.pytorch.utils.feature.FeaturePlatform;
import org.bytedeco.pytorch.utils.feature.core.Entity;
import org.bytedeco.pytorch.utils.feature.core.FeatureService;
import org.bytedeco.pytorch.utils.feature.core.FeatureView;
import org.bytedeco.pytorch.utils.feature.core.Field;
import org.bytedeco.pytorch.utils.feature.core.OnDemandFeatureView;
import org.bytedeco.pytorch.utils.feature.core.ValueType;
import org.bytedeco.pytorch.utils.feature.materialize.MaterializationResult;
import org.bytedeco.pytorch.utils.feature.serving.FeatureRequest;
import org.bytedeco.pytorch.utils.feature.serving.FeatureResponse;
import org.bytedeco.pytorch.utils.feature.transform.OnDemandCompute;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Online get + batch fanout + materialize consistency. */
public final class OnlineServingBenchmark {

    private OnlineServingBenchmark() {}

    public static void run(BenchCase.Suite suite) {
        materializeAndServe(suite);
        batchFanout(suite);
        onDemand(suite);
        concurrency(suite);
    }

    private static void materializeAndServe(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            Entity user = Entity.of("user_id");
            fp.entity(user);
            FeatureView view = FeatureView.builder("u_stats")
                    .entities(user)
                    .schema(Field.of("clicks", ValueType.INT64), Field.of("score", ValueType.FLOAT64))
                    .online(true)
                    .build();
            fp.featureView(view);
            fp.featureService(FeatureService.builder("svc")
                    .views("u_stats")
                    .build());

            long now = System.currentTimeMillis();
            List<Map<String, Object>> rows = new ArrayList<>();
            for (int i = 1; i <= 100; i++) {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("user_id", (long) i);
                r.put("event_timestamp", now - i * 1000L);
                r.put("clicks", (long) (i * 3));
                r.put("score", i * 0.1);
                rows.add(r);
            }
            // older row for user 1 that should lose to newer
            Map<String, Object> older = new LinkedHashMap<>();
            older.put("user_id", 1L);
            older.put("event_timestamp", now - 1_000_000L);
            older.put("clicks", 1L);
            older.put("score", 0.01);
            rows.add(older);

            fp.putOffline("default", "u_stats", rows);
            MaterializationResult mat = fp.materializeViews(List.of(view));
            if (!mat.success() || mat.rowsWritten() < 100) {
                suite.add(BenchCase.fail("materialize", mat.toString(), System.nanoTime() - t0));
                return;
            }

            FeatureResponse resp = fp.getOnlineFeatures("svc", Map.of("user_id", 1L));
            Object clicks = resp.vector().raw().get("clicks");
            if (clicks == null) clicks = resp.vector().raw().get("u_stats__clicks");
            long c = clicks instanceof Number ? ((Number) clicks).longValue() : -1L;
            if (c != 3L) {
                suite.add(BenchCase.fail("online_consistency",
                        "expected clicks=3 got " + clicks + " resp=" + resp, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("materialize_online",
                        "written=" + mat.rowsWritten() + " clicks=" + c + " ms=" + resp.elapsedMs(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("materialize_online", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void batchFanout(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            Entity user = Entity.of("user_id");
            Entity item = Entity.builder("item_id").joinKey("item_id").build();
            fp.entity(user);
            fp.entity(item);
            FeatureView uv = FeatureView.builder("user_f")
                    .entities(user)
                    .schema(Field.of("u", ValueType.FLOAT64))
                    .online(true).build();
            FeatureView iv = FeatureView.builder("item_f")
                    .entities(item)
                    .schema(Field.of("i", ValueType.FLOAT64))
                    .online(true).build();
            fp.featureView(uv);
            fp.featureView(iv);
            fp.featureService(FeatureService.builder("rank").views("user_f", "item_f").build());

            long now = System.currentTimeMillis();
            fp.putOffline("default", "user_f", List.of(Map.of(
                    "user_id", 1L, "event_timestamp", now, "u", 1.5)));
            List<Map<String, Object>> items = new ArrayList<>();
            for (int i = 1; i <= 100; i++) {
                items.add(Map.of("item_id", (long) i, "event_timestamp", now, "i", i * 0.01));
            }
            fp.putOffline("default", "item_f", items);
            fp.materializeViews(List.of(uv, iv));

            List<Map<String, Object>> entityRows = new ArrayList<>();
            for (int i = 1; i <= 100; i++) {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("user_id", 1L);
                row.put("item_id", (long) i);
                entityRows.add(row);
            }
            FeatureResponse resp = fp.getOnlineFeatures(FeatureRequest.builder()
                    .featureService("rank")
                    .entityRows(entityRows)
                    .build());
            if (resp.size() != 100 || !resp.success()) {
                suite.add(BenchCase.fail("online_batch_fanout",
                        "size=" + resp.size() + " " + resp, System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("online_batch_fanout",
                        "n=100 ms=" + resp.elapsedMs() + " hit=" + resp.viewsHit(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("online_batch_fanout", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void onDemand(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            fp.onDemand(OnDemandFeatureView.builder("od_time")
                    .schema(Field.of("hour_of_day", ValueType.INT64),
                            Field.of("day_of_week", ValueType.INT64),
                            Field.of("is_weekend", ValueType.INT64))
                    .compute(OnDemandCompute.timeContext("request_ts"))
                    .build());
            // minimal service with only on-demand — need a dummy view or empty views list
            // FeatureService with only onDemand
            fp.featureService(FeatureService.builder("od_only")
                    .onDemandView("od_time")
                    .build());

            // Online path with empty batch views still runs on-demand if entity rows present
            FeatureResponse resp = fp.getOnlineFeatures(FeatureRequest.builder()
                    .featureService("od_only")
                    .entity("user_id", 1L)
                    .requestContext("request_ts", 1_704_067_200_000L) // fixed epoch
                    .build());
            Object hour = resp.vector().raw().get("hour_of_day");
            if (hour == null) hour = resp.vector().raw().get("od_time__hour_of_day");
            if (!(hour instanceof Number) || resp.onDemandComputed() < 1) {
                suite.add(BenchCase.fail("on_demand",
                        "hour=" + hour + " od=" + resp.onDemandComputed() + " " + resp.vector(),
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("on_demand",
                        "hour=" + hour + " od=" + resp.onDemandComputed(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("on_demand", e.toString(), System.nanoTime() - t0));
        }
    }

    private static void concurrency(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            Entity user = Entity.of("user_id");
            fp.entity(user);
            FeatureView view = FeatureView.builder("c_view")
                    .entities(user)
                    .schema(Field.of("x", ValueType.INT64))
                    .online(true).build();
            fp.featureView(view);
            fp.featureService(FeatureService.builder("c_svc").views("c_view").build());
            long now = System.currentTimeMillis();
            List<Map<String, Object>> rows = new ArrayList<>();
            for (int i = 1; i <= 50; i++) {
                rows.add(Map.of("user_id", (long) i, "event_timestamp", now, "x", (long) i));
            }
            fp.putOffline("default", "c_view", rows);
            fp.materializeViews(List.of(view));

            int threads = 8;
            int perThread = 50;
            Thread[] ts = new Thread[threads];
            final boolean[] ok = {true};
            for (int t = 0; t < threads; t++) {
                ts[t] = new Thread(() -> {
                    try {
                        for (int i = 0; i < perThread; i++) {
                            FeatureResponse r = fp.getOnlineFeatures("c_svc",
                                    Map.of("user_id", (long) (i % 50 + 1)));
                            if (!r.success()) ok[0] = false;
                        }
                    } catch (Exception e) {
                        ok[0] = false;
                    }
                });
                ts[t].start();
            }
            for (Thread th : ts) th.join();
            if (!ok[0]) {
                suite.add(BenchCase.fail("concurrency", "parallel online reads failed", System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("concurrency",
                        "threads=" + threads + " gets=" + (threads * perThread),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("concurrency", e.toString(), System.nanoTime() - t0));
        }
    }
}
