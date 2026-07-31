/*
 * Materialize-focused bench (incremental cursor + replace consistency).
 */
package org.bytedeco.pytorch.feature.benchmarks;

import org.bytedeco.pytorch.feature.FeaturePlatform;
import org.bytedeco.pytorch.feature.core.Entity;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;
import org.bytedeco.pytorch.feature.core.ValueType;
import org.bytedeco.pytorch.feature.materialize.MaterializationResult;
import org.bytedeco.pytorch.feature.online.OnlineFeatureRow;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Incremental materialization watermark bench. */
public final class MaterializeBenchmark {

    private MaterializeBenchmark() {}

    public static void run(BenchCase.Suite suite) {
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            Entity user = Entity.of("user_id");
            fp.entity(user);
            FeatureView view = FeatureView.builder("inc_v")
                    .entities(user)
                    .schema(Field.of("x", ValueType.INT64))
                    .online(true)
                    .build();
            fp.featureView(view);

            long tBase = 1_700_000_000_000L;
            List<Map<String, Object>> batch1 = new ArrayList<>();
            for (int i = 1; i <= 10; i++) {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("user_id", (long) i);
                r.put("event_timestamp", tBase + i);
                r.put("x", 1L);
                batch1.add(r);
            }
            fp.putOffline("default", "inc_v", batch1);
            MaterializationResult r1 = fp.materialize().materializeIncremental(List.of(view), null);
            long wm1 = fp.cursor().get("default", "inc_v");

            // second batch with newer timestamps + update user 1
            List<Map<String, Object>> batch2 = new ArrayList<>();
            for (int i = 1; i <= 10; i++) {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("user_id", (long) i);
                r.put("event_timestamp", tBase + 1000 + i);
                r.put("x", 2L);
                batch2.add(r);
            }
            fp.putOffline("default", "inc_v", batch2);
            MaterializationResult r2 = fp.materialize().materializeIncremental(List.of(view), null);
            long wm2 = fp.cursor().get("default", "inc_v");

            Optional<OnlineFeatureRow> row = fp.online().onlineRead("default", "inc_v", "1");
            Object x = row.map(rr -> rr.get("x")).orElse(null);
            long xv = x instanceof Number ? ((Number) x).longValue() : -1L;

            boolean ok = r1.success() && r2.success()
                    && wm2 > wm1
                    && xv == 2L
                    && r2.rowsWritten() >= 1;
            if (!ok) {
                suite.add(BenchCase.fail("materialize_incremental",
                        "wm1=" + wm1 + " wm2=" + wm2 + " x=" + x + " r1=" + r1 + " r2=" + r2,
                        System.nanoTime() - t0));
            } else {
                suite.add(BenchCase.pass("materialize_incremental",
                        "wm " + wm1 + "→" + wm2 + " x=2 written2=" + r2.rowsWritten(),
                        System.nanoTime() - t0));
            }
        } catch (Exception e) {
            suite.add(BenchCase.fail("materialize_incremental", e.toString(), System.nanoTime() - t0));
        }
    }
}
