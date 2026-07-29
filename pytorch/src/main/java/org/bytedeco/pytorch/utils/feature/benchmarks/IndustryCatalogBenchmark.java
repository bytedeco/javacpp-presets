/*
 * Industry catalog end-to-end smoke: register → sample data → materialize → serve.
 */
package org.bytedeco.pytorch.utils.feature.benchmarks;

import org.bytedeco.pytorch.utils.feature.FeaturePlatform;
import org.bytedeco.pytorch.utils.feature.industry.EcommerceCatalog;
import org.bytedeco.pytorch.utils.feature.industry.FintechCatalog;
import org.bytedeco.pytorch.utils.feature.industry.IndustryFeatureCatalog;
import org.bytedeco.pytorch.utils.feature.industry.NewsCatalog;
import org.bytedeco.pytorch.utils.feature.industry.PharmaBioCatalog;
import org.bytedeco.pytorch.utils.feature.industry.RecsysCatalog;
import org.bytedeco.pytorch.utils.feature.materialize.MaterializationResult;
import org.bytedeco.pytorch.utils.feature.serving.FeatureRequest;
import org.bytedeco.pytorch.utils.feature.serving.FeatureResponse;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Per-domain catalog smoke benchmarks. */
public final class IndustryCatalogBenchmark {

    private IndustryCatalogBenchmark() {}

    public static void run(BenchCase.Suite suite) {
        runOne(suite, new RecsysCatalog(), entityRecsys());
        runOne(suite, new EcommerceCatalog(), entityEcom());
        runOne(suite, new FintechCatalog(), entityFintech());
        runOne(suite, new NewsCatalog(), entityNews());
        runOne(suite, new PharmaBioCatalog(), entityPharma());
    }

    private static Map<String, Object> entityRecsys() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("user_id", 1L);
        m.put("item_id", 1L);
        return m;
    }

    private static Map<String, Object> entityEcom() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("user_id", 1L);
        m.put("sku_id", 1L);
        return m;
    }

    private static Map<String, Object> entityFintech() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("account_id", 1L);
        m.put("device_id", "dev_1");
        return m;
    }

    private static Map<String, Object> entityNews() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("user_id", 1L);
        m.put("news_id", 1L);
        return m;
    }

    private static Map<String, Object> entityPharma() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("compound_id", 1L);
        m.put("target_id", 1L);
        return m;
    }

    private static void runOne(BenchCase.Suite suite, IndustryFeatureCatalog catalog,
                               Map<String, Object> entity) {
        String name = "industry_" + catalog.domain().name().toLowerCase();
        long t0 = System.nanoTime();
        try (FeaturePlatform fp = FeaturePlatform.inMemory()) {
            catalog.registerAll(fp);
            long now = System.currentTimeMillis();
            Map<String, List<Map<String, Object>>> sample = catalog.sampleOfflineData(now, 20, 20);
            for (Map.Entry<String, List<Map<String, Object>>> e : sample.entrySet()) {
                fp.putOffline(catalog.project(), e.getKey(), e.getValue());
            }
            MaterializationResult mat = fp.materializeAll(catalog.project());
            if (!mat.success()) {
                suite.add(BenchCase.fail(name, "materialize failed: " + mat, System.nanoTime() - t0));
                return;
            }
            FeatureResponse resp = fp.getOnlineFeatures(FeatureRequest.builder()
                    .project(catalog.project())
                    .featureService(catalog.primaryService())
                    .entities(entity)
                    .requestContext("request_ts", now)
                    .requestContext("price", 10.0)
                    .build());
            if (!resp.success()) {
                suite.add(BenchCase.fail(name, "serve failed: " + resp, System.nanoTime() - t0));
                return;
            }
            suite.add(BenchCase.pass(name,
                    "views=" + catalog.featureViews().size()
                            + " written=" + mat.rowsWritten()
                            + " feats=" + resp.vector().size()
                            + " hit=" + resp.viewsHit(),
                    System.nanoTime() - t0));
        } catch (Exception e) {
            suite.add(BenchCase.fail(name, e.toString(), System.nanoTime() - t0));
        }
    }
}
