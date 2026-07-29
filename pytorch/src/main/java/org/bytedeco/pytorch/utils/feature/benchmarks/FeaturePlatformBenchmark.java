/*
 * FeaturePlatformBenchmark — full multi-dimension suite for the enterprise
 * feature management platform under org.bytedeco.pytorch.utils.feature.
 *
 * Dimensions:
 *   1) registry          — register / promote / lifecycle / file round-trip
 *   2) pit               — point-in-time join correctness (zero leakage) + perf + TTL
 *   3) online            — materialize→serve consistency, batch fanout, on-demand, concurrency
 *   4) materialize       — incremental watermark
 *   5) industry          — recsys/ecom/fintech/news/pharma catalog smoke
 *   6) multimodal        — embedding integrity, bridge, drift/freshness, schema/ACL
 *   7) store             — MEMORY/SQLITE/DUCKDB switch + Redis probe + codec
 *   8) dataframe         — DataFrame FE ↔ FeatureStore ingest/materialize/PIT/batch
 *   9) lifecycle         — raw→FE→ingest→materialize→online→PIT export→DeepFM→quality
 *
 * Run all:
 *   java org.bytedeco.pytorch.utils.feature.benchmarks.FeaturePlatformBenchmark
 * Filter:
 *   java ... FeaturePlatformBenchmark pit online industry
 *   java ... FeaturePlatformBenchmark registry multimodal
 *
 * System properties:
 *   -Dfeature.bench.entities=10000   PIT entity count (default 2000 smoke / 10000 full)
 *   -Dfeature.bench.views=200        registry view count (default 200)
 *   -Dfeature.bench.strict=true      exit non-zero on any failure (default true)
 */
package org.bytedeco.pytorch.utils.feature.benchmarks;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** Master benchmark runner for utils.feature. */
public final class FeaturePlatformBenchmark {

    private FeaturePlatformBenchmark() {}

    public static void main(String[] args) {
        int exit = run(args);
        System.exit(exit);
    }

    /** @return 0 if all selected suites passed (or strict=false), else 1 */
    public static int run(String[] args) {
        System.out.println("============================================================");
        System.out.println(" FeaturePlatformBenchmark — enterprise feature store suite");
        System.out.println("============================================================");

        Set<String> filters = new LinkedHashSet<>();
        if (args != null) {
            for (String a : args) {
                if (a != null && !a.isBlank()) filters.add(a.trim().toLowerCase(Locale.ROOT));
            }
        }
        boolean all = filters.isEmpty();

        int entities = intProp("feature.bench.entities", all ? 2000 : 2000);
        int views = intProp("feature.bench.views", 200);
        boolean strict = boolProp("feature.bench.strict", true);

        List<BenchCase.Suite> suites = new ArrayList<>();
        long t0 = System.nanoTime();

        if (all || filters.contains("registry")) {
            BenchCase.Suite s = new BenchCase.Suite("registry");
            System.out.println("\n--- registry ---");
            RegistryBenchmark.run(s, views);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("pit")) {
            BenchCase.Suite s = new BenchCase.Suite("pit");
            System.out.println("\n--- pit (point-in-time join) ---");
            PitJoinBenchmark.run(s, entities);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("online") || filters.contains("serving")) {
            BenchCase.Suite s = new BenchCase.Suite("online");
            System.out.println("\n--- online serving ---");
            OnlineServingBenchmark.run(s);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("materialize")) {
            BenchCase.Suite s = new BenchCase.Suite("materialize");
            System.out.println("\n--- materialize ---");
            MaterializeBenchmark.run(s);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("industry")) {
            BenchCase.Suite s = new BenchCase.Suite("industry");
            System.out.println("\n--- industry catalogs ---");
            IndustryCatalogBenchmark.run(s);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("multimodal") || filters.contains("bridge")
                || filters.contains("schema") || filters.contains("drift")) {
            BenchCase.Suite s = new BenchCase.Suite("multimodal");
            System.out.println("\n--- multimodal / bridge / quality ---");
            MultimodalBenchmark.run(s);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("store") || filters.contains("stores")
                || filters.contains("redis") || filters.contains("sqlite")) {
            BenchCase.Suite s = new BenchCase.Suite("store");
            System.out.println("\n--- store backends (memory/sqlite/duckdb/redis-probe) ---");
            StoreSwitchBenchmark.run(s);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("dataframe") || filters.contains("df")
                || filters.contains("ingest")) {
            BenchCase.Suite s = new BenchCase.Suite("dataframe");
            System.out.println("\n--- dataframe ↔ feature store ---");
            DataFrameFeatureStoreBenchmark.run(s);
            s.summary();
            suites.add(s);
        }
        if (all || filters.contains("lifecycle") || filters.contains("pipeline")
                || filters.contains("e2e")) {
            BenchCase.Suite s = new BenchCase.Suite("lifecycle");
            System.out.println("\n--- full lifecycle pipeline (FE→store→train) ---");
            LifecyclePipelineBenchmark.run(s);
            s.summary();
            suites.add(s);
        }

        int passed = 0;
        int failed = 0;
        int total = 0;
        for (BenchCase.Suite s : suites) {
            passed += s.passed();
            failed += s.failed();
            total += s.cases.size();
        }
        double ms = (System.nanoTime() - t0) / 1_000_000.0;
        System.out.println("\n============================================================");
        System.out.printf(Locale.ROOT,
                " TOTAL: %d/%d passed, %d failed, wall=%.1f ms%n",
                passed, total, failed, ms);
        System.out.println("============================================================");

        if (failed > 0) {
            System.out.println("Failed cases:");
            for (BenchCase.Suite s : suites) {
                for (BenchCase c : s.cases) {
                    if (!c.passed) System.out.println("  " + s.name + " :: " + c);
                }
            }
            return strict ? 1 : 0;
        }
        return 0;
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

    private static boolean boolProp(String key, boolean dflt) {
        String v = System.getProperty(key);
        if (v == null || v.isBlank()) return dflt;
        return "true".equalsIgnoreCase(v.trim()) || "1".equals(v.trim());
    }
}
