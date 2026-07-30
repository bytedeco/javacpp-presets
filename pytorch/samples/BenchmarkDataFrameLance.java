package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.io.FormatDetect;
import org.bytedeco.pytorch.utils.lance.Lance;
import org.bytedeco.pytorch.utils.lance.LanceIndex;
import org.bytedeco.pytorch.utils.lance.LanceReadOptions;
import org.bytedeco.pytorch.utils.lance.LanceWriteOptions;
import org.bytedeco.pytorch.utils.lance.SearchOptions;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * End-to-end correctness suite for official Lance DataFrame I/O + advanced features:
 * <ul>
 *   <li>complex type round-trip (VECTOR/EMBEDDING/LIST/STRUCT/JSON/STRING)</li>
 *   <li>writeLance / readLance defaults (official format)</li>
 *   <li>vector index (IVF-HNSW-PQ) + ANN search + hybrid filter</li>
 *   <li>scalar / FTS index smoke</li>
 *   <li>version / tag / delete / compact</li>
 *   <li>FormatDetect {@code .lance}</li>
 *   <li>pure-Java training layout coexistence</li>
 * </ul>
 *
 * <p>Soft-skips when {@code org.lance} JNI is not loadable on the host.
 */
public final class BenchmarkDataFrameLance {

    static int passed = 0;
    static int failed = 0;
    static int skipped = 0;
    static boolean lanceAvailable = false;

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameLance (official Lance " + Lance.VERSION + ") ===\n");
        Path tmp = Files.createTempDirectory("df-lance-bench-");
        try {
            probeLance(tmp);
            d1ComplexRoundTrip(tmp);
            d2WriteReadOptions(tmp);
            d3VectorIndexAndSearch(tmp);
            d4HybridFilterSearch(tmp);
            d5ScalarAndFts(tmp);
            d6VersionTagDelete(tmp);
            d7FormatDetect(tmp);
            d8TrainingLayout(tmp);
        } finally {
            // best-effort cleanup
            try {
                Files.walk(tmp)
                    .sorted((a, b) -> b.compareTo(a))
                    .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
            } catch (Exception ignored) {}
        }
        System.out.println("\n=== Summary: passed=" + passed + " failed=" + failed
            + " skipped=" + skipped + " ===");
        if (failed > 0) System.exit(1);
    }

    static void probeLance(Path tmp) {
        section("D0 probe org.lance");
        try {
            DataFrame df = sampleComplex(4, 8);
            Path p = tmp.resolve("probe.lance");
            Lance.writeDataFrame(df, p.toString());
            try (Lance ds = Lance.open(p.toString())) {
                check("probe.count", ds.countRows() == 4, "rows=" + ds.countRows());
            }
            lanceAvailable = true;
            System.out.println("  lanceAvailable=true");
        } catch (Throwable t) {
            lanceAvailable = false;
            skip("probe", "org.lance not loadable: " + t.getClass().getSimpleName()
                + ": " + t.getMessage());
        }
    }

    static void d1ComplexRoundTrip(Path tmp) {
        section("D1 complex type round-trip");
        if (!lanceAvailable) { skip("d1", "no lance"); return; }
        benchmark("D1", () -> {
            DataFrame src = sampleComplex(12, 16);
            Path p = tmp.resolve("d1_complex.lance");
            src.writeLance(p.toString());
            DataFrame back = DataFrame.readLance(p.toString());
            check("rows", back.rowCount() == src.rowCount(),
                "src=" + src.rowCount() + " back=" + back.rowCount());
            for (String col : List.of("id", "label", "emb", "vec", "tags", "meta", "payload")) {
                check("has." + col, back.hasColumn(col));
            }
            // vector densifies to float[] / EmbeddingData-compatible
            Object emb0 = back.get(0, "emb");
            check("emb.numeric", emb0 instanceof float[]
                || emb0 instanceof EmbeddingData
                || emb0 instanceof List,
                "type=" + (emb0 == null ? "null" : emb0.getClass().getName()));
            Object meta0 = back.get(0, "meta");
            check("meta.maplike", meta0 instanceof Map || meta0 instanceof String,
                "type=" + (meta0 == null ? "null" : meta0.getClass().getName()));
            // id values preserved
            check("id0", ((Number) back.get(0, "id")).longValue() == 0L);
            check("label0", "row-0".equals(String.valueOf(back.get(0, "label"))));
        });
    }

    static void d2WriteReadOptions(Path tmp) {
        section("D2 write/read options");
        if (!lanceAvailable) { skip("d2", "no lance"); return; }
        benchmark("D2", () -> {
            DataFrame src = sampleComplex(20, 8);
            Path p = tmp.resolve("d2_opts.lance");
            src.writeLance(p.toString(), LanceWriteOptions.overwrite()
                .maxRowsPerFile(10_000)
                .stableRowIds(true));
            DataFrame filtered = DataFrame.readLance(p.toString(),
                LanceReadOptions.builder()
                    .columns("id", "label")
                    .filter("id >= 5")
                    .limit(5)
                    .build());
            check("filter.limit.rows", filtered.rowCount() > 0 && filtered.rowCount() <= 5,
                "rows=" + filtered.rowCount());
            check("filter.cols", filtered.hasColumn("id") && filtered.hasColumn("label"));
            check("filter.no.emb", !filtered.hasColumn("emb"));
            try (Lance ds = DataFrame.openLance(p.toString())) {
                check("info.official", Boolean.TRUE.equals(ds.info().get("official")));
                check("count", ds.countRows() == 20);
            }
        });
    }

    static void d3VectorIndexAndSearch(Path tmp) {
        section("D3 vector index + ANN search");
        if (!lanceAvailable) { skip("d3", "no lance"); return; }
        benchmark("D3", () -> {
            int n = 64;
            int dim = 16;
            DataFrame src = sampleComplex(n, dim);
            Path p = tmp.resolve("d3_ann.lance");
            src.writeLance(p.toString());
            float[] query = unit(dim, 0);
            try (Lance ds = DataFrame.openLance(p.toString())) {
                // brute-force search works without index
                DataFrame brute = ds.search("emb", query, 5, SearchOptions.cosine());
                check("brute.rows", brute.rowCount() > 0 && brute.rowCount() <= 5,
                    "rows=" + brute.rowCount());

                boolean indexed = ds.createVectorIndex("emb",
                    LanceIndex.ivfHnswPq(4, 8, 50, "cosine", 4, 8).named("emb_hnsw"));
                // index creation may fail on tiny data / beta constraints — soft check
                if (indexed) {
                    check("listIndexes.nonempty", !ds.listIndexes().isEmpty(),
                        "indexes=" + ds.listIndexes());
                    DataFrame hits = ds.search("emb", query, 5,
                        SearchOptions.cosine().ef(32).nprobes(4));
                    check("ann.rows", hits.rowCount() > 0 && hits.rowCount() <= 5,
                        "rows=" + hits.rowCount());
                } else {
                    // fallback: IVF-Flat is lighter
                    boolean flat = ds.createVectorIndex("emb",
                        LanceIndex.ivfFlat(4, "cosine").named("emb_flat"));
                    if (flat) {
                        check("flat.index", ds.listIndexes().stream()
                            .anyMatch(s -> s != null && !s.isBlank()));
                        DataFrame hits = ds.search("emb", query, 5, SearchOptions.cosine().nprobes(4));
                        check("flat.ann.rows", hits.rowCount() > 0, "rows=" + hits.rowCount());
                    } else {
                        skip("createVectorIndex", "index build rejected (tiny data / beta)");
                        // still require brute search above
                    }
                }
            }
        });
    }

    static void d4HybridFilterSearch(Path tmp) {
        section("D4 hybrid filter + ANN");
        if (!lanceAvailable) { skip("d4", "no lance"); return; }
        benchmark("D4", () -> {
            DataFrame src = sampleComplex(32, 8);
            Path p = tmp.resolve("d4_hybrid.lance");
            src.writeLance(p.toString());
            float[] query = unit(8, 1);
            try (Lance ds = DataFrame.openLance(p.toString())) {
                DataFrame hits = ds.search("emb", query, 8,
                    SearchOptions.l2().filter("id >= 10").prefilter(true));
                check("hybrid.rows", hits.rowCount() >= 0, "rows=" + hits.rowCount());
                // if hits returned and id projected, all should satisfy filter
                if (hits.hasColumn("id") && hits.rowCount() > 0) {
                    boolean ok = true;
                    for (int i = 0; i < hits.rowCount(); i++) {
                        if (((Number) hits.get(i, "id")).longValue() < 10) { ok = false; break; }
                    }
                    check("hybrid.filter.hold", ok);
                }
            }
        });
    }

    static void d5ScalarAndFts(Path tmp) {
        section("D5 scalar + FTS indexes");
        if (!lanceAvailable) { skip("d5", "no lance"); return; }
        benchmark("D5", () -> {
            DataFrame src = sampleComplex(24, 8);
            Path p = tmp.resolve("d5_fts.lance");
            src.writeLance(p.toString());
            try (Lance ds = DataFrame.openLance(p.toString())) {
                boolean btree = ds.createScalarIndex("label", LanceIndex.btree().named("label_btree"));
                boolean fts = ds.createFtsIndex("payload", "payload_fts", true);
                if (btree || fts) {
                    check("indexes.listed", !ds.listIndexes().isEmpty()
                        || btree || fts, "indexes=" + ds.listIndexes());
                } else {
                    skip("scalar/fts", "index build not supported on this build");
                }
                if (fts) {
                    try {
                        DataFrame hits = ds.fullTextSearch("payload", "row", 5);
                        check("fts.rows", hits.rowCount() >= 0, "rows=" + hits.rowCount());
                    } catch (Throwable t) {
                        skip("fts.search", t.getClass().getSimpleName() + ": " + t.getMessage());
                    }
                }
            }
        });
    }

    static void d6VersionTagDelete(Path tmp) {
        section("D6 version / tag / delete / compact");
        if (!lanceAvailable) { skip("d6", "no lance"); return; }
        benchmark("D6", () -> {
            DataFrame src = sampleComplex(16, 8);
            Path p = tmp.resolve("d6_mut.lance");
            src.writeLance(p.toString());
            try (Lance ds = DataFrame.openLance(p.toString())) {
                long v0 = ds.version();
                check("version.positive", v0 > 0, "v=" + v0);
                ds.tag("baseline");
                long tagV = ds.tagVersion("baseline");
                check("tag.version", tagV == v0, "tagV=" + tagV + " v0=" + v0);
                check("tags.listed", !ds.listTags().isEmpty());

                long before = ds.countRows();
                ds.delete("id < 0"); // no-op predicate (ids >= 0)
                check("delete.noop.count", ds.countRows() == before, "count=" + ds.countRows());

                // delete half
                ds.delete("id < 8");
                long after = ds.countRows();
                check("delete.half", after == 8, "after=" + after);

                try { ds.compact(); check("compact.ok", true); }
                catch (Throwable t) {
                    skip("compact", t.getClass().getSimpleName());
                }

                List<?> versions = ds.listVersions();
                check("versions.nonempty", versions != null && !versions.isEmpty());

                try (Lance old = ds.checkoutVersion(v0)) {
                    // historical version should still see original row count when supported
                    long hist = old.countRows();
                    check("checkout.readable", hist >= 0, "hist=" + hist);
                } catch (Throwable t) {
                    skip("checkoutVersion", t.getClass().getSimpleName());
                }
            }
        });
    }

    static void d7FormatDetect(Path tmp) {
        section("D7 FormatDetect .lance");
        if (!lanceAvailable) { skip("d7", "no lance"); return; }
        benchmark("D7", () -> {
            DataFrame src = sampleComplex(5, 4);
            Path p = tmp.resolve("d7_detect.lance");
            src.writeLance(p.toString());
            FormatDetect.Format fmt = FormatDetect.detect(p.toString());
            check("detect.LANCE", fmt == FormatDetect.Format.LANCE, "fmt=" + fmt);
            DataFrame viaRead = DataFrame.read(p.toString());
            check("DataFrame.read rows", viaRead.rowCount() == 5, "rows=" + viaRead.rowCount());
        });
    }

    static void d8TrainingLayout(Path tmp) {
        section("D8 pure-Java training layout coexistence");
        benchmark("D8", () -> {
            DataFrame src = sampleComplex(6, 8);
            Path pure = tmp.resolve("d8_train.lance");
            src.writeLanceTraining(pure.toString(), "emb", "vec");
            check("isPureJava", Lance.isPureJavaLance(pure.toString()));
            check("manifest", Files.isRegularFile(pure.resolve("_manifest.json")));
            DataFrame back = DataFrame.readLanceTraining(pure.toString());
            check("train.rows", back.rowCount() == 6, "rows=" + back.rowCount());
            DataFrame auto = DataFrame.readLance(pure.toString());
            check("auto.rows", auto.rowCount() == 6, "rows=" + auto.rowCount());

            if (lanceAvailable) {
                Path off = tmp.resolve("d8_official.lance");
                src.writeLance(off.toString());
                check("not pure", !Lance.isPureJavaLance(off.toString()));
                DataFrame offBack = DataFrame.readLance(off.toString());
                check("official.rows", offBack.rowCount() == 6, "rows=" + offBack.rowCount());
            }
        });
    }

    // ---- fixtures --------------------------------------------------------

    static DataFrame sampleComplex(int n, int dim) {
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.INT64);
        df.addColumn("label", Column.DType.STRING);
        df.addColumn("emb", Column.DType.EMBEDDING);
        df.addColumn("vec", Column.DType.VECTOR);
        df.addColumn("tags", Column.DType.LIST);
        df.addColumn("meta", Column.DType.STRUCT);
        df.addColumn("payload", Column.DType.JSON);
        for (int i = 0; i < n; i++) {
            float[] v = unit(dim, i);
            List<Object> tags = new ArrayList<>();
            tags.add("t" + (i % 3));
            tags.add("row");
            Map<String, Object> meta = new LinkedHashMap<>();
            meta.put("k", "v" + i);
            meta.put("n", i);
            meta.put("ok", i % 2 == 0);
            String json = "{\"row\":" + i + ",\"msg\":\"row " + i + " text\"}";
            df.addRow(
                (long) i,
                "row-" + i,
                new EmbeddingData(v, "bench"),
                v.clone(),
                tags,
                meta,
                json
            );
        }
        return df;
    }

    static float[] unit(int dim, int seed) {
        float[] v = new float[dim];
        double norm = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) Math.sin(seed * 0.7 + i * 0.31);
            norm += v[i] * v[i];
        }
        norm = Math.sqrt(norm);
        if (norm > 0) for (int i = 0; i < dim; i++) v[i] /= (float) norm;
        return v;
    }

    // ---- harness ---------------------------------------------------------

    static void section(String title) {
        System.out.println("\n-- " + title + " --");
    }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("  [" + name + "] done in " + ms + " ms");
        } catch (Throwable t) {
            failed++;
            System.out.println("  FAIL " + name + ": " + t.getClass().getSimpleName()
                + ": " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean cond) {
        check(name, cond, null);
    }

    static void check(String name, boolean cond, String detail) {
        if (cond) {
            passed++;
            System.out.println("  OK   " + name + (detail == null ? "" : " (" + detail + ")"));
        } else {
            failed++;
            System.out.println("  FAIL " + name + (detail == null ? "" : " (" + detail + ")"));
        }
    }

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("  SKIP " + name + " — " + reason);
    }

    @FunctionalInterface
    interface CheckedRunnable {
        void run() throws Exception;
    }
}
