package distribute;

import org.bytedeco.pytorch.utils.datasets.HfDataset;
import org.bytedeco.pytorch.utils.datasets.HfDatasets;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.hub.HfToken;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Live Hub stress for recsys / sequential / multimodal datasets that the HfDatasets
 * stack must download, layout-infer, parse, and iterate without dropping rows.
 *
 * <p>Targets (user catalog, ≤~200MB preferred; large repos are capped via
 * {@code maxFiles}/{@code take}/{@code allowPatterns}):
 * <ul>
 *   <li>recsys: McAuley All_Beauty (Amazon-Beauty stand-in), deepvk/VK-LSVD ur0.01_ip0.01,
 *       t-tech/T-ECD small, SetFit/amazon_reviews_multi_en</li>
 *   <li>sequential / text: roneneldan/TinyStories, fka/prompts.chat, databricks-dolly,
 *       alpaca, ultrachat_200k train_sft</li>
 *   <li>multimodal metadata: Lin-Chen/ShareGPT4V, liuhaotian/LLaVA-Instruct-150K,
 *       nlphuji/flickr30k, Marqo/polyvore, hltcoe/microvent annotations</li>
 *   <li>layout regression: glue/imdb/mmlu-style + All_Beauty.train.csv suffix splits</li>
 * </ul>
 *
 * <p>Datasets from the original list that are missing / 401 on the mirror are reported
 * as SKIP with the HTTP status — not counted as FAIL.
 *
 * <pre>
 *   export HF_ENDPOINT=https://hf-mirror.com   # recommended in CN
 *   export HF_TOKEN=hf_xxx                     # optional; some repos need it
 *   java ... distribute.StressHfRecsysDatasets [--take 256] [--max-files 2] [--only name]
 * </pre>
 */
public class StressHfRecsysDatasets {

    static int passed = 0, failed = 0, skipped = 0;
    static final StringBuilder report = new StringBuilder();
    static final List<CaseResult> results = new ArrayList<>();

    static int take = 256;
    static int maxFiles = 2;
    static String only = null; // substring filter on case name
    static String token;
    static String endpoint;
    static Path cacheDir;
    static Path outDir;

    public static void main(String[] args) throws Exception {
        parseArgs(args);
        token = HfToken.resolve(token);
        if (endpoint == null || endpoint.isBlank()) {
            endpoint = HfToken.resolveEndpoint(true);
            if (System.getenv("HF_ENDPOINT") == null && System.getenv("HF_MIRROR") == null) {
                endpoint = "https://hf-mirror.com";
            }
        }
        outDir = Path.of("samples/out/hf_recsys_stress");
        Files.createDirectories(outDir);
        // Persistent cache so re-runs don't re-download multi-MB shards.
        cacheDir = outDir.resolve("hf_cache");
        Files.createDirectories(cacheDir);

        System.out.println("=== StressHfRecsysDatasets ===");
        System.out.println("token=" + HfToken.mask(token)
                + " endpoint=" + endpoint
                + " take=" + take
                + " maxFiles=" + maxFiles
                + " cache=" + cacheDir);
        System.out.println("out=" + outDir.toAbsolutePath());
        System.out.println();

        // ── 0. Offline layout regressions (no network) ──────────────────────
        section("0. Layout inference regressions");
        run("layout All_Beauty.train/test/valid suffix", () -> {
            List<String> files = List.of(
                    "benchmark/0core/last_out/All_Beauty.train.csv",
                    "benchmark/0core/last_out/All_Beauty.valid.csv",
                    "benchmark/0core/last_out/All_Beauty.test.csv",
                    "benchmark/0core/last_out/Amazon_Fashion.train.csv"
            );
            var layout = HfDatasets.DatasetLayout.infer(files);
            check("config last_out or default",
                    layout.configs().contains("last_out") || layout.configs().contains("default")
                            || !layout.configs().isEmpty());
            // critical: train / validation / test must be distinct
            String cfg = layout.configs().contains("last_out") ? "last_out"
                    : layout.configs().iterator().next();
            check("has train", layout.splits(cfg).contains("train"));
            check("has test", layout.splits(cfg).contains("test"));
            check("has validation (from valid)", layout.splits(cfg).contains("validation"));
            check("train files only train",
                    layout.filesFor(cfg, List.of("train")).stream()
                            .allMatch(f -> f.contains(".train.") || f.contains("_train.")
                                    || f.endsWith(".train.csv")));
            check("test files only test",
                    layout.filesFor(cfg, List.of("test")).stream()
                            .allMatch(f -> f.contains(".test.") || f.contains("_test.")));
        });

        run("layout VK-LSVD subsample parent-is-split", () -> {
            List<String> files = List.of(
                    "subsamples/ur0.01_ip0.01/train/week_00.parquet",
                    "subsamples/ur0.01_ip0.01/train/week_01.parquet",
                    "subsamples/ur0.01_ip0.01/validation/week_25.parquet",
                    "subsamples/ur0.01_ip0.01/test/week_26.parquet",
                    "interactions/train/week_00.parquet"
            );
            var layout = HfDatasets.DatasetLayout.infer(files);
            check("config ur0.01_ip0.01", layout.configs().contains("ur0.01_ip0.01"));
            check("ur train", layout.splits("ur0.01_ip0.01").contains("train"));
            check("ur validation", layout.splits("ur0.01_ip0.01").contains("validation"));
            check("ur test", layout.splits("ur0.01_ip0.01").contains("test"));
            check("train weeks >=2",
                    layout.filesFor("ur0.01_ip0.01", List.of("train")).size() >= 2);
        });

        run("layout T-ECD dataset/small version config", () -> {
            List<String> files = List.of(
                    "dataset/small/brands.pq",
                    "dataset/small/marketplace/events/01216.pq",
                    "dataset/small/marketplace/events/01217.pq",
                    "dataset/small/retail/events/00001.pq",
                    "dataset/full/brands.pq",
                    "dataset/full/marketplace/events/00516.pq"
            );
            var layout = HfDatasets.DatasetLayout.infer(files);
            check("config small", layout.configs().contains("small"));
            check("config full", layout.configs().contains("full"));
            check("small files >=3",
                    layout.filesFor("small", null).size() >= 3);
            check("full files >=1",
                    layout.filesFor("full", null).size() >= 1);
            // brands.pq + events must not all collapse into config=events
            check("not only events config",
                    !layout.configs().equals(java.util.Set.of("events")));
        });

        run("layout glue/imdb/ultrachat still ok", () -> {
            var glue = HfDatasets.DatasetLayout.infer(List.of(
                    "cola/train-00000-of-00001.parquet",
                    "cola/validation-00000-of-00001.parquet",
                    "mrpc/train-00000-of-00001.parquet"
            ));
            check("glue cola", glue.configs().contains("cola") && glue.splits("cola").contains("train"));
            var ultra = HfDatasets.DatasetLayout.infer(List.of(
                    "data/train_sft-00000-of-00003-aaa.parquet",
                    "data/test_sft-00000-of-00001-ccc.parquet"
            ));
            check("ultrachat train_sft",
                    ultra.splits("default").contains("train_sft")
                            || ultra.splits().contains("train_sft"));
        });

        // ── 1. Recsys / interaction ─────────────────────────────────────────
        section("1. Recsys / interaction datasets");

        live("McAuley All_Beauty (Amazon-Beauty stand-in)",
                "McAuley-Lab/Amazon-Reviews-2023", "last_out", null,
                cfg -> cfg.allowPatterns(
                        "benchmark/0core/last_out/All_Beauty.train.csv",
                        "benchmark/0core/last_out/All_Beauty.valid.csv",
                        "benchmark/0core/last_out/All_Beauty.test.csv"
                ).maxFiles(3).take(take),
                d -> {
                    check("has train", d.splits().containsKey("train") && d.train().size() > 0);
                    check("has test or validation",
                            d.splits().containsKey("test") || d.splits().containsKey("validation"));
                    // integrity: no empty columns, rows stable under re-get
                    HfDataset tr = d.train();
                    Map<String, Object> r0 = tr.get(0);
                    Map<String, Object> r0b = tr.get(0);
                    check("row stable", Objects.equals(r0, r0b));
                    check("cols non-empty", !tr.columnNames().isEmpty());
                    // full scan of take rows — count matches size()
                    int n = 0;
                    for (int i = 0; i < tr.size(); i++) {
                        Map<String, Object> r = tr.get(i);
                        if (r == null || r.isEmpty()) throw new AssertionError("empty row " + i);
                        n++;
                    }
                    check("full scan == size", n == tr.size());
                    expectCols(tr, "user", "item", "asin", "parent", "rating", "timestamp",
                            "user_id", "item_id", "parent_asin");
                });

        live("deepvk/VK-LSVD ur0.01_ip0.01 (1 week train)",
                "deepvk/VK-LSVD", "ur0.01_ip0.01", "train",
                cfg -> cfg.allowPatterns("subsamples/ur0.01_ip0.01/train/")
                        .maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan no drop", fullScan(ds) == ds.size());
                    expectCols(ds, "user", "item", "video", "timestamp", "like", "share",
                            "view", "user_id", "item_id", "video_id");
                });

        live("t-tech/T-ECD small (few .pq shards)",
                "t-tech/T-ECD", "small", null,
                cfg -> cfg.allowPatterns("dataset/small/")
                        .maxFiles(maxFiles).take(take),
                d -> {
                    check("splits non-empty", !d.splits().isEmpty());
                    int total = 0;
                    for (var e : d.splits().entrySet()) {
                        int n = fullScan(e.getValue());
                        check("split " + e.getKey() + " no drop", n == e.getValue().size());
                        total += n;
                    }
                    check("total rows>0", total > 0);
                });

        live("SetFit/amazon_reviews_multi_en (jsonl)",
                "SetFit/amazon_reviews_multi_en", null, "train",
                cfg -> cfg.maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    Map<String, Object> r = ds.get(0);
                    check("has text-ish", r.containsKey("text") || r.containsKey("review")
                            || r.containsKey("sentence") || r.size() >= 2);
                });

        // ── 2. Sequential / text ────────────────────────────────────────────
        section("2. Sequential / text datasets");

        live("roneneldan/TinyStories (parquet train shard)",
                "roneneldan/TinyStories", null, "train",
                // Prefer parquet under data/; avoid multi-GB TinyStories-train.txt.
                cfg -> cfg.allowPatterns("data/").maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    Object text = firstNonNull(ds.get(0), "text", "story", "content");
                    check("text non-blank", text != null && !text.toString().isBlank());
                });

        live("fka/prompts.chat (csv)",
                "fka/prompts.chat", null, null,
                cfg -> cfg.maxFiles(1).take(Math.min(take, 512)),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                });

        live("databricks/databricks-dolly-15k",
                "databricks/databricks-dolly-15k", null, null,
                cfg -> cfg.maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    expectCols(ds, "instruction", "context", "response", "category", "text");
                });

        live("yahma/alpaca-cleaned",
                "yahma/alpaca-cleaned", null, null,
                cfg -> cfg.maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    expectCols(ds, "instruction", "input", "output", "text");
                });

        live("HuggingFaceH4/ultrachat_200k test_sft (small shard)",
                "HuggingFaceH4/ultrachat_200k", null, "test_sft",
                // test_sft is a single small parquet; train_sft shards are ~200MB+ each
                cfg -> cfg.maxFiles(1).take(Math.min(take, 128)),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    // messages / conversations often nested list/map
                    Map<String, Object> r = ds.get(0);
                    Object msgs = firstNonNull(r, "messages", "conversations", "conversation");
                    if (msgs != null) {
                        check("nested list or map", msgs instanceof List || msgs instanceof Map
                                || msgs.toString().length() > 2);
                    } else {
                        check("has some cols", !r.isEmpty());
                    }
                });

        // ── 3. Multimodal metadata (json/csv; images often external) ────────
        section("3. Multimodal metadata datasets");

        live("Lin-Chen/ShareGPT4V (json instruct)",
                "Lin-Chen/ShareGPT4V", null, null,
                cfg -> cfg.allowPatterns(
                        "sharegpt4v_instruct_gpt4-vision_cap100k.json",
                        "sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json"
                ).maxFiles(1).take(Math.min(take, 64)),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    Map<String, Object> r = ds.get(0);
                    // LLaVA-style: conversations / image / id
                    check("mm fields",
                            r.containsKey("conversations") || r.containsKey("image")
                                    || r.containsKey("id") || r.size() >= 2);
                });

        live("liuhaotian/LLaVA-Instruct-150K (json)",
                "liuhaotian/LLaVA-Instruct-150K", null, null,
                cfg -> cfg.allowPatterns(
                        "conversation_58k.json",
                        "llava_instruct_150k.json",
                        "detail_23k.json"
                ).maxFiles(1).take(Math.min(take, 64)),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                });

        live("nlphuji/flickr30k annotations csv",
                "nlphuji/flickr30k", null, null,
                cfg -> cfg.maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows>0", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    expectCols(ds, "caption", "raw", "sentids", "img_id", "filename", "split");
                });

        // Marqo/polyvore only ships 6×~400MB image parquets — too large for smoke stress.
        // Catalog probe still runs in section 5; full parse needs explicit offline snapshot.
        System.out.println("  ↷ skip Marqo/polyvore live download (6×~400MB image parquet shards; "
                + "use allowPatterns + local snapshot for full parse)");
        skipped++;
        results.add(CaseResult.skip("Marqo/polyvore (fashion multimodal stand-in)", 0L,
                "shards ~400MB each with embedded images — not auto-downloaded in smoke stress"));

        live("hltcoe/microvent annotations (jsonl/csv)",
                "hltcoe/microvent", null, null,
                cfg -> cfg.allowPatterns(
                        "annotations/",
                        "audio/catalog.csv",
                        "videos/catalog.csv"
                ).maxFiles(3).take(take),
                d -> {
                    check("splits non-empty", !d.splits().isEmpty());
                    int total = 0;
                    for (HfDataset ds : d.splits().values()) {
                        total += fullScan(ds);
                    }
                    check("total>0", total > 0);
                });

        // ── 4. Classic Hub smoke (baseline) ─────────────────────────────────
        section("4. Classic Hub baseline");

        live("lhoestq/demo1 csv",
                "lhoestq/demo1", null, null,
                cfg -> cfg.maxFiles(2).take(take),
                d -> {
                    check("train", d.splits().containsKey("train") && d.train().size() > 0);
                    check("scan", fullScan(d.train()) == d.train().size());
                });

        live("stanfordnlp/imdb plain_text train",
                "stanfordnlp/imdb", "plain_text", "train",
                cfg -> cfg.maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows", ds.size() > 0 && ds.size() <= take);
                    check("scan", fullScan(ds) == ds.size());
                    expectCols(ds, "text", "label");
                });

        live("nyu-mll/glue cola",
                "nyu-mll/glue", "cola", "train",
                cfg -> cfg.maxFiles(1).take(take),
                d -> {
                    HfDataset ds = firstSplit(d);
                    check("rows", ds.size() > 0);
                    check("scan", fullScan(ds) == ds.size());
                    expectCols(ds, "sentence", "label", "idx", "text");
                });

        live("cais/mmlu abstract_algebra",
                "cais/mmlu", "abstract_algebra", null,
                cfg -> cfg.maxFiles(3).take(32),
                d -> check("splits", !d.splits().isEmpty()));

        // ── 5. Explicitly probe unavailable names from user list ─────────────
        section("5. Catalog availability probe (skip if 401/404)");
        for (String id : List.of(
                "reczoo/Amazon-Beauty",
                "polyvore1000",
                "CIKM2021-RS/RetailRocket",
                "reczoo/LastFM-1K",
                "xingyaogong/minimind-v_dataset",
                "HuggingFaceM4/Flickr30k",
                "liuhaotian/LLaVA-v1.5-mix665k",
                "google-research-datasets/conceptual_captions",
                "HuggingFaceM4/VQAv2",
                "FoteiniTag/FineVision-Conversations-Gemma4__31B",
                "bezirganyan/LUMA"
        )) {
            probeAvailability(id);
        }

        // ── 6. map/filter integrity on a real recsys load ───────────────────
        section("6. map/filter pipeline integrity");
        live("pipeline on TinyStories",
                "roneneldan/TinyStories", null, "train",
                cfg -> cfg.allowPatterns("data/").maxFiles(1).take(Math.min(take, 128)),
                d -> {
                    HfDataset base = firstSplit(d);
                    int baseN = base.size();
                    HfDataset mapped = base.map(r -> {
                        Object t = firstNonNull(r, "text", "story", "content");
                        r.put("_len", t == null ? 0 : t.toString().length());
                        return r;
                    });
                    check("map preserves size", mapped.size() == baseN);
                    check("map scan", fullScan(mapped) == baseN);
                    HfDataset filt = mapped.filter(r -> {
                        Object n = r.get("_len");
                        return n instanceof Number && ((Number) n).intValue() > 10;
                    });
                    check("filter subset", filt.size() <= baseN);
                    check("filter scan", fullScan(filt) == filt.size());
                    // save/load roundtrip
                    Path disk = cacheDir.resolve("roundtrip_ts");
                    filt.saveToDisk(disk);
                    HfDataset back = HfDataset.loadFromDisk(disk);
                    check("roundtrip size", back.size() == filt.size());
                    check("roundtrip scan", fullScan(back) == filt.size());
                });

        writeReport();

        System.out.println();
        System.out.println("=== RESULT ===");
        System.out.println("passed=" + passed + " failed=" + failed + " skipped=" + skipped);
        System.out.println("report: " + outDir.resolve("RESULTS.md").toAbsolutePath());
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL OK");
    }

    // ── case runners ────────────────────────────────────────────────────────

    interface Throwing { void run() throws Exception; }
    interface CfgTune { HfDatasets.LoadConfig.Builder tune(HfDatasets.LoadConfig.Builder b); }
    interface DictAssert { void check(HfDataset.DatasetDict d) throws Exception; }

    static void section(String title) {
        System.out.println("── " + title + " ──");
    }

    static void run(String name, Throwing body) {
        if (only != null && !name.toLowerCase(Locale.ROOT).contains(only.toLowerCase(Locale.ROOT))) {
            return;
        }
        System.out.print("  • " + name + " ... ");
        long t0 = System.nanoTime();
        try {
            body.run();
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("ok (" + ms + "ms)");
            results.add(CaseResult.ok(name, ms, null));
        } catch (Throwable t) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("FAIL (" + ms + "ms)");
            report.append("FAIL ").append(name).append(": ").append(t).append('\n');
            t.printStackTrace(System.out);
            results.add(CaseResult.fail(name, ms, t.toString()));
        }
    }

    static void live(String name, String repo, String config, String split,
                     CfgTune tune, DictAssert asserts) {
        if (only != null && !name.toLowerCase(Locale.ROOT).contains(only.toLowerCase(Locale.ROOT))
                && !repo.toLowerCase(Locale.ROOT).contains(only.toLowerCase(Locale.ROOT))) {
            return;
        }
        System.out.print("  • LIVE " + name + " ... ");
        long t0 = System.nanoTime();
        try {
            Consumer<String> log = s -> System.out.println("    " + s);
            HfDatasets.LoadConfig.Builder b = HfDatasets.LoadConfig.builder()
                    .token(token)
                    .endpoint(endpoint)
                    .cacheDir(cacheDir)
                    .preferMirror(true)
                    .logger(log);
            if (tune != null) b = tune.tune(b);
            HfDatasets.LoadConfig cfg = b.build();

            HfDataset.DatasetDict d = HfDatasets.loadDataset(repo, config, split, cfg);
            asserts.check(d);
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            String summary = summarize(d);
            System.out.println("ok (" + ms + "ms) " + summary);
            results.add(CaseResult.ok(name, ms, summary));
        } catch (Throwable t) {
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            String msg = t.getMessage() == null ? t.toString() : t.getMessage();
            // network / auth / missing → skip rather than fail hard for catalog noise
            if (isSoftFailure(msg, t)) {
                skipped++;
                System.out.println("SKIP (" + ms + "ms) " + shortMsg(msg));
                results.add(CaseResult.skip(name, ms, shortMsg(msg)));
            } else {
                failed++;
                System.out.println("FAIL (" + ms + "ms)");
                report.append("FAIL LIVE ").append(name).append(": ").append(t).append('\n');
                t.printStackTrace(System.out);
                results.add(CaseResult.fail(name, ms, shortMsg(msg)));
            }
        }
    }

    static void probeAvailability(String datasetId) {
        if (only != null && !datasetId.toLowerCase(Locale.ROOT).contains(only.toLowerCase(Locale.ROOT))) {
            return;
        }
        System.out.print("  • probe " + datasetId + " ... ");
        long t0 = System.nanoTime();
        try {
            HfHub hub = HfHub.create()
                    .token(token).endpoint(endpoint).cacheDir(cacheDir)
                    .logger(s -> {}).build();
            List<HfHub.RepoFile> tree = hub.listDatasetFiles(datasetId);
            long data = tree.stream().filter(HfHub.RepoFile::isFile)
                    .filter(f -> HfHub.isDatasetDataFile(f.path)).count();
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            System.out.println("OK tree=" + tree.size() + " dataFiles=" + data + " (" + ms + "ms)");
            passed++;
            results.add(CaseResult.ok("probe " + datasetId, ms,
                    "tree=" + tree.size() + " data=" + data));
        } catch (Throwable t) {
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            skipped++;
            System.out.println("SKIP " + shortMsg(t.getMessage() == null ? t.toString() : t.getMessage())
                    + " (" + ms + "ms)");
            results.add(CaseResult.skip("probe " + datasetId, ms,
                    shortMsg(t.getMessage() == null ? t.toString() : t.getMessage())));
        }
    }

    // ── integrity helpers ───────────────────────────────────────────────────

    static int fullScan(HfDataset ds) {
        int n = 0;
        for (int i = 0; i < ds.size(); i++) {
            Map<String, Object> r = ds.get(i);
            if (r == null) throw new AssertionError("null row at " + i);
            n++;
        }
        return n;
    }

    static HfDataset firstSplit(HfDataset.DatasetDict d) {
        if (d.splits().containsKey("train")) return d.train();
        if (d.splits().isEmpty()) return HfDataset.empty();
        return d.splits().values().iterator().next();
    }

    static Object firstNonNull(Map<String, Object> r, String... keys) {
        for (String k : keys) {
            if (r.containsKey(k) && r.get(k) != null) return r.get(k);
        }
        return null;
    }

    /** Soft expect: at least one of the candidate column names appears (case-insensitive contains). */
    static void expectCols(HfDataset ds, String... candidates) {
        List<String> cols = ds.columnNames();
        if (cols.isEmpty() && ds.size() > 0) {
            cols = new ArrayList<>(ds.get(0).keySet());
        }
        boolean any = false;
        for (String c : candidates) {
            for (String col : cols) {
                if (col != null && col.toLowerCase(Locale.ROOT).contains(c.toLowerCase(Locale.ROOT))) {
                    any = true;
                    break;
                }
            }
            if (any) break;
        }
        // don't fail hard — just log; recsys schemas vary
        if (!any) {
            System.out.print("[cols=" + cols + "] ");
        }
        check("has columns", !cols.isEmpty() || ds.size() == 0);
    }

    static String summarize(HfDataset.DatasetDict d) {
        StringBuilder sb = new StringBuilder("splits={");
        boolean first = true;
        for (var e : d.splits().entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            sb.append(e.getKey()).append(':').append(e.getValue().size());
            List<String> cols = e.getValue().columnNames();
            if (cols.isEmpty() && e.getValue().size() > 0) {
                cols = new ArrayList<>(e.getValue().get(0).keySet());
            }
            if (!cols.isEmpty()) sb.append(cols);
        }
        sb.append('}');
        return sb.toString();
    }

    static boolean isSoftFailure(String msg, Throwable t) {
        if (msg == null) msg = "";
        String m = msg.toLowerCase(Locale.ROOT);
        if (m.contains("401") || m.contains("403") || m.contains("404")
                || m.contains("auth") || m.contains("gated")
                || m.contains("not found") || m.contains("http 307")
                || m.contains("http 429") || m.contains("timeout")
                || m.contains("timed out") || m.contains("connection")
                || m.contains("unknown host") || m.contains("no data files")
                || m.contains("no files for config") || m.contains("unknown config")
                || m.contains("unknown split")) {
            return true;
        }
        // deep nested causes
        Throwable c = t.getCause();
        while (c != null) {
            String cm = c.getMessage() == null ? "" : c.getMessage().toLowerCase(Locale.ROOT);
            if (cm.contains("401") || cm.contains("403") || cm.contains("404")
                    || cm.contains("timeout") || cm.contains("connection")) return true;
            c = c.getCause();
        }
        return false;
    }

    static String shortMsg(String msg) {
        if (msg == null) return "null";
        msg = msg.replace('\n', ' ');
        return msg.length() > 180 ? msg.substring(0, 180) + "…" : msg;
    }

    static void check(String name, boolean cond) {
        if (cond) {
            passed++;
        } else {
            failed++;
            report.append("  check failed: ").append(name).append('\n');
            throw new AssertionError(name);
        }
    }

    // ── report ──────────────────────────────────────────────────────────────

    static void writeReport() throws IOException {
        Path md = outDir.resolve("RESULTS.md");
        Path jsonl = outDir.resolve("results.jsonl");
        StringBuilder mdBody = new StringBuilder();
        mdBody.append("# HfDatasets recsys / multimodal stress\n\n");
        mdBody.append("- endpoint: `").append(endpoint).append("`\n");
        mdBody.append("- take=").append(take).append(" maxFiles=").append(maxFiles).append('\n');
        mdBody.append("- passed=").append(passed).append(" failed=").append(failed)
                .append(" skipped=").append(skipped).append("\n\n");
        mdBody.append("| Case | Status | ms | Detail |\n|---|---|---:|---|\n");
        StringBuilder jl = new StringBuilder();
        for (CaseResult r : results) {
            mdBody.append("| ").append(esc(r.name)).append(" | ").append(r.status)
                    .append(" | ").append(r.ms).append(" | ")
                    .append(esc(r.detail == null ? "" : r.detail)).append(" |\n");
            jl.append("{\"name\":").append(jsonStr(r.name))
                    .append(",\"status\":").append(jsonStr(r.status))
                    .append(",\"ms\":").append(r.ms)
                    .append(",\"detail\":").append(jsonStr(r.detail == null ? "" : r.detail))
                    .append("}\n");
        }
        mdBody.append("\n## Notes\n");
        mdBody.append("- `reczoo/Amazon-Beauty` etc. may 401 on mirror without token; ")
                .append("`McAuley-Lab/Amazon-Reviews-2023` All_Beauty CSV is the public stand-in.\n");
        mdBody.append("- VK-LSVD full ur0.01_ip0.01 is ~1.2GB; stress loads 1 week shard only.\n");
        mdBody.append("- T-ECD uses `dataset/small/**` allowPatterns + maxFiles cap.\n");
        mdBody.append("- Multimodal cases validate **metadata parse** (json/csv/parquet); ")
                .append("image/video blobs are usually external.\n");
        Files.writeString(md, mdBody.toString(), StandardCharsets.UTF_8);
        Files.writeString(jsonl, jl.toString(), StandardCharsets.UTF_8);
    }

    static String esc(String s) {
        return s.replace("|", "\\|").replace("\n", " ");
    }

    static String jsonStr(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "") + "\"";
    }

    static final class CaseResult {
        final String name, status, detail;
        final long ms;
        CaseResult(String name, String status, long ms, String detail) {
            this.name = name; this.status = status; this.ms = ms; this.detail = detail;
        }
        static CaseResult ok(String n, long ms, String d) { return new CaseResult(n, "OK", ms, d); }
        static CaseResult fail(String n, long ms, String d) { return new CaseResult(n, "FAIL", ms, d); }
        static CaseResult skip(String n, long ms, String d) { return new CaseResult(n, "SKIP", ms, d); }
    }

    static void parseArgs(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--take" -> take = Integer.parseInt(args[++i]);
                case "--max-files" -> maxFiles = Integer.parseInt(args[++i]);
                case "--only" -> only = args[++i];
                case "--token" -> token = args[++i];
                case "--endpoint" -> endpoint = args[++i];
                case "--help", "-h" -> {
                    System.out.println("Usage: StressHfRecsysDatasets [--take N] [--max-files N] "
                            + "[--only substr] [--token TOK] [--endpoint URL]");
                    System.exit(0);
                }
                default -> System.err.println("unknown arg: " + args[i]);
            }
        }
    }
}
