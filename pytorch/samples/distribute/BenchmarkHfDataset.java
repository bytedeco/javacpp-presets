package distribute;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.utils.datasets.HfDataset;
import org.bytedeco.pytorch.utils.datasets.HfDatasets;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.hub.HfToken;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Multi-dimensional stress / parity benchmark for HuggingFace datasets support.
 *
 * <p>Covers:
 * <ul>
 *   <li>Token + endpoint resolution</li>
 *   <li>Local multi-format loaders (csv/tsv/json/jsonl/text/parquet via DataFrame)</li>
 *   <li>Dataset layout inference (glue/imdb/mmlu-style paths)</li>
 *   <li>map/filter/split/shard/save-load roundtrip</li>
 *   <li>Live Hub downloads (parquet configs/splits) when network + token available</li>
 * </ul>
 *
 * <pre>
 *   export HF_TOKEN=hf_xxx
 *   export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
 *   java ... distribute.BenchmarkHfDataset [--live] [--mirror] [--take N]
 * </pre>
 */
public class BenchmarkHfDataset {

    static int passed = 0;
    static int failed = 0;
    static int skipped = 0;
    static final StringBuilder report = new StringBuilder();
    static Path tmp;
    static boolean live = false;
    static boolean preferMirror = false;
    static int take = 64;
    static String token;
    static String endpoint;

    public static void main(String[] args) throws Exception {
        parseArgs(args);
        token = HfToken.resolve(token);
        if (endpoint == null || endpoint.isBlank()) {
            endpoint = HfToken.resolveEndpoint(preferMirror);
        }
        tmp = Files.createTempDirectory("hfds_bench_");
        System.out.println("=== HuggingFace Datasets Benchmark ===");
        System.out.println("tmp=" + tmp);
        System.out.println("token=" + HfToken.mask(token)
                + " endpoint=" + endpoint
                + " live=" + live
                + " take=" + take);
        System.out.println();

        try {
            // ── 1. Token / endpoint ────────────────────────────────────────
            section("1. Token & endpoint resolution");
            benchmark("HfToken.resolve non-null or explicit", () -> {
                // may be null offline; just ensure API doesn't throw
                String t = HfToken.resolve();
                check("resolve() callable", true);
                check("mask(null)", HfToken.mask(null).contains("none"));
                check("mask(short)", HfToken.mask("abcd").equals("****"));
                check("mask(long)", HfToken.mask("hf_abcdefghijklmnop").startsWith("hf_"));
                check("endpoint non-blank", endpoint != null && !endpoint.isBlank());
                check("defaultHub builds", HfToken.defaultHub() != null);
            });

            // ── 2. In-memory ops ───────────────────────────────────────────
            section("2. In-memory HfDataset ops");
            benchmark("fakeText + map/filter/split/shard", () -> {
                HfDataset ds = HfDataset.fakeText(100, 42L);
                check("size=100", ds.size() == 100);
                check("cols has text", ds.columnNames().contains("text"));
                HfDataset mapped = ds.map(r -> {
                    r.put("len", r.get("text").toString().length());
                    return r;
                });
                check("map adds len", mapped.columnNames().contains("len") || mapped.get(0).containsKey("len"));
                HfDataset filt = mapped.filter(r -> ((Number) r.get("label")).intValue() == 0);
                check("filter reduced", filt.size() < 100 && filt.size() > 0);
                HfDataset.DatasetDict split = ds.trainTestSplit(0.2, 7L);
                check("train+test=100", split.train().size() + split.test().size() == 100);
                check("test ~20", Math.abs(split.test().size() - 20) <= 1);
                HfDataset sh0 = ds.shard(4, 0);
                HfDataset sh1 = ds.shard(4, 1);
                check("shard sizes", sh0.size() + sh1.size() + ds.shard(4, 2).size() + ds.shard(4, 3).size() == 100);
                HfDataset taken = ds.take(10).skip(2);
                check("take/skip", taken.size() == 8);
            });

            benchmark("fromDict / fromList / concatenate / rename", () -> {
                Map<String, List<?>> cols = new LinkedHashMap<>();
                cols.put("a", List.of(1, 2, 3));
                cols.put("b", List.of("x", "y", "z"));
                HfDataset d1 = HfDataset.fromDict(cols);
                check("fromDict size", d1.size() == 3);
                check("fromDict a0", ((Number) d1.get(0).get("a")).intValue() == 1);
                HfDataset d2 = HfDataset.fromList(List.of(
                        Map.of("a", 4, "b", "w"),
                        Map.of("a", 5, "b", "v")
                ));
                HfDataset cat = HfDataset.concatenate(d1, d2);
                check("concat size", cat.size() == 5);
                HfDataset ren = cat.renameColumn("a", "id");
                check("rename", ren.columnNames().contains("id") && !ren.get(0).containsKey("a"));
                HfDataset sel = ren.selectColumns("id");
                check("selectColumns", sel.columnNames().size() == 1 && sel.get(0).containsKey("id"));
            });

            // ── 3. Local multi-format I/O ──────────────────────────────────
            section("3. Local multi-format loaders");
            Path dataDir = tmp.resolve("local");
            Files.createDirectories(dataDir);

            benchmark("CSV / TSV roundtrip", () -> {
                Path csv = dataDir.resolve("tiny.csv");
                Files.writeString(csv, "text,label\nhello,0\nworld,1\n\"a,b\",2\n", StandardCharsets.UTF_8);
                HfDataset ds = HfDataset.fromCsv(csv, true);
                check("csv rows", ds.size() == 3);
                check("csv quoted comma", "a,b".equals(ds.get(2).get("text")));
                check("csv autoType label", ds.get(0).get("label") instanceof Number);

                Path tsv = dataDir.resolve("tiny.tsv");
                Files.writeString(tsv, "text\tlabel\nfoo\t1\nbar\t0\n", StandardCharsets.UTF_8);
                HfDataset ts = HfDataset.fromTsv(tsv, true);
                check("tsv rows", ts.size() == 2);
                check("tsv col", "foo".equals(ts.get(0).get("text")));
            });

            benchmark("JSONL / JSON / TEXT", () -> {
                Path jsonl = dataDir.resolve("tiny.jsonl");
                Files.writeString(jsonl,
                        "{\"text\":\"hi\",\"label\":0}\n{\"text\":\"yo\",\"label\":1,\"meta\":{\"n\":2}}\n",
                        StandardCharsets.UTF_8);
                HfDataset jl = HfDataset.fromJsonl(jsonl);
                check("jsonl rows", jl.size() == 2);
                check("jsonl nested", jl.get(1).get("meta") instanceof Map);

                Path json = dataDir.resolve("tiny.json");
                Files.writeString(json,
                        "[{\"text\":\"a\",\"label\":0},{\"text\":\"b\",\"label\":1}]",
                        StandardCharsets.UTF_8);
                HfDataset ja = HfDataset.fromJson(json);
                check("json array rows", ja.size() == 2);

                Path txt = dataDir.resolve("tiny.txt");
                Files.writeString(txt, "line one\nline two\nline three\n", StandardCharsets.UTF_8);
                HfDataset tx = HfDataset.fromText(txt);
                check("text rows", tx.size() == 3);
                check("text col", "line one".equals(tx.get(0).get("text")));
            });

            benchmark("fromFile auto-detect + save/load disk", () -> {
                Path jsonl = dataDir.resolve("auto.jsonl");
                Files.writeString(jsonl, "{\"x\":1}\n{\"x\":2}\n", StandardCharsets.UTF_8);
                HfDataset a = HfDataset.fromFile(jsonl);
                check("fromFile jsonl", a.size() == 2);
                Path disk = dataDir.resolve("saved_ds");
                a.saveToDisk(disk);
                HfDataset b = HfDataset.loadFromDisk(disk);
                check("loadFromDisk size", b.size() == 2);
                check("loadFromDisk value", ((Number) b.get(1).get("x")).intValue() == 2);
            });

            benchmark("Parquet pure-Java LocalParquetWriter → fromParquet", () -> {
                // Pure-Java write (no DataFrame / Tensor natives)
                Path pq = dataDir.resolve("tiny.parquet");
                org.apache.parquet.schema.MessageType schema =
                        org.bytedeco.pytorch.data.parquet.SchemaBuilder.builder()
                                .optionalString("text")
                                .requiredInt64("label")
                                .build();
                try (var w = org.bytedeco.pytorch.data.parquet.LocalParquetWriter
                        .builder(pq.toString(), schema)
                        .withCompression(org.apache.parquet.hadoop.metadata.CompressionCodecName.UNCOMPRESSED)
                        .withDictionary(false)
                        .build()) {
                    String[] texts = {"alpha", "beta", "gamma"};
                    long[] labels = {0L, 1L, 0L};
                    for (int i = 0; i < texts.length; i++) {
                        var g = w.makeGroup();
                        g.add("text", texts[i]);
                        g.add("label", labels[i]);
                        w.write(g);
                    }
                }
                check("parquet written", Files.isRegularFile(pq) && Files.size(pq) > 0);

                HfDataset ds = HfDataset.fromParquet(pq);
                check("parquet rows", ds.size() == 3);
                check("parquet text", "alpha".equals(String.valueOf(ds.get(0).get("text"))));
                check("parquet cols", ds.columnNames().contains("text") && ds.columnNames().contains("label"));
                check("parquet label type", ds.get(1).get("label") instanceof Number
                        && ((Number) ds.get(1).get("label")).longValue() == 1L);

                // fromFile auto
                HfDataset auto = HfDataset.fromFile(pq);
                check("fromFile parquet", auto.size() == 3);

                // Optional DataFrame bridge — skip if natives unavailable
                try {
                    var df2 = ds.toDataFrame();
                    check("toDataFrame rows", df2.rowCount() == 3);
                    HfDataset back = HfDataset.fromDataFrame(df2);
                    check("fromDataFrame", back.size() == 3);
                } catch (NoClassDefFoundError | ExceptionInInitializerError e) {
                    System.out.print("(df bridge skipped: " + e.getClass().getSimpleName() + ") ");
                }
            });

            benchmark("HfDatasets.load(format, data_files)", () -> {
                Path csv = dataDir.resolve("load_fmt.csv");
                Files.writeString(csv, "q,a\none,1\ntwo,2\n", StandardCharsets.UTF_8);
                HfDataset ds = HfDatasets.load("csv", Map.of("data_files", csv.toString(), "take", 1));
                check("load csv take", ds.size() == 1);

                Path jl = dataDir.resolve("load_fmt.jsonl");
                Files.writeString(jl, "{\"q\":\"x\"}\n{\"q\":\"y\"}\n", StandardCharsets.UTF_8);
                HfDataset ds2 = HfDatasets.load("jsonl", Map.of("data_files", jl.toString()));
                check("load jsonl", ds2.size() == 2);
            });

            // ── 4. Layout inference ────────────────────────────────────────
            section("4. DatasetLayout inference (glue/imdb/mmlu-style)");
            benchmark("layout glue-style multi-config", () -> {
                List<String> files = List.of(
                        "cola/train-00000-of-00001.parquet",
                        "cola/validation-00000-of-00001.parquet",
                        "cola/test-00000-of-00001.parquet",
                        "mrpc/train-00000-of-00001.parquet",
                        "mrpc/validation-00000-of-00001.parquet",
                        "ax/test-00000-of-00001.parquet"
                );
                var layout = HfDatasets.DatasetLayout.infer(files);
                check("configs has cola", layout.configs().contains("cola"));
                check("configs has mrpc", layout.configs().contains("mrpc"));
                check("cola splits", layout.splits("cola").contains("train")
                        && layout.splits("cola").contains("validation")
                        && layout.splits("cola").contains("test"));
                List<String> colaTrain = layout.filesFor("cola", List.of("train"));
                check("cola train files", colaTrain.size() == 1 && colaTrain.get(0).contains("cola/train"));
            });

            benchmark("layout imdb plain_text + mmlu subject", () -> {
                var imdb = HfDatasets.DatasetLayout.infer(List.of(
                        "plain_text/train-00000-of-00001.parquet",
                        "plain_text/test-00000-of-00001.parquet",
                        "plain_text/unsupervised-00000-of-00001.parquet"
                ));
                check("imdb config plain_text", imdb.configs().contains("plain_text"));
                check("imdb unsupervised", imdb.splits("plain_text").contains("unsupervised"));

                var mmlu = HfDatasets.DatasetLayout.infer(List.of(
                        "abstract_algebra/test-00000-of-00001.parquet",
                        "abstract_algebra/dev-00000-of-00001.parquet",
                        "all/test-00000-of-00001.parquet"
                ));
                check("mmlu subject config", mmlu.configs().contains("abstract_algebra"));
                // "dev" stays distinct from validation (MMLU few-shot prompts)
                check("mmlu keeps dev", mmlu.splits("abstract_algebra").contains("dev"));
            });

            benchmark("layout data/ csv + ultrachat train_sft", () -> {
                var demo = HfDatasets.DatasetLayout.infer(List.of(
                        "data/train.csv",
                        "data/test.csv"
                ));
                check("csv config default", demo.configs().contains("default"));
                check("csv splits", demo.splits("default").contains("train")
                        && demo.splits("default").contains("test"));

                var ultra = HfDatasets.DatasetLayout.infer(List.of(
                        "data/train_sft-00000-of-00003-aaa.parquet",
                        "data/train_sft-00001-of-00003-bbb.parquet",
                        "data/test_sft-00000-of-00001-ccc.parquet"
                ));
                check("ultrachat train_sft", ultra.splits("default").contains("train_sft")
                        || ultra.splits().contains("train_sft"));
            });

            benchmark("glob / pattern matching", () -> {
                check("*.parquet", HfHub.matchPattern("cola/train-00000.parquet", "*.parquet")
                        || HfHub.matchPattern("cola/train-00000.parquet", "**/*.parquet")
                        || HfHub.matchPattern("cola/train-00000.parquet", "cola/*"));
                check("cola/*", HfHub.matchPattern("cola/train.parquet", "cola/*"));
                check("prefix dir", HfHub.matchPattern("cola/train.parquet", "cola/"));
                check("neg", !HfHub.matchPattern("mrpc/train.parquet", "cola/*"));
            });

            // ── 5. Offline seeded hub dataset ──────────────────────────────
            section("5. Offline seedLocal dataset snapshot");
            benchmark("seedLocal datasets + loadLocal", () -> {
                Path cacheRoot = tmp.resolve("hf_cache");
                HfHub hub = HfHub.create()
                        .cacheDir(cacheRoot)
                        .offline(true)
                        .token(token)
                        .endpoint(endpoint)
                        .logger(System.out::println)
                        .build();
                // seed a mini glue/cola-like layout as text CSV (offline)
                Map<String, String> files = new LinkedHashMap<>();
                files.put("cola/train.csv", "sentence,label\nThis is good.,1\nThis is bad.,0\nA third.,1\n");
                files.put("cola/validation.csv", "sentence,label\nVal one.,0\n");
                files.put("cola/test.csv", "sentence,label\nTest one.,1\n");
                files.put("README.md", "---\npretty_name: tiny-cola\n---\n");
                Path snap = hub.seedLocal("datasets", "bench/tiny-cola", "main", files);
                check("seeded snapshot exists", Files.isDirectory(snap));

                // load via local path materialisation
                HfDataset.DatasetDict dict = HfDatasets.loadLocal(snap, "cola", null,
                        HfDatasets.LoadConfig.builder()
                                .offline(true)
                                .cacheDir(cacheRoot)
                                .logger(System.out::println)
                                .build());
                check("offline splits", dict.splits().containsKey("train")
                        && dict.splits().containsKey("validation")
                        && dict.splits().containsKey("test"));
                check("offline train rows", dict.train().size() == 3);
                check("offline val rows", dict.validation().size() == 1);
            });

            // ── 6. Live Hub (optional) ─────────────────────────────────────
            section("6. Live Hub downloads");
            if (!live) {
                skip("live hub tests (pass --live to enable)");
            } else if (token == null || token.isBlank()) {
                skip("live hub tests (no HF_TOKEN)");
            } else {
                Path liveCache = tmp.resolve("live_cache");
                Consumer<String> log = s -> System.out.println("  " + s);
                HfDatasets.LoadConfig cfg = HfDatasets.LoadConfig.builder()
                        .token(token)
                        .endpoint(endpoint)
                        .cacheDir(liveCache)
                        .take(take)
                        .maxFiles(4)
                        .logger(log)
                        .build();

                liveCase("listConfigs glue", () -> {
                    List<String> configs = HfDatasets.listConfigs("glue", cfg);
                    check("glue has cola", configs.stream().anyMatch(c -> c.equalsIgnoreCase("cola")));
                    check("glue multi-config", configs.size() >= 5);
                    System.out.println("    configs=" + configs);
                });

                liveCase("listSplits imdb", () -> {
                    List<String> splits = HfDatasets.listSplits("imdb", null, cfg);
                    check("imdb has train", splits.contains("train"));
                    check("imdb has test", splits.contains("test"));
                    System.out.println("    splits=" + splits);
                });

                liveCase("load_dataset lhoestq/demo1 (csv)", () -> {
                    // small public CSV dataset
                    HfDataset.DatasetDict d = HfDatasets.loadDataset(
                            "lhoestq/demo1", null, null,
                            HfDatasets.LoadConfig.builder()
                                    .token(token).endpoint(endpoint).cacheDir(liveCache)
                                    .logger(log).build());
                    check("demo1 has train", d.splits().containsKey("train"));
                    check("demo1 train rows>0", d.train().size() > 0);
                    check("demo1 cols", !d.train().columnNames().isEmpty());
                    System.out.println("    " + d + " cols=" + d.train().columnNames());
                });

                liveCase("load_dataset glue/cola parquet take=" + take, () -> {
                    HfDataset.DatasetDict d = HfDatasets.loadDataset("glue", "cola", null, cfg);
                    check("cola train", d.splits().containsKey("train") && d.train().size() > 0);
                    check("cola validation", d.splits().containsKey("validation"));
                    // typical cola columns: sentence, label, idx
                    Map<String, Object> row0 = d.train().get(0);
                    check("cola has sentence or text",
                            row0.containsKey("sentence") || row0.containsKey("text") || row0.size() >= 2);
                    System.out.println("    " + d + " sample=" + row0);
                });

                liveCase("load_dataset imdb train (HfDataset)", () -> {
                    // download only train shard(s)
                    HfDataset.DatasetDict d = HfDatasets.loadDataset("imdb", "plain_text", "train",
                            HfDatasets.LoadConfig.builder()
                                    .token(token).endpoint(endpoint).cacheDir(liveCache)
                                    .take(take).maxFiles(1).logger(log).build());
                    HfDataset train = d.splits().containsKey("train") ? d.train()
                            : d.splits().values().iterator().next();
                    check("imdb train rows", train.size() > 0 && train.size() <= take);
                    check("imdb text col", train.get(0).containsKey("text") || train.get(0).containsKey("label"));
                    System.out.println("    rows=" + train.size() + " sample=" + train.get(0));
                });

                liveCase("load_dataset squad validation", () -> {
                    HfDataset.DatasetDict d = HfDatasets.loadDataset("squad", null, "validation",
                            HfDatasets.LoadConfig.builder()
                                    .token(token).endpoint(endpoint).cacheDir(liveCache)
                                    .take(Math.min(take, 32)).maxFiles(1).logger(log).build());
                    HfDataset val = d.splits().containsKey("validation") ? d.get("validation")
                            : d.splits().values().iterator().next();
                    check("squad val rows", val.size() > 0);
                    Map<String, Object> r = val.get(0);
                    check("squad has answers or context",
                            r.containsKey("answers") || r.containsKey("context") || r.containsKey("question"));
                    System.out.println("    rows=" + val.size() + " cols=" + val.columnNames());
                });

                liveCase("load_dataset cais/mmlu abstract_algebra", () -> {
                    HfDataset.DatasetDict d = HfDatasets.loadDataset("cais/mmlu", "abstract_algebra", null,
                            HfDatasets.LoadConfig.builder()
                                    .token(token).endpoint(endpoint).cacheDir(liveCache)
                                    .take(16).maxFiles(3).logger(log).build());
                    check("mmlu splits non-empty", !d.splits().isEmpty());
                    System.out.println("    " + d);
                });

                liveCase("load_dataset nyu-mll/glue mrpc (alias path)", () -> {
                    HfDataset.DatasetDict d = HfDatasets.loadDataset("nyu-mll/glue", "mrpc", "train",
                            HfDatasets.LoadConfig.builder()
                                    .token(token).endpoint(endpoint).cacheDir(liveCache)
                                    .take(take).maxFiles(1).logger(log).build());
                    check("mrpc train", d.splits().containsKey("train") && d.train().size() > 0);
                    System.out.println("    " + d + " sample=" + d.train().get(0));
                });

                liveCase("tree API pagination / large tree mmlu", () -> {
                    HfHub hub = HfHub.create()
                            .token(token).endpoint(endpoint).cacheDir(liveCache)
                            .logger(log).build();
                    List<HfHub.RepoFile> tree = hub.listDatasetFiles("cais/mmlu");
                    check("mmlu tree large", tree.size() > 50);
                    long dataFiles = tree.stream().filter(HfHub.RepoFile::isFile)
                            .filter(f -> HfHub.isDatasetDataFile(f.path)).count();
                    check("mmlu data files", dataFiles > 50);
                    System.out.println("    tree=" + tree.size() + " dataFiles=" + dataFiles);
                });

                liveCase("datasetInfo imdb", () -> {
                    HfHub hub = HfHub.create()
                            .token(token).endpoint(endpoint).cacheDir(liveCache)
                            .logger(log).build();
                    Map<String, Object> info = hub.datasetInfo("imdb");
                    check("info has id or siblings", info.containsKey("id") || info.containsKey("siblings")
                            || info.containsKey("configs"));
                    System.out.println("    keys=" + info.keySet());
                });

                // stress: map/filter pipeline on real data
                liveCase("map/filter pipeline on glue/cola", () -> {
                    HfDataset.DatasetDict d = HfDatasets.loadDataset("glue", "cola", "train",
                            HfDatasets.LoadConfig.builder()
                                    .token(token).endpoint(endpoint).cacheDir(liveCache)
                                    .take(take).maxFiles(1).logger(log).build());
                    HfDataset train = d.train();
                    HfDataset mapped = train.map(r -> {
                        Object s = r.get("sentence");
                        if (s == null) s = r.get("text");
                        r.put("char_len", s == null ? 0 : s.toString().length());
                        return r;
                    });
                    HfDataset longOnes = mapped.filter(r -> {
                        Object n = r.get("char_len");
                        return n instanceof Number && ((Number) n).intValue() > 20;
                    });
                    check("pipeline rows", mapped.size() == train.size());
                    check("filter subset", longOnes.size() <= mapped.size());
                    try {
                        var df = mapped.toDataFrame();
                        check("df rows", df.rowCount() == mapped.size());
                        System.out.println("    mapped=" + mapped.size() + " long=" + longOnes.size()
                                + " dfCols=" + df.getColumnNames());
                    } catch (NoClassDefFoundError | ExceptionInInitializerError e) {
                        System.out.println("    mapped=" + mapped.size() + " long=" + longOnes.size()
                                + " (df bridge skipped)");
                    }
                });
            }

            // ── 7. Edge cases / robustness ─────────────────────────────────
            section("7. Edge cases");
            benchmark("empty / bad inputs", () -> {
                check("empty size", HfDataset.empty().size() == 0);
                check("fromList null", HfDataset.fromList(null).size() == 0);
                try {
                    HfDataset.fakeText(5, 1).shard(0, 0);
                    check("shard numShards=0 throws", false);
                } catch (IllegalArgumentException e) {
                    check("shard numShards=0 throws", true);
                }
                try {
                    HfDataset.fakeText(5, 1).trainTestSplit(1.5, 1);
                    check("bad testSize throws", false);
                } catch (IllegalArgumentException e) {
                    check("bad testSize throws", true);
                }
            });

            benchmark("features() + select indices", () -> {
                HfDataset ds = HfDataset.fakeText(10, 3);
                Map<String, Object> feats = ds.features();
                check("features non-empty", !feats.isEmpty());
                HfDataset sel = ds.select(0, 2, 4);
                check("select 3", sel.size() == 3);
            });

        } finally {
            // leave tmp for inspection on failure; delete on full pass
            if (failed == 0) {
                try {
                    deleteRecursive(tmp);
                } catch (Exception ignored) {}
            } else {
                System.out.println("\n(tmp kept for inspection: " + tmp + ")");
            }
        }

        System.out.println();
        System.out.println("=== RESULT ===");
        System.out.println("passed=" + passed + " failed=" + failed + " skipped=" + skipped);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL OK");
    }

    // ---- helpers -----------------------------------------------------------

    static void parseArgs(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--live" -> live = true;
                case "--mirror" -> preferMirror = true;
                case "--take" -> take = Integer.parseInt(args[++i]);
                case "--token" -> token = args[++i];
                case "--endpoint" -> endpoint = args[++i];
                case "--help", "-h" -> {
                    System.out.println("Usage: BenchmarkHfDataset [--live] [--mirror] [--take N] [--token TOK] [--endpoint URL]");
                    System.exit(0);
                }
                default -> System.err.println("unknown arg: " + args[i]);
            }
        }
        // auto-enable live if token present and user didn't forbid
        if (!live && HfToken.resolve(token) != null) {
            // keep offline-by-default unless --live; user asked for stress with token so default live on when token set
            live = true;
        }
    }

    static void section(String title) {
        System.out.println("── " + title + " ──");
    }

    interface ThrowingRunnable { void run() throws Exception; }

    static void benchmark(String name, ThrowingRunnable r) {
        System.out.print("  • " + name + " ... ");
        try {
            r.run();
            System.out.println("ok");
        } catch (Throwable t) {
            failed++;
            System.out.println("FAIL");
            report.append("FAIL ").append(name).append(": ").append(t).append('\n');
            t.printStackTrace(System.out);
        }
    }

    static void liveCase(String name, ThrowingRunnable r) {
        System.out.print("  • LIVE " + name + " ... ");
        try {
            r.run();
            System.out.println("ok");
        } catch (Throwable t) {
            failed++;
            System.out.println("FAIL");
            report.append("FAIL LIVE ").append(name).append(": ").append(t).append('\n');
            t.printStackTrace(System.out);
        }
    }

    static void skip(String reason) {
        skipped++;
        System.out.println("  ↷ skip " + reason);
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

    static void deleteRecursive(Path p) throws IOException {
        if (p == null || !Files.exists(p)) return;
        try (var walk = Files.walk(p)) {
            List<Path> paths = new ArrayList<>();
            walk.sorted((a, b) -> b.compareTo(a)).forEach(paths::add);
            for (Path x : paths) Files.deleteIfExists(x);
        }
    }
}
