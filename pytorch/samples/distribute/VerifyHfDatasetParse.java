package distribute;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataLoader;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataset;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameNativeDataset;
import org.bytedeco.pytorch.data.orc.LocalOrcFormatWriter;
import org.bytedeco.pytorch.data.avro.LocalAvroWriter;
import org.bytedeco.pytorch.data.arrow.LocalArrowIpcWriter;
import org.bytedeco.pytorch.data.parquet.LocalParquetWriter;
import org.bytedeco.pytorch.data.parquet.SchemaBuilder;
import org.bytedeco.pytorch.utils.datasets.HfDataset;
import org.bytedeco.pytorch.utils.datasets.HfDatasets;
import org.bytedeco.pytorch.llm.hub.HfToken;
import org.bytedeco.pytorch.data.Example;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * End-to-end verification:
 * <ol>
 *   <li>Print real Hub download samples (glue/cola, imdb, squad) and assert columns/types</li>
 *   <li>Local multi-format roundtrip: csv/tsv/jsonl/json/text/parquet/orc/avro/arrow-ipc</li>
 *   <li>HfDataset → DataFrameDataset → DataLoader / native Dataset</li>
 * </ol>
 *
 * <pre>
 *   export HF_TOKEN=hf_xxx
 *   export HF_ENDPOINT=https://hf-mirror.com
 *   java ... distribute.VerifyHfDatasetParse [--take 8]
 * </pre>
 */
public class VerifyHfDatasetParse {

    static int passed = 0, failed = 0;
    static final StringBuilder failures = new StringBuilder();
    static int take = 8;
    static String token;
    static String endpoint;

    public static void main(String[] args) throws Exception {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--take" -> take = Integer.parseInt(args[++i]);
                case "--token" -> token = args[++i];
                case "--endpoint" -> endpoint = args[++i];
                default -> {}
            }
        }
        token = HfToken.resolve(token);
        if (endpoint == null || endpoint.isBlank()) {
            endpoint = HfToken.resolveEndpoint(true); // prefer mirror when unset
            if (System.getenv("HF_ENDPOINT") == null && System.getenv("HF_MIRROR") == null) {
                endpoint = "https://hf-mirror.com";
            }
        }

        Path tmp = Files.createTempDirectory("hf_verify_");
        System.out.println("=== VerifyHfDatasetParse ===");
        System.out.println("token=" + HfToken.mask(token) + " endpoint=" + endpoint + " take=" + take);
        System.out.println("tmp=" + tmp);
        System.out.println();

        // ── 1. Local multi-format ──────────────────────────────────────────
        section("1. Local multi-format loaders (incl. orc/avro/ipc)");
        Path local = tmp.resolve("local");
        Files.createDirectories(local);

        // CSV / TSV / JSONL / JSON / TEXT
        Path csv = local.resolve("a.csv");
        Files.writeString(csv, "text,label\nhello world,0\nfoo bar,1\n\"a,b\",2\n", StandardCharsets.UTF_8);
        Path tsv = local.resolve("a.tsv");
        Files.writeString(tsv, "text\tlabel\nx\t1\ny\t0\n", StandardCharsets.UTF_8);
        Path jsonl = local.resolve("a.jsonl");
        Files.writeString(jsonl,
                "{\"text\":\"hi\",\"label\":0,\"tags\":[\"a\",\"b\"]}\n"
                        + "{\"text\":\"yo\",\"label\":1,\"meta\":{\"n\":3}}\n",
                StandardCharsets.UTF_8);
        Path json = local.resolve("a.json");
        Files.writeString(json, "[{\"text\":\"p\",\"label\":0},{\"text\":\"q\",\"label\":1}]", StandardCharsets.UTF_8);
        Path txt = local.resolve("a.txt");
        Files.writeString(txt, "line1\nline2\n", StandardCharsets.UTF_8);

        check("csv", () -> {
            HfDataset d = HfDataset.fromCsv(csv, true);
            System.out.print(d.headString(3));
            assertEq("csv rows", 3, d.size());
            assertEq("csv quoted", "a,b", d.get(2).get("text"));
            assertTrue("csv label num", d.get(0).get("label") instanceof Number);
        });
        check("tsv", () -> {
            HfDataset d = HfDataset.fromTsv(tsv, true);
            System.out.print(d.headString(2));
            assertEq("tsv rows", 2, d.size());
        });
        check("jsonl nested", () -> {
            HfDataset d = HfDataset.fromJsonl(jsonl);
            System.out.print(d.headString(2));
            assertEq("jsonl rows", 2, d.size());
            assertTrue("jsonl list", d.get(0).get("tags") instanceof List);
            assertTrue("jsonl map", d.get(1).get("meta") instanceof Map);
        });
        check("json array", () -> {
            HfDataset d = HfDataset.fromJson(json);
            System.out.print(d.headString(2));
            assertEq("json rows", 2, d.size());
        });
        check("text", () -> {
            HfDataset d = HfDataset.fromText(txt);
            System.out.print(d.headString(2));
            assertEq("text rows", 2, d.size());
        });

        // Parquet pure-Java
        Path pq = local.resolve("a.parquet");
        check("parquet write+read", () -> {
            var schema = SchemaBuilder.builder()
                    .optionalString("text").requiredInt64("label").build();
            try (var w = LocalParquetWriter.builder(pq.toString(), schema)
                    .withDictionary(false).build()) {
                var g = w.makeGroup(); g.add("text", "alpha"); g.add("label", 0L); w.write(g);
                g = w.makeGroup(); g.add("text", "beta"); g.add("label", 1L); w.write(g);
            }
            HfDataset d = HfDataset.fromParquet(pq);
            System.out.print(d.headString(2));
            assertEq("pq rows", 2, d.size());
            assertEq("pq text", "alpha", String.valueOf(d.get(0).get("text")));
            assertEq("pq label", 1L, ((Number) d.get(1).get("label")).longValue());
            // fromFile auto
            assertEq("fromFile pq", 2, HfDataset.fromFile(pq).size());
        });

        // Build a small DataFrame for ORC / Avro / Arrow writers
        DataFrame seed = new DataFrame();
        seed.addColumn("text", Column.DType.STRING);
        seed.addColumn("label", Column.DType.INT64);
        seed.column("text").add("orc-a"); seed.column("label").add(0L);
        seed.column("text").add("orc-b"); seed.column("label").add(1L);
        seed.syncRowCountPublic();

        Path orc = local.resolve("a.orc");
        check("orc write+read", () -> {
            LocalOrcFormatWriter.write(seed, orc.toString());
            HfDataset d = HfDataset.fromOrc(orc);
            System.out.print(d.headString(2));
            assertEq("orc rows", 2, d.size());
            assertTrue("orc has text", d.get(0).containsKey("text"));
            assertEq("fromFile orc", 2, HfDataset.fromFile(orc).size());
        });

        Path avro = local.resolve("a.avro");
        check("avro write+read", () -> {
            LocalAvroWriter.write(seed, avro.toString());
            HfDataset d = HfDataset.fromAvro(avro);
            System.out.print(d.headString(2));
            assertEq("avro rows", 2, d.size());
            assertEq("fromFile avro", 2, HfDataset.fromFile(avro).size());
        });

        Path arrow = local.resolve("a.arrow");
        check("arrow ipc write+read", () -> {
            LocalArrowIpcWriter.write(seed, arrow.toString());
            HfDataset d = HfDataset.fromArrow(arrow);
            System.out.print(d.headString(2));
            assertEq("arrow rows", 2, d.size());
            assertEq("fromFile arrow", 2, HfDataset.fromFile(arrow).size());
        });

        check("HfDatasets.load format dispatch", () -> {
            assertEq("load csv", 3, HfDatasets.load("csv", Map.of("data_files", csv.toString())).size());
            assertEq("load tsv", 2, HfDatasets.load("tsv", Map.of("data_files", tsv.toString())).size());
            assertEq("load jsonl", 2, HfDatasets.load("jsonl", Map.of("data_files", jsonl.toString())).size());
            assertEq("load parquet", 2, HfDatasets.load("parquet", Map.of("data_files", pq.toString())).size());
            assertEq("load orc", 2, HfDatasets.load("orc", Map.of("data_files", orc.toString())).size());
            assertEq("load avro", 2, HfDatasets.load("avro", Map.of("data_files", avro.toString())).size());
            assertEq("load ipc", 2, HfDatasets.load("ipc", Map.of("data_files", arrow.toString())).size());
        });

        // ── 2. Live Hub samples (print rows) ───────────────────────────────
        section("2. Live Hub download + parse samples");
        if (token == null || token.isBlank()) {
            System.out.println("  (skip live: no HF_TOKEN)");
        } else {
            Path cache = tmp.resolve("hub_cache");
            var cfg = HfDatasets.LoadConfig.builder()
                    .token(token)
                    .endpoint(endpoint)
                    .cacheDir(cache)
                    .take(take)
                    .maxFiles(2)
                    .logger(s -> System.out.println("  " + s))
                    .build();

            check("LIVE glue/cola", () -> {
                HfDataset.DatasetDict d = HfDatasets.loadDataset("glue", "cola", null, cfg);
                System.out.println("  " + d);
                HfDataset train = d.train();
                System.out.print(train.headString(Math.min(3, take)));
                assertTrue("cola train>0", train.size() > 0);
                Map<String, Object> r0 = train.get(0);
                assertTrue("cola has sentence", r0.containsKey("sentence") || r0.containsKey("text"));
                assertTrue("cola has label", r0.containsKey("label"));
                Object lab = r0.get("label");
                assertTrue("cola label numeric", lab instanceof Number);
                System.out.println("  features=" + train.features());
                System.out.println("  cols=" + train.columnNames());
            });

            check("LIVE imdb train", () -> {
                HfDataset.DatasetDict d = HfDatasets.loadDataset("imdb", "plain_text", "train", cfg);
                HfDataset train = d.train();
                System.out.print(train.headString(2));
                assertTrue("imdb rows", train.size() > 0 && train.size() <= take);
                Map<String, Object> r0 = train.get(0);
                assertTrue("imdb text", r0.containsKey("text"));
                assertTrue("imdb label", r0.containsKey("label"));
                // text should be a long-ish review string
                String text = String.valueOf(r0.get("text"));
                assertTrue("imdb text length", text.length() > 20);
                System.out.println("  text[0] len=" + text.length() + " label=" + r0.get("label"));
            });

            check("LIVE squad validation (nested answers)", () -> {
                HfDataset.DatasetDict d = HfDatasets.loadDataset("squad", null, "validation",
                        HfDatasets.LoadConfig.builder()
                                .token(token).endpoint(endpoint).cacheDir(cache)
                                .take(Math.min(take, 4)).maxFiles(1)
                                .logger(s -> System.out.println("  " + s)).build());
                HfDataset val = d.splits().containsKey("validation") ? d.get("validation")
                        : d.splits().values().iterator().next();
                System.out.print(val.headString(2));
                assertTrue("squad rows", val.size() > 0);
                Map<String, Object> r0 = val.get(0);
                assertTrue("squad question", r0.containsKey("question"));
                assertTrue("squad context", r0.containsKey("context"));
                assertTrue("squad answers", r0.containsKey("answers"));
                Object ans = r0.get("answers");
                System.out.println("  answers class=" + (ans == null ? "null" : ans.getClass().getName()));
                System.out.println("  answers value=" + truncate(String.valueOf(ans), 200));
                // answers is typically a struct/map with text + answer_start lists
                assertTrue("answers structured", ans instanceof Map || ans instanceof List);
            });

            check("LIVE lhoestq/demo1 csv", () -> {
                HfDataset.DatasetDict d = HfDatasets.loadDataset("lhoestq/demo1", null, null,
                        HfDatasets.LoadConfig.builder()
                                .token(token).endpoint(endpoint).cacheDir(cache)
                                .logger(s -> System.out.println("  " + s)).build());
                System.out.println("  " + d);
                System.out.print(d.train().headString(3));
                assertTrue("demo1 train", d.train().size() > 0);
            });
        }

        // ── 3. Dataset / DataLoader interop ────────────────────────────────
        section("3. HfDataset ↔ javacpp Dataset / DataLoader");
        check("asDataFrameDataset + pure-Java dataloader", () -> {
            // numeric features + label (string text excluded from float pack)
            HfDataset raw = HfDataset.fromList(List.of(
                    Map.of("f1", 1.0, "f2", 2.0, "label", 0),
                    Map.of("f1", 3.0, "f2", 4.0, "label", 1),
                    Map.of("f1", 5.0, "f2", 6.0, "label", 0),
                    Map.of("f1", 7.0, "f2", 8.0, "label", 1)
            ));
            DataFrameDataset dfd = raw.asDataFrameDataset();
            System.out.println("  dfd size=" + dfd.size()
                    + " scalars=" + java.util.Arrays.toString(dfd.scalarFeatureNames())
                    + " labels=" + java.util.Arrays.toString(dfd.labelNames()));
            assertEq("dfd size", 4, dfd.size());
            assertTrue("has labels", dfd.labelNames().length > 0);

            DataFrameDataLoader loader = dfd.dataloader().batchSize(2).shuffle(false).build();
            int batches = 0;
            for (DataFrameDataLoader.Batch b : loader) {
                batches++;
                System.out.println("  batch#" + batches
                        + " features=" + b.features()
                        + " labels=" + b.labels());
                assertTrue("batch features non-null", b.features() != null);
                assertTrue("batch labels non-null", b.labels() != null);
            }
            assertEq("num batches", 2, batches);
        });

        check("asDataset native Example get()", () -> {
            HfDataset raw = HfDataset.fromList(List.of(
                    Map.of("x", 1.5, "label", 0),
                    Map.of("x", 2.5, "label", 1)
            ));
            DataFrameNativeDataset nativeDs = raw.asDataset();
            assertTrue("native size", nativeDs.size().has_value() && nativeDs.size().get() == 2);
            Example ex0 = nativeDs.get(0);
            System.out.println("  example0 data=" + ex0.data() + " target=" + ex0.target());
            assertTrue("data dim", ex0.data().numel() > 0);
            assertTrue("target dim", ex0.target().numel() > 0);
        });

        check("encodeColumn for text labels → dataloader", () -> {
            HfDataset raw = HfDataset.fromList(List.of(
                    Map.of("score", 0.1, "label_text", "neg"),
                    Map.of("score", 0.9, "label_text", "pos"),
                    Map.of("score", 0.2, "label_text", "neg")
            ));
            // encode string labels to ids, use score as feature
            HfDataset enc = raw.encodeColumn("label_text", "label");
            System.out.print(enc.headString(3));
            DataFrameDataset dfd = enc.asDataFrameDataset(
                    new String[]{"score"}, new String[]{"label"});
            DataFrameDataLoader loader = dfd.dataloader(2);
            int n = 0;
            for (DataFrameDataLoader.Batch b : loader) {
                n++;
                System.out.println("  enc batch features=" + b.features() + " labels=" + b.labels());
            }
            assertTrue("enc batches", n >= 1);
        });

        check("glue-like path: live or local → asDataset", () -> {
            HfDataset ds;
            if (token != null && !token.isBlank()) {
                // reuse tiny local parquet if live already tested
                ds = HfDataset.fromParquet(pq);
            } else {
                ds = HfDataset.fromParquet(pq);
            }
            // encode is not needed — label is already numeric
            DataFrameDataset dfd = ds.asDataFrameDataset(
                    null, new String[]{"label"});
            System.out.println("  size=" + dfd.size()
                    + " features=" + java.util.Arrays.toString(dfd.scalarFeatureNames())
                    + " labels=" + java.util.Arrays.toString(dfd.labelNames()));
            // string-only features may yield empty scalar pack — still valid Dataset
            DataFrameNativeDataset nd = dfd.asDataset();
            assertTrue("nd size", nd.size().has_value());
            System.out.println("  native size=" + nd.size().get());
        });

        System.out.println();
        System.out.println("=== RESULT passed=" + passed + " failed=" + failed + " ===");
        if (failed > 0) {
            System.out.println(failures);
            System.exit(1);
        }
        System.out.println("ALL OK — samples printed above confirm parse correctness.");
    }

    // ---- helpers -----------------------------------------------------------

    static void section(String t) { System.out.println("── " + t + " ──"); }

    interface Throwing { void run() throws Exception; }

    static void check(String name, Throwing r) {
        System.out.println("• " + name);
        try {
            r.run();
            System.out.println("  → ok");
        } catch (Throwable t) {
            failed++;
            failures.append("FAIL ").append(name).append(": ").append(t).append('\n');
            System.out.println("  → FAIL " + t);
            t.printStackTrace(System.out);
        }
    }

    static void assertEq(String name, Object exp, Object act) {
        boolean ok = exp == null ? act == null : exp.equals(act);
        if (!ok && exp instanceof Number && act instanceof Number) {
            ok = ((Number) exp).doubleValue() == ((Number) act).doubleValue();
        }
        if (ok) passed++;
        else {
            failed++;
            String msg = name + " expected=" + exp + " actual=" + act;
            failures.append(msg).append('\n');
            throw new AssertionError(msg);
        }
    }

    static void assertTrue(String name, boolean cond) {
        if (cond) passed++;
        else {
            failed++;
            failures.append(name).append('\n');
            throw new AssertionError(name);
        }
    }

    static String truncate(String s, int n) {
        if (s == null) return "null";
        return s.length() <= n ? s : s.substring(0, n) + "...";
    }
}
