package dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.feature.imputation.SimpleImputer;
import org.bytedeco.pytorch.dataframe.feature.pipeline.DataFramePipeline;
import org.bytedeco.pytorch.dataframe.feature.pipeline.Pipeline;
import org.bytedeco.pytorch.dataframe.feature.scaling.StandardScaler;
import org.bytedeco.pytorch.dataframe.json.JsonOptions;
import org.bytedeco.pytorch.dataframe.ml.classification.LogisticRegression;
import org.bytedeco.pytorch.data.json.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * JSON / JSONL DataFrame I/O + feature pipeline correctness suite.
 *
 * <pre>
 *   java --add-opens=java.base/java.nio=ALL-UNNAMED -cp ... dataframe.BenchmarkDataFrameJson
 * </pre>
 */
public class BenchmarkDataFrameJson {
    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e);
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) passed++;
        else {
            failed++;
            report.append("  check failed: ").append(name).append('\n');
            throw new AssertionError(name);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("=== BenchmarkDataFrameJson ===");
        Path tmp = Files.createTempDirectory("df-json-");

        try {
            // ---- pure JSON module ----
            benchmark("1. Json.parse object + path", () -> {
                JsonValue v = Json.parse("{\"a\":1,\"b\":{\"c\":[true,null,\"x\"]}}");
                check("a long", v.at("a").asLong() == 1L);
                check("b.c[0]", v.at("b.c[0]").asBoolean());
                check("b.c[2]", "x".equals(v.at("b.c[2]").asString()));
                check("valid", Json.isValid("{\"ok\":true}"));
                check("invalid", !Json.isValid("{nope"));
            });

            benchmark("2. escapes / unicode / numbers", () -> {
                JsonValue v = Json.parse("{\"s\":\"a\\\"b\\\\c\\n\",\"n\":1.5e2,\"i\":9223372036854775807}");
                check("escape quote", v.get("s").asString().contains("\""));
                check("sci", Math.abs(v.get("n").asDouble() - 150.0) < 1e-9);
                check("big long", v.get("i").asLong() == 9223372036854775807L);
                String pretty = v.toPrettyString();
                check("pretty has newline", pretty.contains("\n"));
                JsonValue round = Json.parse(pretty);
                check("roundtrip i", round.get("i").asLong() == 9223372036854775807L);
            });

            benchmark("3. lenient comments + trailing commas", () -> {
                String text = "{\n  // line comment\n  \"a\": 1, /* block */ \"b\": [1,2,],\n}\n";
                JsonValue v = Json.parse(text, JsonReadOptions.lenient());
                check("a", v.get("a").asLong() == 1);
                check("b size", v.get("b").size() == 2);
            });

            benchmark("4. deepMerge + flatten", () -> {
                JsonValue a = Json.obj("x", 1, "nested", Json.obj("p", 1, "q", 2));
                JsonValue b = Json.obj("y", 2, "nested", Json.obj("q", 9, "r", 3));
                JsonValue m = Json.deepMerge(a, b);
                check("x kept", m.get("x").asLong() == 1);
                check("y added", m.get("y").asLong() == 2);
                check("q overwritten", m.at("nested.q").asLong() == 9);
                check("p kept", m.at("nested.p").asLong() == 1);
                Map<String, Object> flat = Json.flatten(m);
                check("flat nested.q", Integer.valueOf(9).equals(flat.get("nested.q"))
                    || Long.valueOf(9L).equals(flat.get("nested.q")));
            });

            // ---- DataFrame JSON ----
            benchmark("5. records orient roundtrip", () -> {
                Path p = tmp.resolve("rec.json");
                String json = "["
                    + "{\"name\":\"alice\",\"age\":30,\"score\":9.5},"
                    + "{\"name\":\"bob\",\"age\":25,\"score\":7.0},"
                    + "{\"name\":\"carol\",\"age\":null,\"score\":8.2}"
                    + "]";
                Files.writeString(p, json, StandardCharsets.UTF_8);
                DataFrame df = DataFrame.readJson(p.toString());
                check("rows", df.rowCount() == 3);
                check("cols", df.columnCount() == 3);
                check("alice", "alice".equals(String.valueOf(df.get(0, "name"))));
                check("age num", df.get(0, "age") instanceof Number);
                check("null age", df.get(2, "age") == null);

                Path out = tmp.resolve("rec-out.json");
                df.toJson(out.toString(), JsonOptions.builder().pretty(true).build());
                DataFrame df2 = DataFrame.readJson(out.toString());
                check("round rows", df2.rowCount() == 3);
                check("round score", ((Number) df2.get(1, "score")).doubleValue() == 7.0);
            });

            benchmark("6. JSONL read/write", () -> {
                Path p = tmp.resolve("rows.jsonl");
                String jsonl = "{\"id\":1,\"tag\":\"a\"}\n"
                    + "{\"id\":2,\"tag\":\"b\"}\n"
                    + "\n"
                    + "{\"id\":3,\"tag\":\"c\",\"extra\":true}\n";
                Files.writeString(p, jsonl, StandardCharsets.UTF_8);
                DataFrame df = DataFrame.readJsonl(p.toString());
                check("jsonl rows", df.rowCount() == 3);
                check("has extra col", df.hasColumn("extra"));
                check("id2", ((Number) df.get(1, "id")).longValue() == 2L);

                Path out = tmp.resolve("rows-out.jsonl");
                df.toJsonl(out.toString());
                DataFrame df2 = DataFrame.readJsonl(out.toString());
                check("jsonl round rows", df2.rowCount() == 3);
            });

            benchmark("7. columns / values / split orients", () -> {
                String columnsJson = "{\"x\":[1,2,3],\"y\":[10,20,30]}";
                DataFrame dfC = DataFrame.readJsonString(columnsJson,
                    JsonOptions.builder().orient(JsonOptions.Orient.COLUMNS).build());
                check("cols rows", dfC.rowCount() == 3);
                check("cols x1", ((Number) dfC.get(1, "x")).intValue() == 2);

                String valuesJson = "[[1,\"a\"],[2,\"b\"]]";
                DataFrame dfV = DataFrame.readJsonString(valuesJson,
                    JsonOptions.builder().orient(JsonOptions.Orient.VALUES)
                        .columnNames("n", "s").build());
                check("values rows", dfV.rowCount() == 2);
                check("values s", "b".equals(String.valueOf(dfV.get(1, "s"))));

                DataFrame base = DataFrame.create();
                base.addColumn("a", Column.DType.INT64);
                base.addColumn("b", Column.DType.STRING);
                base.addRow(1L, "x");
                base.addRow(2L, "y");
                String split = base.toJsonString(JsonOptions.builder().orient(JsonOptions.Orient.SPLIT).build());
                check("split has columns", split.contains("\"columns\""));
                DataFrame dfS = DataFrame.readJsonString(split,
                    JsonOptions.builder().orient(JsonOptions.Orient.SPLIT).build());
                check("split rows", dfS.rowCount() == 2);
            });

            benchmark("8. flatten nested + recordPath", () -> {
                String nested = "["
                    + "{\"user\":{\"id\":1,\"name\":\"a\"},\"score\":1},"
                    + "{\"user\":{\"id\":2,\"name\":\"b\"},\"score\":2}"
                    + "]";
                DataFrame flat = DataFrame.readJsonString(nested,
                    JsonOptions.builder().flatten(true).build());
                check("flat has user.id", flat.hasColumn("user.id") || flat.hasColumn("user" + "." + "id"));
                check("flat rows", flat.rowCount() == 2);

                String wrapped = "{\"meta\":{\"src\":\"t\"},\"items\":["
                    + "{\"v\":1},{\"v\":2},{\"v\":3}]}";
                DataFrame rp = DataFrame.readJsonString(wrapped,
                    JsonOptions.builder()
                        .recordPath("items")
                        .metaPaths(Collections.singletonList("meta.src"))
                        .build());
                check("recordPath rows", rp.rowCount() == 3);
                check("meta src col", rp.hasColumn("src"));
            });

            benchmark("9. auto-detect jsonl by extension", () -> {
                Path p = tmp.resolve("auto.jsonl");
                Files.writeString(p, "{\"a\":1}\n{\"a\":2}\n", StandardCharsets.UTF_8);
                DataFrame df = DataFrame.readJson(p.toString()); // no explicit lines()
                check("auto rows", df.rowCount() == 2);
            });

            benchmark("10. dataframe operators rename/replace/str", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("Name", Column.DType.STRING);
                df.addColumn("Age", Column.DType.FLOAT64);
                df.addRow(" Alice ", 10.0);
                df.addRow("Bob", 20.0);
                df.addRow("NA", null);
                DataFrame d2 = df.rename("Name", "name")
                    .rename("Age", "age")
                    .strStrip("name")
                    .strLower("name")
                    .replace("na", "unknown", "name")
                    .fillna(0.0, "age");
                check("renamed", d2.hasColumn("name") && d2.hasColumn("age"));
                check("strip lower", "alice".equals(String.valueOf(d2.get(0, "name"))));
                check("fillna", ((Number) d2.get(2, "age")).doubleValue() == 0.0);
            });

            benchmark("11. feature pipeline chain", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("x", Column.DType.FLOAT64);
                df.addColumn("y", Column.DType.FLOAT64);
                df.addColumn("cat", Column.DType.STRING);
                df.addColumn("label", Column.DType.FLOAT64);
                Random rng = new Random(0);
                for (int i = 0; i < 40; i++) {
                    double x = rng.nextGaussian();
                    double y = rng.nextGaussian() + (i % 2) * 2;
                    df.addRow(x, y, i % 2 == 0 ? "A" : "B", (double) (i % 2));
                }
                // inject a null
                df.set(0, "x", null);

                DataFrame out = df.feature()
                    .impute("mean", "x", "y")
                    .standardScale("x", "y")
                    .oneHot("cat")
                    .build();
                check("no null x", out.column("x").get(0) != null);
                check("ohe cols", out.hasColumn("cat_A") || out.columnCount() >= 4);

                DataFramePipeline pipe = df.pipeline()
                    .append("impute", new SimpleImputer("mean", "x", "y"))
                    .append("scale", new StandardScaler("x", "y"));
                DataFrame t = pipe.fitTransform();
                check("pipeline rows", t.rowCount() == df.rowCount());
            });

            benchmark("12. sklearn Pipeline with estimator", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("x1", Column.DType.FLOAT64);
                df.addColumn("x2", Column.DType.FLOAT64);
                df.addColumn("y", Column.DType.FLOAT64);
                Random rng = new Random(1);
                for (int i = 0; i < 60; i++) {
                    double x1 = rng.nextGaussian() + (i < 30 ? 0 : 3);
                    double x2 = rng.nextGaussian() + (i < 30 ? 0 : 3);
                    df.addRow(x1, x2, i < 30 ? 0.0 : 1.0);
                }
                Pipeline pipe = new Pipeline()
                    .addStep("scale", new StandardScaler("x1", "x2"))
                    .addStep("clf", new LogisticRegression());
                pipe.fit(df, new String[]{"x1", "x2"}, "y");
                double[] preds = pipe.predict(df, new String[]{"x1", "x2"});
                check("preds len", preds.length == df.rowCount());
                int correct = 0;
                for (int i = 0; i < preds.length; i++) {
                    double label = ((Number) df.get(i, "y")).doubleValue();
                    if ((preds[i] >= 0.5 ? 1.0 : 0.0) == label) correct++;
                }
                check("accuracy decent", correct >= 40);
            });

            benchmark("13. table orient + toRecords", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("id", Column.DType.INT64);
                df.addColumn("name", Column.DType.STRING);
                df.addRow(1L, "x");
                df.addRow(2L, "y");
                String table = df.toJsonString(JsonOptions.builder().orient(JsonOptions.Orient.TABLE).build());
                check("table schema", table.contains("schema") && table.contains("fields"));
                DataFrame back = DataFrame.readJsonString(table,
                    JsonOptions.builder().orient(JsonOptions.Orient.TABLE).build());
                check("table rows", back.rowCount() == 2);
                List<Map<String, Object>> recs = df.toRecords();
                check("toRecords", recs.size() == 2 && recs.get(0).containsKey("name"));
            });

            benchmark("14. arithmetic + filterRows + map", () -> {
                DataFrame df = DataFrame.create();
                df.addColumn("a", Column.DType.FLOAT64);
                df.addColumn("b", Column.DType.FLOAT64);
                df.addRow(2.0, 3.0);
                df.addRow(4.0, 5.0);
                DataFrame d2 = df.withArithmetic("sum", "a", "b", '+')
                    .filterRows(m -> ((Number) m.get("sum")).doubleValue() > 6)
                    .map("a", v -> ((Number) v).doubleValue() * 10);
                check("filtered rows", d2.rowCount() == 1);
                check("mapped a", ((Number) d2.get(0, "a")).doubleValue() == 40.0);
            });

        } finally {
            // best-effort cleanup
            try {
                Files.walk(tmp)
                    .sorted(Comparator.reverseOrder())
                    .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
            } catch (Exception ignored) {}
        }

        System.out.println();
        System.out.println("Passed checks: " + passed + "  Failed benches: " + failed);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        System.out.println("ALL OK");
    }
}
