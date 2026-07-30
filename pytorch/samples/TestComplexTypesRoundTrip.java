package samples;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.io.ComplexCellCodec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Smoke test: LIST / VECTOR / EMBEDDING / MAP / STRUCT / JSON complex cells
 * round-trip across formats that should preserve them (or lossless JSON text).
 */
public final class TestComplexTypesRoundTrip {
    private TestComplexTypesRoundTrip() {}

    public static void main(String[] args) throws Exception {
        Path dir = Files.createTempDirectory("complex-rt-");
        int failures = 0;

        DataFrame src = buildSample();
        System.out.println("source rows=" + src.rowCount() + " cols=" + src.columnCount());
        dump(src, "SRC");

        // ---- JSON ----
        failures += check("json", () -> {
            Path p = dir.resolve("t.json");
            src.toJson(p.toString());
            DataFrame df = DataFrame.readJson(p.toString());
            assertComplex(df);
        });

        // ---- CSV ----
        failures += check("csv", () -> {
            Path p = dir.resolve("t.csv");
            src.toCsv(p.toString());
            DataFrame df = DataFrame.readCsv(p.toString());
            assertComplex(df);
        });

        // ---- Parquet ----
        failures += check("parquet", () -> {
            Path p = dir.resolve("t.parquet");
            src.writeParquet(p.toString());
            DataFrame df = DataFrame.readParquet(p.toString());
            assertComplex(df);
        });

        // ---- Arrow IPC ----
        failures += check("arrow", () -> {
            Path p = dir.resolve("t.arrow");
            src.writeArrow(p.toString());
            DataFrame df = DataFrame.readArrow(p.toString());
            assertComplex(df);
        });

        // ---- Avro ----
        failures += check("avro", () -> {
            Path p = dir.resolve("t.avro");
            src.toAvro(p.toString());
            DataFrame df = DataFrame.readAvro(p.toString());
            assertComplex(df);
        });

        // ---- ORC format (pure Java) ----
        failures += check("orc-format", () -> {
            Path p = dir.resolve("t.orc");
            src.toOrcFormat(p.toString());
            DataFrame df = DataFrame.readOrcFormat(p.toString());
            // LIST/VECTOR native; MAP/STRUCT as JSON string
            assertListVector(df);
        });

        // ---- Pickle ----
        failures += check("pickle", () -> {
            Path p = dir.resolve("t.pkl");
            src.toPickle(p.toString());
            DataFrame df = DataFrame.readPickle(p.toString());
            assertComplex(df);
        });

        // ---- HDF5 ----
        failures += check("hdf5", () -> {
            Path p = dir.resolve("t.h5");
            src.toHdf(p.toString(), "/df");
            DataFrame df = DataFrame.readHdf(p.toString(), "/df");
            assertComplex(df);
        });

        // ---- Excel ----
        failures += check("excel", () -> {
            Path p = dir.resolve("t.xlsx");
            src.toExcel(p.toString());
            DataFrame df = DataFrame.readExcel(p.toString());
            assertComplex(df);
        });

        // codec unit checks
        failures += check("codec-vector", () -> {
            float[] v = (float[]) ComplexCellCodec.decodeText("[1.0, 2.5, 3]", Column.DType.VECTOR);
            if (v == null || v.length != 3 || Math.abs(v[1] - 2.5f) > 1e-5)
                throw new AssertionError("vector decode failed: " + Arrays.toString(v));
            Object list = ComplexCellCodec.decodeText("[1,2,3]", Column.DType.LIST);
            Object map = ComplexCellCodec.decodeText("{\"a\":1,\"b\":[2,3]}", Column.DType.MAP);
            if (!(map instanceof Map)) throw new AssertionError("map decode failed");
            if (list == null) throw new AssertionError("list decode failed");
        });

        System.out.println("==== done failures=" + failures + " dir=" + dir);
        if (failures > 0) System.exit(1);
    }

    private static DataFrame buildSample() {
        DataFrame df = DataFrame.create();
        df.addColumn("id", Column.DType.INT64);
        df.addColumn("tags", Column.DType.LIST);
        df.addColumn("emb", Column.DType.VECTOR);
        df.addColumn("meta", Column.DType.MAP);
        df.addColumn("point", Column.DType.STRUCT);
        df.addColumn("payload", Column.DType.JSON);

        Map<String, Object> meta0 = new LinkedHashMap<>();
        meta0.put("k", "v");
        meta0.put("n", 42);
        Map<String, Object> point0 = new LinkedHashMap<>();
        point0.put("x", 1.5);
        point0.put("y", 2.5);
        Map<String, Object> json0 = new LinkedHashMap<>();
        json0.put("ok", true);
        json0.put("items", Arrays.asList(1, 2));

        int r0 = df.addEmptyRow();
        df.set(r0, "id", 1L);
        df.set(r0, "tags", Arrays.asList(10L, 20L, 30L));
        df.set(r0, "emb", new float[]{0.1f, 0.2f, 0.3f});
        df.set(r0, "meta", meta0);
        df.set(r0, "point", point0);
        df.set(r0, "payload", json0);

        int r1 = df.addEmptyRow();
        df.set(r1, "id", 2L);
        df.set(r1, "tags", new long[]{7, 8});
        df.set(r1, "emb", new float[]{1f, 0f, 0f});
        Map<String, Object> meta1 = new LinkedHashMap<>();
        meta1.put("nested", Arrays.asList("a", "b"));
        df.set(r1, "meta", meta1);
        Map<String, Object> point1 = new LinkedHashMap<>();
        point1.put("x", -1);
        point1.put("y", 0);
        df.set(r1, "point", point1);
        df.set(r1, "payload", "{\"z\":3}");

        return df;
    }

    private static void assertComplex(DataFrame df) {
        assertListVector(df);
        if (!df.hasColumn("meta") && !df.hasColumn("point") && !df.hasColumn("payload")) {
            throw new AssertionError("missing map/struct/json columns: " + names(df));
        }
        // at least one nested cell materializes as Map or JSON-decodable string
        Object meta = firstNonNull(df, "meta");
        if (meta != null) {
            Object coerced = ComplexCellCodec.coerceComplex(meta, Column.DType.MAP);
            if (!(coerced instanceof Map)) {
                throw new AssertionError("meta not map-like: " + meta.getClass() + " = " + meta);
            }
        }
        Object emb = firstNonNull(df, "emb");
        if (emb != null) {
            Object v = ComplexCellCodec.coerceComplex(emb, Column.DType.VECTOR);
            if (!(v instanceof float[])) {
                throw new AssertionError("emb not float[]: " + (v == null ? null : v.getClass()) + " = " + v);
            }
        }
    }

    private static void assertListVector(DataFrame df) {
        if (df.rowCount() < 1) throw new AssertionError("empty frame");
        Object tags = firstNonNull(df, "tags");
        if (tags == null && df.hasColumn("tags")) {
            // allow empty but column must exist
        } else if (tags != null) {
            Object list = ComplexCellCodec.coerceComplex(tags, Column.DType.LIST);
            if (list == null) throw new AssertionError("tags null after coerce");
            List<Object> elems = ComplexCellCodec.asObjectList(list);
            if (elems == null || elems.isEmpty()) {
                throw new AssertionError("tags empty: " + tags);
            }
        }
    }

    private static Object firstNonNull(DataFrame df, String col) {
        if (!df.hasColumn(col)) return null;
        for (int i = 0; i < df.rowCount(); i++) {
            Object v = df.get(i, col);
            if (v != null) return v;
        }
        return null;
    }

    private static List<String> names(DataFrame df) {
        return df.columns().stream().map(Column::name).toList();
    }

    private static void dump(DataFrame df, String label) {
        System.out.println("-- " + label + " --");
        for (Column c : df.columns()) {
            System.out.println("  " + c.name() + ":" + c.dtype() + " sample=" + c.get(0));
        }
    }

    private interface Throwing {
        void run() throws Exception;
    }

    private static int check(String name, Throwing t) {
        try {
            t.run();
            System.out.println("[OK] " + name);
            return 0;
        } catch (Throwable e) {
            System.out.println("[FAIL] " + name + ": " + e);
            e.printStackTrace(System.out);
            return 1;
        }
    }
}
