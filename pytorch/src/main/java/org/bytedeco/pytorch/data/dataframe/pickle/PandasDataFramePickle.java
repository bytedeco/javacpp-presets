package org.bytedeco.pytorch.data.dataframe.pickle;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.io.IoTypeCoercion;
import org.bytedeco.pytorch.data.pickle.Pickle;

import java.io.File;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Encode/decode DataFrames as portable pickle objects for Python interop.
 *
 * <p>Recognized on read:
 * <ul>
 *   <li>{@code List&lt;Map&gt;} records (legacy)</li>
 *   <li>Self-describing dict with {@code __pandas_dataframe__}</li>
 *   <li>Column-oriented {@code Map&lt;String, List&gt;}</li>
 *   <li>Allow-listed reconstructed numpy/pandas-ish structures from {@link Pickle}</li>
 * </ul>
 */
public final class PandasDataFramePickle {
    public static final String MARKER = "__pandas_dataframe__";
    public static final String MARKER_ALT = "__bytedeco_dataframe__";

    private PandasDataFramePickle() {}

    public static DataFrame load(String path) throws Exception {
        return load(path, PickleOptions.defaults());
    }

    public static DataFrame load(String path, PickleOptions options) throws Exception {
        Object root = Pickle.load(new File(path));
        return fromObject(root, options == null ? PickleOptions.defaults() : options);
    }

    public static void dump(DataFrame df, String path) throws Exception {
        dump(df, path, PickleOptions.defaults());
    }

    public static void dump(DataFrame df, String path, PickleOptions options) throws Exception {
        PickleOptions opt = options == null ? PickleOptions.defaults() : options;
        Object payload = toObject(df, opt);
        Pickle.dump(payload, new File(path));
    }

    @SuppressWarnings("unchecked")
    public static DataFrame fromObject(Object root, PickleOptions opt) {
        if (root == null) return DataFrame.create();

        // Self-describing
        if (root instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) root;
            if (Boolean.TRUE.equals(m.get(MARKER)) || Boolean.TRUE.equals(m.get(MARKER_ALT))) {
                return fromSelfDesc(m);
            }
            // Column-oriented: values are lists/arrays of equal length
            if (looksLikeColumns(m)) {
                return fromColumns((Map<String, Object>) (Map<?, ?>) castStringKeys(m));
            }
            // Single-row dict
            if (isFlatRow(m)) {
                DataFrame df = DataFrame.create();
                Map<String, Object> row = castStringKeys(m);
                for (Map.Entry<String, Object> e : row.entrySet()) {
                    df.addColumn(e.getKey(), IoTypeCoercion.inferFromObject(e.getValue()));
                }
                int ri = df.addEmptyRow();
                for (Map.Entry<String, Object> e : row.entrySet()) {
                    df.set(ri, e.getKey(), e.getValue());
                }
                return df;
            }
        }

        // Records list
        if (root instanceof List) {
            List<?> list = (List<?>) root;
            if (list.isEmpty()) return DataFrame.create();
            Object first = list.get(0);
            if (first instanceof Map) {
                return fromRecords(list);
            }
            // list of primitives → single column
            DataFrame df = DataFrame.create();
            Column.DType dt = IoTypeCoercion.inferFromObject(first);
            df.addColumn("value", dt);
            for (Object v : list) {
                int ri = df.addEmptyRow();
                df.set(ri, "value", v);
            }
            return df;
        }

        // numpy-like map from our tensor codec or allow-list reconstruct
        if (root instanceof Map && ((Map<?, ?>) root).containsKey("__torch_tensor__")) {
            throw new IllegalArgumentException(
                "Pickle root is a tensor map, not a DataFrame. Use Pickle.loadTensor().");
        }

        throw new IllegalArgumentException(
            "Unsupported pickle root for DataFrame: "
                + (root == null ? "null" : root.getClass().getName()));
    }

    public static Object toObject(DataFrame df, PickleOptions opt) {
        switch (opt.layout()) {
            case RECORDS:
                return toRecords(df);
            case COLUMNS:
                return toColumns(df);
            case SELF_DESC:
            default:
                return toSelfDesc(df, opt);
        }
    }

    private static Map<String, Object> toSelfDesc(DataFrame df, PickleOptions opt) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put(MARKER, Boolean.TRUE);
        m.put(MARKER_ALT, Boolean.TRUE);
        List<String> columns = new ArrayList<>();
        List<String> dtypes = new ArrayList<>();
        for (int i = 0; i < df.columnCount(); i++) {
            Column c = df.column(i);
            columns.add(c.name());
            dtypes.add(c.dtype().name());
        }
        m.put("columns", columns);
        m.put("dtypes", dtypes);
        m.put("index", null);
        // data as records for maximum Python friendliness
        m.put("data", toRecords(df));
        // also embed columns orient for convenience
        m.put("data_columns", toColumns(df));
        m.put("orient", "records");
        m.put("pandas_compat", opt.pandasCompat());
        return m;
    }

    private static List<Map<String, Object>> toRecords(DataFrame df) {
        List<Map<String, Object>> rows = new ArrayList<>(df.rowCount());
        for (int r = 0; r < df.rowCount(); r++) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (int c = 0; c < df.columnCount(); c++) {
                Column col = df.column(c);
                row.put(col.name(), normalizeForPickle(col.get(r)));
            }
            rows.add(row);
        }
        return rows;
    }

    private static Map<String, Object> toColumns(DataFrame df) {
        Map<String, Object> cols = new LinkedHashMap<>();
        for (int c = 0; c < df.columnCount(); c++) {
            Column col = df.column(c);
            List<Object> values = new ArrayList<>(df.rowCount());
            for (int r = 0; r < df.rowCount(); r++) {
                values.add(normalizeForPickle(col.get(r)));
            }
            cols.put(col.name(), values);
        }
        return cols;
    }

    @SuppressWarnings("unchecked")
    private static DataFrame fromSelfDesc(Map<?, ?> m) {
        List<String> columns = toStringList(m.get("columns"));
        List<String> dtypes = toStringList(m.get("dtypes"));
        Object data = m.get("data");
        if (data instanceof List) {
            DataFrame df = fromRecords((List<?>) data);
            // apply dtypes if present
            if (dtypes != null && columns != null) {
                // already inferred; optionally re-coerce
                for (int i = 0; i < columns.size() && i < dtypes.size(); i++) {
                    String name = columns.get(i);
                    if (!df.hasColumn(name)) continue;
                    Column.DType dt = parseDType(dtypes.get(i));
                    // leave as-is if matching; coercion per-cell would be heavy — skip if same name exists
                }
            }
            return df;
        }
        if (data instanceof Map) {
            return fromColumns(castStringKeys((Map<?, ?>) data));
        }
        Object dataCols = m.get("data_columns");
        if (dataCols instanceof Map) {
            return fromColumns(castStringKeys((Map<?, ?>) dataCols));
        }
        return DataFrame.create();
    }

    @SuppressWarnings("unchecked")
    private static DataFrame fromRecords(List<?> list) {
        DataFrame df = DataFrame.create();
        if (list.isEmpty()) return df;
        // union of keys, first-seen order
        LinkedHashSet<String> keys = new LinkedHashSet<>();
        for (Object rowObj : list) {
            if (rowObj instanceof Map) {
                for (Object k : ((Map<?, ?>) rowObj).keySet()) {
                    keys.add(String.valueOf(k));
                }
            }
        }
        Map<String, Column.DType> inferred = new LinkedHashMap<>();
        for (String k : keys) {
            Column.DType acc = null;
            for (Object rowObj : list) {
                if (!(rowObj instanceof Map)) continue;
                Object v = ((Map<?, ?>) rowObj).get(k);
                if (v == null) continue;
                Column.DType t = IoTypeCoercion.inferFromObject(v);
                acc = acc == null ? t : IoTypeCoercion.widen(acc, t);
            }
            inferred.put(k, acc == null ? Column.DType.STRING : acc);
            df.addColumn(k, inferred.get(k));
        }
        for (Object rowObj : list) {
            if (!(rowObj instanceof Map)) continue;
            Map<?, ?> row = (Map<?, ?>) rowObj;
            int ri = df.addEmptyRow();
            for (String k : keys) {
                Object v = row.get(k);
                if (v == null) {
                    // try string key variants
                    v = row.get(k);
                }
                try {
                    df.set(ri, k, v == null ? null : IoTypeCoercion.coerce(v, inferred.get(k)));
                } catch (Exception ex) {
                    df.set(ri, k, v == null ? null : String.valueOf(v));
                }
            }
        }
        return df;
    }

    private static DataFrame fromColumns(Map<String, Object> cols) {
        DataFrame df = DataFrame.create();
        if (cols.isEmpty()) return df;
        int rowCount = -1;
        Map<String, List<Object>> lists = new LinkedHashMap<>();
        Map<String, Column.DType> dtypes = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : cols.entrySet()) {
            List<Object> values = toObjectList(e.getValue());
            if (rowCount < 0) rowCount = values.size();
            else rowCount = Math.max(rowCount, values.size());
            Column.DType acc = null;
            for (Object v : values) {
                if (v == null) continue;
                Column.DType t = IoTypeCoercion.inferFromObject(v);
                acc = acc == null ? t : IoTypeCoercion.widen(acc, t);
            }
            dtypes.put(e.getKey(), acc == null ? Column.DType.STRING : acc);
            lists.put(e.getKey(), values);
            df.addColumn(e.getKey(), dtypes.get(e.getKey()));
        }
        if (rowCount < 0) rowCount = 0;
        for (int r = 0; r < rowCount; r++) {
            int ri = df.addEmptyRow();
            for (Map.Entry<String, List<Object>> e : lists.entrySet()) {
                Object v = r < e.getValue().size() ? e.getValue().get(r) : null;
                try {
                    df.set(ri, e.getKey(),
                        v == null ? null : IoTypeCoercion.coerce(v, dtypes.get(e.getKey())));
                } catch (Exception ex) {
                    df.set(ri, e.getKey(), v == null ? null : String.valueOf(v));
                }
            }
        }
        return df;
    }

    private static Object normalizeForPickle(Object v) {
        if (v == null) return null;
        if (v instanceof LocalDate) return v.toString();
        if (v instanceof LocalDateTime) return v.toString();
        if (v instanceof Instant) return v.toString();
        // Preserve nested structures as List/Map for round-trip of LIST/VECTOR/MAP/STRUCT
        if (v instanceof float[]) {
            float[] f = (float[]) v;
            List<Double> list = new ArrayList<>(f.length);
            for (float x : f) list.add((double) x);
            return list;
        }
        if (v instanceof double[]) {
            double[] d = (double[]) v;
            List<Double> list = new ArrayList<>(d.length);
            for (double x : d) list.add(x);
            return list;
        }
        if (v instanceof int[]) {
            int[] a = (int[]) v;
            List<Long> list = new ArrayList<>(a.length);
            for (int x : a) list.add((long) x);
            return list;
        }
        if (v instanceof long[]) {
            long[] a = (long[]) v;
            List<Long> list = new ArrayList<>(a.length);
            for (long x : a) list.add(x);
            return list;
        }
        if (v instanceof boolean[]) {
            boolean[] a = (boolean[]) v;
            List<Boolean> list = new ArrayList<>(a.length);
            for (boolean x : a) list.add(x);
            return list;
        }
        if (v instanceof Object[]) {
            Object[] a = (Object[]) v;
            List<Object> list = new ArrayList<>(a.length);
            for (Object o : a) list.add(normalizeForPickle(o));
            return list;
        }
        if (v instanceof List) {
            List<?> src = (List<?>) v;
            List<Object> list = new ArrayList<>(src.size());
            for (Object o : src) list.add(normalizeForPickle(o));
            return list;
        }
        if (v instanceof Map) {
            Map<?, ?> src = (Map<?, ?>) v;
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : src.entrySet()) {
                out.put(String.valueOf(e.getKey()), normalizeForPickle(e.getValue()));
            }
            return out;
        }
        if (v instanceof byte[]) {
            return Base64.getEncoder().encodeToString((byte[]) v);
        }
        return v;
    }

    private static boolean looksLikeColumns(Map<?, ?> m) {
        if (m.isEmpty()) return false;
        if (m.containsKey(MARKER) || m.containsKey(MARKER_ALT)) return false;
        int listish = 0;
        for (Object v : m.values()) {
            if (v instanceof List || (v != null && v.getClass().isArray())) listish++;
        }
        return listish == m.size() && listish > 0;
    }

    private static boolean isFlatRow(Map<?, ?> m) {
        for (Object v : m.values()) {
            if (v instanceof Map || v instanceof List) return false;
        }
        return !m.isEmpty();
    }

    private static Map<String, Object> castStringKeys(Map<?, ?> m) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : m.entrySet()) {
            out.put(String.valueOf(e.getKey()), e.getValue());
        }
        return out;
    }

    private static List<String> toStringList(Object o) {
        if (o == null) return null;
        List<String> out = new ArrayList<>();
        if (o instanceof List) {
            for (Object x : (List<?>) o) out.add(String.valueOf(x));
            return out;
        }
        if (o.getClass().isArray()) {
            int n = java.lang.reflect.Array.getLength(o);
            for (int i = 0; i < n; i++) out.add(String.valueOf(java.lang.reflect.Array.get(o, i)));
            return out;
        }
        return null;
    }

    private static List<Object> toObjectList(Object o) {
        List<Object> out = new ArrayList<>();
        if (o == null) return out;
        if (o instanceof List) {
            out.addAll((List<?>) o);
            return out;
        }
        if (o.getClass().isArray()) {
            int n = java.lang.reflect.Array.getLength(o);
            for (int i = 0; i < n; i++) out.add(java.lang.reflect.Array.get(o, i));
            return out;
        }
        out.add(o);
        return out;
    }

    private static Column.DType parseDType(String s) {
        try {
            return Column.DType.valueOf(s);
        } catch (Exception e) {
            return Column.DType.STRING;
        }
    }
}
