package org.bytedeco.pytorch.utils.kafka;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.recommend.basic.features.DenseFeature;
import org.bytedeco.pytorch.recommend.basic.features.Feature;
import org.bytedeco.pytorch.recommend.basic.features.SequenceFeature;
import org.bytedeco.pytorch.recommend.basic.features.SparseFeature;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Bridge Kafka / DataFrame batches into tensors and recommend feature maps.
 *
 * <p>Typical online path:
 * <pre>{@code
 * k.streamDataFrame(opts, 2048, batch -> {
 *     DataFrame clicks = KafkaFeatureBridge.selectEvents(batch, "click", "expose");
 *     Map&lt;String, Tensor&gt; feats = KafkaFeatureBridge.toFeatureTensors(clicks, featureList);
 *     Tensor dense = KafkaFeatureBridge.toDenseTensor(clicks, "age", "score");
 *     // model forward / online train
 * });
 * }</pre>
 *
 * <p>Also supports offline dump → FE:
 * <pre>{@code
 * DataFrame dump = KafkaFile.readJsonl(Path.of("expose.jsonl"));
 * DataFrame x = dump.feature().impute("mean", "age").standardScale("age").build();
 * Tensor t = KafkaFeatureBridge.toDenseTensor(x, "age", "price");
 * }</pre>
 */
public final class KafkaFeatureBridge {

    public static final String EVENT_TYPE_COL = "event_type";
    public static final String DEFAULT_EVENT_COL_CANDIDATES =
            "event_type,event,action,evt,type";

    private KafkaFeatureBridge() {}

    // ── dense tensor ─────────────────────────────────────────────────────────

    /**
     * Stack numeric columns into {@code [n_rows, n_cols]} float tensor.
     * Delegates to {@link DataFrame#toTensor(String...)}.
     */
    public static Tensor toDenseTensor(DataFrame df, String... columns) {
        Objects.requireNonNull(df, "df");
        if (columns == null || columns.length == 0) {
            return df.toTensor();
        }
        return df.toTensor(columns);
    }

    public static Tensor toDenseTensor(DataFrame df, List<String> columns) {
        if (columns == null || columns.isEmpty()) return toDenseTensor(df);
        return toDenseTensor(df, columns.toArray(new String[0]));
    }

    /**
     * All numeric columns (INT/LONG/FLOAT/DOUBLE/BOOLEAN) stacked as dense matrix.
     */
    public static Tensor toDenseTensor(DataFrame df) {
        Objects.requireNonNull(df, "df");
        List<String> cols = numericColumns(df);
        if (cols.isEmpty()) {
            return floatTensor(new float[0]);
        }
        return df.toTensor(cols.toArray(new String[0]));
    }

    // ── feature tensors (recommend) ──────────────────────────────────────────

    /**
     * Build a name→Tensor map for a list of recommend {@link Feature}s.
     * <ul>
     *   <li>{@link SparseFeature} → Long tensor {@code [batch]}</li>
     *   <li>{@link DenseFeature} → Float tensor {@code [batch]} or {@code [batch, dim]}</li>
     *   <li>{@link SequenceFeature} → Long tensor {@code [batch, maxLen]} (right-padded)</li>
     * </ul>
     */
    public static Map<String, Tensor> toFeatureTensors(DataFrame df, List<? extends Feature> features) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(features, "features");
        Map<String, Tensor> out = new LinkedHashMap<>();
        int n = df.rowCount();
        for (Feature f : features) {
            if (f == null || f.name() == null) continue;
            String name = f.name();
            if (f instanceof SequenceFeature seq) {
                out.put(name, toSequenceTensor(df, seq));
            } else if (f instanceof SparseFeature sparse) {
                out.put(name, toSparseTensor(df, sparse.name(), n));
            } else if (f instanceof DenseFeature dense) {
                out.put(name, toDenseFeatureTensor(df, dense));
            } else {
                // generic: try sparse-like long ids, else dense float
                if (df.hasColumn(name)) {
                    Column col = df.column(name);
                    if (isIntegral(col)) {
                        out.put(name, toSparseTensor(df, name, n));
                    } else {
                        out.put(name, toDenseTensor(df, name));
                    }
                }
            }
        }
        return out;
    }

    public static Map<String, Tensor> toFeatureTensors(DataFrame df, Feature... features) {
        return toFeatureTensors(df, features == null ? List.of() : List.of(features));
    }

    /** Sparse / id column → {@code Long[batch]}. Missing → 0. */
    public static Tensor toSparseTensor(DataFrame df, String column) {
        Objects.requireNonNull(df, "df");
        return toSparseTensor(df, column, df.rowCount());
    }

    private static Tensor toSparseTensor(DataFrame df, String column, int n) {
        long[] data = new long[n];
        if (!df.hasColumn(column)) {
            return longTensor(data);
        }
        Column col = df.column(column);
        for (int i = 0; i < n; i++) {
            data[i] = toLongId(col.get(i));
        }
        return longTensor(data);
    }

    private static Tensor toDenseFeatureTensor(DataFrame df, DenseFeature dense) {
        String name = dense.name();
        int dim = Math.max(1, dense.embedDim());
        int n = df.rowCount();
        if (dim == 1) {
            if (!df.hasColumn(name)) {
                return floatTensor(new float[n]);
            }
            return df.toTensor(name);
        }
        // multi-dim dense: column may hold float[] / List / VECTOR
        if (!df.hasColumn(name)) {
            return tensor(new float[n * dim], n, dim);
        }
        try {
            return df.toTensorColumn(name);
        } catch (Exception e) {
            // fall back: replicate scalar across dim
            float[] data = new float[n * dim];
            Column col = df.column(name);
            for (int i = 0; i < n; i++) {
                float v = toFloat(col.get(i));
                for (int d = 0; d < dim; d++) data[i * dim + d] = v;
            }
            return tensor(data, n, dim);
        }
    }

    /**
     * Sequence column → {@code Long[batch, maxLen]}.
     * Cell may be {@code List}, {@code long[]}, {@code int[]}, CSV string, or JSON array text.
     */
    public static Tensor toSequenceTensor(DataFrame df, SequenceFeature seq) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(seq, "seq");
        int n = df.rowCount();
        int maxLen = Math.max(1, seq.maxLen());
        long pad = seq.paddingIdx();
        long[] data = new long[n * maxLen];
        // prefill pad
        if (pad != 0L) {
            for (int i = 0; i < data.length; i++) data[i] = pad;
        }
        if (!df.hasColumn(seq.name())) {
            return tensor(data, n, maxLen);
        }
        Column col = df.column(seq.name());
        for (int i = 0; i < n; i++) {
            long[] ids = parseSequence(col.get(i), maxLen, pad);
            System.arraycopy(ids, 0, data, i * maxLen, maxLen);
        }
        return tensor(data, n, maxLen);
    }

    public static Tensor toSequenceTensor(DataFrame df, String column, int maxLen, long paddingIdx) {
        SequenceFeature seq = new SequenceFeature(column, 1L, 8, "mean", null, maxLen, paddingIdx);
        return toSequenceTensor(df, seq);
    }

    // ── event filter / join helpers ──────────────────────────────────────────

    /**
     * Filter rows whose event-type column is in {@code eventTypes} (case-insensitive).
     * Auto-detects column among {@code event_type, event, action, evt, type}.
     */
    public static DataFrame selectEvents(DataFrame df, String... eventTypes) {
        Objects.requireNonNull(df, "df");
        if (eventTypes == null || eventTypes.length == 0) return df;
        String col = findEventColumn(df);
        if (col == null) return df;
        java.util.Set<String> want = new java.util.HashSet<>();
        for (String e : eventTypes) {
            if (e != null) want.add(e.toLowerCase(Locale.ROOT));
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Object v = df.get(i, col);
            if (v == null) continue;
            if (want.contains(String.valueOf(v).toLowerCase(Locale.ROOT))) {
                rows.add(df.toDict(i));
            }
        }
        return DataFrame.fromRecords(rows);
    }

    /**
     * Prefix all non-metadata columns with {@code prefix} (e.g. {@code "u_"} / {@code "i_"}
     * for multi-topic user/item feature join).
     */
    public static DataFrame withFeaturePrefix(DataFrame df, String prefix) {
        Objects.requireNonNull(df, "df");
        if (prefix == null || prefix.isEmpty()) return df;
        List<Map<String, Object>> rows = new ArrayList<>(df.rowCount());
        for (Map<String, Object> row : df.toRecords()) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : row.entrySet()) {
                String k = e.getKey();
                if (k != null && k.startsWith("__")) {
                    out.put(k, e.getValue());
                } else {
                    out.put(prefix + k, e.getValue());
                }
            }
            rows.add(out);
        }
        return DataFrame.fromRecords(rows);
    }

    /**
     * Drop Kafka metadata columns ({@code __topic}, {@code __partition}, …).
     */
    public static DataFrame dropMetadata(DataFrame df) {
        Objects.requireNonNull(df, "df");
        List<Map<String, Object>> rows = new ArrayList<>(df.rowCount());
        for (Map<String, Object> row : df.toRecords()) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : row.entrySet()) {
                String k = e.getKey();
                if (k != null && k.startsWith("__")) continue;
                out.put(k, e.getValue());
            }
            rows.add(out);
        }
        return DataFrame.fromRecords(rows);
    }

    /**
     * Keep only listed columns (plus optional metadata). Missing columns are skipped.
     */
    public static DataFrame selectColumns(DataFrame df, Collection<String> columns, boolean keepMetadata) {
        Objects.requireNonNull(df, "df");
        if (columns == null || columns.isEmpty()) {
            return keepMetadata ? df : dropMetadata(df);
        }
        java.util.Set<String> want = new java.util.LinkedHashSet<>(columns);
        List<Map<String, Object>> rows = new ArrayList<>(df.rowCount());
        for (Map<String, Object> row : df.toRecords()) {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : row.entrySet()) {
                String k = e.getKey();
                if (k == null) continue;
                if (want.contains(k) || (keepMetadata && k.startsWith("__"))) {
                    out.put(k, e.getValue());
                }
            }
            rows.add(out);
        }
        return DataFrame.fromRecords(rows);
    }

    /**
     * Convert a batch of {@link KafkaRecord}s directly to feature tensors.
     */
    public static Map<String, Tensor> recordsToFeatureTensors(
            List<KafkaRecord> records,
            List<? extends Feature> features,
            boolean includeMetadata) {
        DataFrame df = KafkaConsumer.recordsToDataFrame(
                records, includeMetadata, KafkaOptions.ValueFormat.JSON);
        return toFeatureTensors(df, features);
    }

    // ── column discovery ─────────────────────────────────────────────────────

    public static List<String> numericColumns(DataFrame df) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < df.columnCount(); i++) {
            Column c = df.column(i);
            String name = c.name();
            if (name != null && name.startsWith("__")) continue;
            if (isNumeric(c)) out.add(name);
        }
        return out;
    }

    public static String findEventColumn(DataFrame df) {
        String[] candidates = DEFAULT_EVENT_COL_CANDIDATES.split(",");
        for (String c : candidates) {
            if (df.hasColumn(c)) return c;
        }
        // case-insensitive scan
        for (int i = 0; i < df.columnCount(); i++) {
            String n = df.column(i).name();
            if (n == null) continue;
            String lower = n.toLowerCase(Locale.ROOT);
            for (String c : candidates) {
                if (lower.equals(c)) return n;
            }
        }
        return null;
    }

    // ── parsers ──────────────────────────────────────────────────────────────

    static long[] parseSequence(Object cell, int maxLen, long pad) {
        long[] out = new long[maxLen];
        if (pad != 0L) {
            for (int i = 0; i < maxLen; i++) out[i] = pad;
        }
        if (cell == null) return out;
        if (cell instanceof long[] arr) {
            int n = Math.min(maxLen, arr.length);
            System.arraycopy(arr, 0, out, 0, n);
            return out;
        }
        if (cell instanceof int[] arr) {
            int n = Math.min(maxLen, arr.length);
            for (int i = 0; i < n; i++) out[i] = arr[i];
            return out;
        }
        if (cell instanceof float[] arr) {
            int n = Math.min(maxLen, arr.length);
            for (int i = 0; i < n; i++) out[i] = (long) arr[i];
            return out;
        }
        if (cell instanceof double[] arr) {
            int n = Math.min(maxLen, arr.length);
            for (int i = 0; i < n; i++) out[i] = (long) arr[i];
            return out;
        }
        if (cell instanceof List<?> list) {
            int n = Math.min(maxLen, list.size());
            for (int i = 0; i < n; i++) out[i] = toLongId(list.get(i));
            return out;
        }
        if (cell instanceof String s) {
            String t = s.trim();
            if (t.startsWith("[") && t.endsWith("]")) {
                try {
                    Object decoded = org.bytedeco.pytorch.utils.json.Json.decode(t);
                    return parseSequence(decoded, maxLen, pad);
                } catch (Exception ignored) {
                }
            }
            String[] parts = t.split("[,;\\s]+");
            int n = Math.min(maxLen, parts.length);
            int j = 0;
            for (int i = 0; i < parts.length && j < n; i++) {
                if (parts[i].isEmpty()) continue;
                out[j++] = toLongId(parts[i]);
            }
            return out;
        }
        // scalar → length-1 sequence
        out[0] = toLongId(cell);
        return out;
    }

    static long toLongId(Object v) {
        if (v == null) return 0L;
        if (v instanceof Number n) return n.longValue();
        if (v instanceof Boolean b) return b ? 1L : 0L;
        String s = String.valueOf(v).trim();
        if (s.isEmpty()) return 0L;
        try {
            if (s.indexOf('.') >= 0) return (long) Double.parseDouble(s);
            return Long.parseLong(s);
        } catch (NumberFormatException e) {
            // stable string hash bucket (non-cryptographic) for raw categorical tokens
            return Math.floorMod(s.hashCode(), 1_000_000_007);
        }
    }

    static float toFloat(Object v) {
        if (v == null) return 0f;
        if (v instanceof Number n) return n.floatValue();
        if (v instanceof Boolean b) return b ? 1f : 0f;
        try {
            return Float.parseFloat(String.valueOf(v));
        } catch (NumberFormatException e) {
            return 0f;
        }
    }

    private static boolean isNumeric(Column c) {
        if (c == null) return false;
        Column.DType dt = c.dtype();
        if (dt == null) return false;
        return switch (dt) {
            case INT32, INT64, FLOAT32, FLOAT64, BOOLEAN -> true;
            default -> false;
        };
    }

    private static boolean isIntegral(Column c) {
        if (c == null) return false;
        Column.DType dt = c.dtype();
        return dt == Column.DType.INT32 || dt == Column.DType.INT64 || dt == Column.DType.BOOLEAN;
    }

    // ── local tensor constructors (avoid TensorHelpers compile coupling) ─────

    private static Tensor floatTensor(float[] data) {
        if (data == null) data = new float[0];
        return Tensor.create(data, data.length);
    }

    private static Tensor longTensor(long[] data) {
        if (data == null) data = new long[0];
        return Tensor.create(data, data.length);
    }

    private static Tensor tensor(float[] data, long rows, long cols) {
        if (data == null) data = new float[0];
        return Tensor.create(data, rows, cols);
    }

    private static Tensor tensor(long[] data, long rows, long cols) {
        if (data == null) data = new long[0];
        return Tensor.create(data, rows, cols);
    }
}

