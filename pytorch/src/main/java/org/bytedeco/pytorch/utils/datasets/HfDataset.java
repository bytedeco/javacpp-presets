/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.datasets;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * HuggingFace {@code datasets}-style in-memory dataset.
 *
 * <p>Rows are {@code Map&lt;String, Object&gt;} feature dicts. Supports map/filter,
 * train/test split, sharding, multi-format load (CSV/TSV/JSON/JSONL/text/Parquet via
 * pure-Java {@code LocalParquetReader}), DataFrame bridge, and simple disk cache.
 *
 * <p>For Hub download + config/split selection use {@link HfDatasets#loadDataset(String)}
 * (token via {@code HF_TOKEN} / {@link org.bytedeco.pytorch.utils.hub.HfToken},
 * endpoint via {@code HF_ENDPOINT} / {@code HF_MIRROR}).
 *
 * <pre>{@code
 * // local
 * HfDataset ds = HfDataset.fromParquet(Path.of("train.parquet"));
 * ds = ds.map(r -> { r.put("len", r.get("text").toString().length()); return r; });
 * DatasetDict split = ds.trainTestSplit(0.2, 42L);
 *
 * // hub
 * DatasetDict glue = HfDatasets.loadDataset("glue", "cola");
 * }</pre>
 */
public final class HfDataset implements Iterable<Map<String, Object>> {

    private final List<Map<String, Object>> rows;
    private final List<String> columnNames;
    private final boolean streaming;
    private final String info;

    private HfDataset(List<Map<String, Object>> rows, boolean streaming, String info) {
        this.rows = rows == null ? new ArrayList<>() : rows;
        this.streaming = streaming;
        this.info = info == null ? "" : info;
        this.columnNames = rows == null || rows.isEmpty()
                ? new ArrayList<>()
                : new ArrayList<>(rows.get(0).keySet());
    }

    public static HfDataset empty() {
        return new HfDataset(new ArrayList<>(), false, "empty");
    }

    public static HfDataset fromList(List<Map<String, Object>> rows) {
        List<Map<String, Object>> copy = new ArrayList<>(rows == null ? 0 : rows.size());
        if (rows != null) {
            for (Map<String, Object> r : rows) {
                copy.add(new LinkedHashMap<>(r));
            }
        }
        return new HfDataset(copy, false, "fromList");
    }

    public static HfDataset fromDict(Map<String, List<?>> columns) {
        Objects.requireNonNull(columns, "columns");
        int n = 0;
        for (List<?> col : columns.values()) {
            n = Math.max(n, col == null ? 0 : col.size());
        }
        List<Map<String, Object>> rows = new ArrayList<>(n);
        List<String> keys = new ArrayList<>(columns.keySet());
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (String k : keys) {
                List<?> col = columns.get(k);
                row.put(k, col != null && i < col.size() ? col.get(i) : null);
            }
            rows.add(row);
        }
        return new HfDataset(rows, false, "fromDict");
    }

    /** Synthetic text classification rows for offline benchmarks. */
    public static HfDataset fakeText(int n, long seed) {
        Random rng = new Random(seed);
        String[] labels = {"neg", "neu", "pos"};
        String[] words = {"good", "bad", "great", "terrible", "ok", "amazing", "poor", "fine"};
        List<Map<String, Object>> rows = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Map<String, Object> r = new LinkedHashMap<>();
            int len = 3 + rng.nextInt(8);
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < len; j++) {
                if (j > 0) sb.append(' ');
                sb.append(words[rng.nextInt(words.length)]);
            }
            r.put("text", sb.toString());
            r.put("label", rng.nextInt(labels.length));
            r.put("label_text", labels[(Integer) r.get("label")]);
            r.put("id", i);
            rows.add(r);
        }
        return new HfDataset(rows, false, "fakeText(n=" + n + ")");
    }

    public static HfDataset fromJsonl(Path path) throws IOException {
        List<Map<String, Object>> rows = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                rows.add(parseJsonObject(line));
            }
        }
        return new HfDataset(rows, false, "jsonl:" + path.getFileName());
    }

    public static HfDataset fromCsv(Path path, boolean hasHeader) throws IOException {
        return fromCsv(path, hasHeader, ',');
    }

    public static HfDataset fromCsv(Path path, boolean hasHeader, char delimiter) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        if (lines.isEmpty()) return empty();
        List<String> headers;
        int start;
        if (hasHeader) {
            headers = splitDelimited(lines.get(0), delimiter);
            start = 1;
        } else {
            int cols = splitDelimited(lines.get(0), delimiter).size();
            headers = new ArrayList<>(cols);
            for (int i = 0; i < cols; i++) headers.add("col" + i);
            start = 0;
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = start; i < lines.size(); i++) {
            String line = lines.get(i);
            if (line == null) continue;
            // keep trailing spaces inside quoted fields; only skip fully empty lines
            if (line.isEmpty() || line.trim().isEmpty()) continue;
            List<String> cells = splitDelimited(line, delimiter);
            Map<String, Object> row = new LinkedHashMap<>();
            for (int c = 0; c < headers.size(); c++) {
                row.put(headers.get(c), c < cells.size() ? autoType(cells.get(c)) : null);
            }
            rows.add(row);
        }
        return new HfDataset(rows, false, "csv:" + path.getFileName());
    }

    public static HfDataset fromTsv(Path path, boolean hasHeader) throws IOException {
        return fromCsv(path, hasHeader, '\t');
    }

    /**
     * Load a Parquet file via the project's pure-Java {@code LocalParquetReader}
     * (no Hadoop, no DataFrame/Tensor natives). Nested / list columns become
     * {@link List} / {@link Map} values.
     */
    public static HfDataset fromParquet(Path path) throws IOException {
        return fromParquet(path, -1);
    }

    /**
     * Load a Parquet file, optionally stopping after {@code maxRows} rows
     * ({@code maxRows <= 0} → all rows). Early-stop avoids materialising multi-million
     * row shards when the caller only needs a smoke / stress sample.
     */
    public static HfDataset fromParquet(Path path, int maxRows) throws IOException {
        Objects.requireNonNull(path, "path");
        List<Map<String, Object>> rows = new ArrayList<>();
        try (org.bytedeco.pytorch.data.parquet.LocalParquetReader reader =
                     org.bytedeco.pytorch.data.parquet.LocalParquetReader.open(path.toString())) {
            org.apache.parquet.schema.MessageType schema = reader.getSchema();
            List<String> fields = reader.getFieldNames();
            for (org.apache.parquet.example.data.Group g = reader.read(); g != null; g = reader.read()) {
                Map<String, Object> row = new LinkedHashMap<>(fields.size());
                for (String fname : fields) {
                    row.put(fname, parquetGroupValue(g, fname, schema.getType(fname)));
                }
                rows.add(row);
                if (maxRows > 0 && rows.size() >= maxRows) break;
            }
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to read parquet: " + path, e);
        }
        return new HfDataset(rows, false, "parquet:" + path.getFileName()
                + (maxRows > 0 ? ":take" + maxRows : ""));
    }

    public static HfDataset fromParquet(String path) throws IOException {
        return fromParquet(Path.of(path));
    }

    public static HfDataset fromParquet(String path, int maxRows) throws IOException {
        return fromParquet(Path.of(path), maxRows);
    }

    /**
     * Load Arrow IPC / Feather v2. Prefers {@code DataFrame.readArrow} when natives
     * are available; otherwise fails with a clear message (Arrow bridge needs Arrow + DF).
     */
    public static HfDataset fromArrow(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        try {
            // Reflective call so this class still loads when Tensor natives are absent.
            Class<?> dfCl = Class.forName("org.bytedeco.pytorch.data.dataframe.DataFrame");
            Object df = dfCl.getMethod("readArrow", String.class).invoke(null, path.toString());
            return fromDataFrameReflect(df, "arrow:" + path.getFileName());
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            throw new IOException("Arrow load requires DataFrame on classpath: " + path, e);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            Throwable c = e.getCause() != null ? e.getCause() : e;
            if (c instanceof IOException io) throw io;
            throw new IOException("Failed to read arrow: " + path, c);
        }
    }

    public static HfDataset fromArrow(String path) throws IOException {
        return fromArrow(Path.of(path));
    }

    /**
     * Load ORC via pure-Java {@code OrcInputFormat} (no Hadoop / DataFrame required).
     */
    public static HfDataset fromOrc(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        List<Map<String, Object>> rows = new ArrayList<>();
        try (org.bytedeco.pytorch.data.orc.OrcInputFormat in =
                     org.bytedeco.pytorch.data.orc.OrcInputFormat.open(path.toString())) {
            List<String> fields = in.fieldNames();
            for (Object[] rec = in.read(); rec != null; rec = in.read()) {
                Map<String, Object> row = new LinkedHashMap<>(fields.size());
                for (int i = 0; i < fields.size(); i++) {
                    row.put(fields.get(i), i < rec.length ? rec[i] : null);
                }
                rows.add(row);
            }
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to read ORC: " + path, e);
        }
        return new HfDataset(rows, false, "orc:" + path.getFileName());
    }

    public static HfDataset fromOrc(String path) throws IOException {
        return fromOrc(Path.of(path));
    }

    /**
     * Load Avro data-file via Apache Avro {@code DataFileReader} (no DataFrame required).
     */
    public static HfDataset fromAvro(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        List<Map<String, Object>> rows = new ArrayList<>();
        try {
            org.apache.avro.file.FileReader<org.apache.avro.generic.GenericRecord> reader =
                    org.apache.avro.file.DataFileReader.openReader(
                            path.toFile(),
                            new org.apache.avro.generic.GenericDatumReader<>());
            try (reader) {
                org.apache.avro.Schema schema = reader.getSchema();
                List<org.apache.avro.Schema.Field> fields = schema.getFields();
                while (reader.hasNext()) {
                    org.apache.avro.generic.GenericRecord rec = reader.next();
                    Map<String, Object> row = new LinkedHashMap<>(fields.size());
                    for (org.apache.avro.Schema.Field f : fields) {
                        row.put(f.name(), avroToJava(rec.get(f.name())));
                    }
                    rows.add(row);
                }
            }
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to read Avro: " + path, e);
        }
        return new HfDataset(rows, false, "avro:" + path.getFileName());
    }

    public static HfDataset fromAvro(String path) throws IOException {
        return fromAvro(Path.of(path));
    }

    /** Recursively convert Avro Utf8 / GenericRecord / arrays to Java values. */
    @SuppressWarnings("unchecked")
    static Object avroToJava(Object v) {
        if (v == null) return null;
        if (v instanceof org.apache.avro.util.Utf8 u) return u.toString();
        if (v instanceof CharSequence cs) return cs.toString();
        if (v instanceof org.apache.avro.generic.GenericRecord rec) {
            Map<String, Object> m = new LinkedHashMap<>();
            for (org.apache.avro.Schema.Field f : rec.getSchema().getFields()) {
                m.put(f.name(), avroToJava(rec.get(f.name())));
            }
            return m;
        }
        if (v instanceof org.apache.avro.generic.GenericArray<?> arr) {
            List<Object> list = new ArrayList<>(arr.size());
            for (Object e : arr) list.add(avroToJava(e));
            return list;
        }
        if (v instanceof java.nio.ByteBuffer bb) {
            byte[] bytes = new byte[bb.remaining()];
            bb.duplicate().get(bytes);
            return bytes;
        }
        if (v instanceof Map<?, ?> map) {
            Map<String, Object> m = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : map.entrySet()) {
                m.put(String.valueOf(avroToJava(e.getKey())), avroToJava(e.getValue()));
            }
            return m;
        }
        if (v instanceof List<?> list) {
            List<Object> out = new ArrayList<>(list.size());
            for (Object e : list) out.add(avroToJava(e));
            return out;
        }
        return v;
    }

    /** Load a JSON array-of-objects file (not JSONL). */
    public static HfDataset fromJson(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        String raw = Files.readString(path, StandardCharsets.UTF_8).trim();
        if (raw.isEmpty()) return empty();
        if (raw.startsWith("[")) {
            List<Map<String, Object>> rows = parseJsonArrayOfObjects(raw);
            return new HfDataset(rows, false, "json:" + path.getFileName());
        }
        // single object → one-row dataset; or NDJSON mistaken for .json
        if (raw.startsWith("{")) {
            // multi-line object or single; also handle NDJSON
            if (raw.contains("\n") && !raw.contains("}\n") && countTopLevelObjects(raw) > 1) {
                return fromJsonl(path);
            }
            // try as one object
            try {
                Map<String, Object> row = parseJsonObject(raw);
                return fromList(List.of(row));
            } catch (RuntimeException ex) {
                // fall through to jsonl
                return fromJsonl(path);
            }
        }
        return fromJsonl(path);
    }

    public static HfDataset fromJson(String path) throws IOException {
        return fromJson(Path.of(path));
    }

    /** One plain-text line per row under column {@code "text"}. */
    public static HfDataset fromText(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        List<Map<String, Object>> rows = new ArrayList<>(lines.size());
        for (int i = 0; i < lines.size(); i++) {
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("text", lines.get(i));
            r.put("line", i);
            rows.add(r);
        }
        return new HfDataset(rows, false, "text:" + path.getFileName());
    }

    /**
     * Auto-detect format from extension (and light magic) and load.
     * Supports: {@code .parquet .pq .arrow .feather .ipc .csv .tsv
     * .json .jsonl .ndjson .txt} and gzip-stripped names.
     */
    public static HfDataset fromFile(Path path) throws IOException {
        return fromFile(path, -1);
    }

    /**
     * Like {@link #fromFile(Path)} but forwards {@code maxRows} to formats that
     * support early-stop (currently Parquet / {@code .pq}). Other formats still
     * load fully; callers may {@link #take(int)} afterwards.
     */
    public static HfDataset fromFile(Path path, int maxRows) throws IOException {
        Objects.requireNonNull(path, "path");
        if (!Files.isRegularFile(path)) {
            throw new IOException("Not a file: " + path);
        }
        String name = path.getFileName().toString().toLowerCase();
        String base = stripCompressionSuffix(name);
        if (base.endsWith(".parquet") || base.endsWith(".pq")) return fromParquet(path, maxRows);
        if (base.endsWith(".arrow") || base.endsWith(".feather") || base.endsWith(".ipc")) return fromArrow(path);
        if (base.endsWith(".orc")) return fromOrc(path);
        if (base.endsWith(".avro")) return fromAvro(path);
        if (base.endsWith(".csv")) return fromCsv(path, true);
        if (base.endsWith(".tsv")) return fromTsv(path, true);
        if (base.endsWith(".jsonl") || base.endsWith(".ndjson")) return fromJsonl(path);
        if (base.endsWith(".json")) return fromJson(path);
        if (base.endsWith(".txt") || base.endsWith(".text")) return fromText(path);
        // magic sniff — pure Java where possible
        try {
            byte[] head = new byte[Math.min(8, (int) Math.min(8L, Files.size(path)))];
            try (var in = Files.newInputStream(path)) {
                int n = in.read(head);
                if (n >= 4 && head[0] == 'P' && head[1] == 'A' && head[2] == 'R' && head[3] == '1') {
                    return fromParquet(path, maxRows);
                }
                // ORC: "ORC" at start of file
                if (n >= 3 && head[0] == 'O' && head[1] == 'R' && head[2] == 'C') {
                    return fromOrc(path);
                }
                // Avro object container: Obj\x01
                if (n >= 4 && head[0] == 'O' && head[1] == 'b' && head[2] == 'j' && head[3] == 0x01) {
                    return fromAvro(path);
                }
                // Arrow IPC magic "ARROW1"
                if (n >= 6 && head[0] == 'A' && head[1] == 'R' && head[2] == 'R'
                        && head[3] == 'O' && head[4] == 'W' && head[5] == '1') {
                    return fromArrow(path);
                }
            }
        } catch (Exception ignored) {}
        // last-resort text attempts
        try {
            return fromJsonl(path);
        } catch (Exception e1) {
            try {
                return fromCsv(path, true);
            } catch (Exception e2) {
                throw new IOException("Unsupported dataset file format: " + path
                        + " (supported: parquet,arrow/ipc/feather,orc,avro,csv,tsv,json,jsonl,txt)", e1);
            }
        }
    }

    public static HfDataset fromFile(String path) throws IOException {
        return fromFile(Path.of(path));
    }

    /**
     * Load and concatenate every supported data file under a directory
     * (non-recursive by default). Useful for multi-shard parquet folders.
     */
    public static HfDataset fromDirectory(Path dir, boolean recursive) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!Files.isDirectory(dir)) throw new IOException("Not a directory: " + dir);
        List<Path> files = new ArrayList<>();
        try (Stream<Path> walk = recursive ? Files.walk(dir) : Files.list(dir)) {
            walk.filter(Files::isRegularFile)
                    .filter(p -> org.bytedeco.pytorch.utils.hub.HfHub.isDatasetDataFile(
                            p.getFileName().toString()))
                    .sorted()
                    .forEach(files::add);
        }
        if (files.isEmpty()) {
            // maybe it's a saveToDisk layout
            Path data = dir.resolve("data.jsonl");
            if (Files.isRegularFile(data)) return loadFromDisk(dir);
            throw new IOException("No data files found under " + dir);
        }
        return fromFiles(files);
    }

    /** Concatenate multiple files (same schema expected). */
    public static HfDataset fromFiles(List<Path> files) throws IOException {
        if (files == null || files.isEmpty()) return empty();
        List<Map<String, Object>> all = new ArrayList<>();
        StringBuilder info = new StringBuilder("files[");
        for (int i = 0; i < files.size(); i++) {
            Path f = files.get(i);
            HfDataset part = fromFile(f);
            all.addAll(part.rows);
            if (i > 0) info.append(',');
            info.append(f.getFileName());
        }
        info.append(']');
        return new HfDataset(all, false, info.toString());
    }

    /** Bridge: materialise a {@code DataFrame} into row maps. */
    public static HfDataset fromDataFrame(org.bytedeco.pytorch.data.dataframe.DataFrame df) {
        return fromDataFrame(df, "dataframe");
    }

    public static HfDataset fromDataFrame(org.bytedeco.pytorch.data.dataframe.DataFrame df, String info) {
        Objects.requireNonNull(df, "df");
        List<String> cols = df.getColumnNames();
        int n = df.rowCount();
        List<Map<String, Object>> rows = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>(cols.size());
            for (String c : cols) {
                row.put(c, df.get(i, c));
            }
            rows.add(row);
        }
        return new HfDataset(rows, false, info == null ? "dataframe" : info);
    }

    /** Export to a DataFrame (columnar). */
    public org.bytedeco.pytorch.data.dataframe.DataFrame toDataFrame() {
        if (rows.isEmpty()) {
            return new org.bytedeco.pytorch.data.dataframe.DataFrame();
        }
        // collect columns
        List<String> cols = columnNames.isEmpty()
                ? new ArrayList<>(rows.get(0).keySet())
                : new ArrayList<>(columnNames);
        // ensure all keys across rows
        for (Map<String, Object> r : rows) {
            for (String k : r.keySet()) {
                if (!cols.contains(k)) cols.add(k);
            }
        }
        org.bytedeco.pytorch.data.dataframe.DataFrame df =
                new org.bytedeco.pytorch.data.dataframe.DataFrame();
        for (String c : cols) {
            // infer dtype loosely as STRING/OBJECT-friendly FLOAT64/INT64/BOOL
            org.bytedeco.pytorch.data.dataframe.Column.DType dt = inferColumnDtype(c);
            df.addColumn(c, dt);
            org.bytedeco.pytorch.data.dataframe.Column col = df.column(c);
            for (Map<String, Object> r : rows) {
                col.add(r.get(c));
            }
        }
        df.syncRowCountPublic();
        return df;
    }

    private org.bytedeco.pytorch.data.dataframe.Column.DType inferColumnDtype(String col) {
        boolean sawLong = false, sawDouble = false, sawBool = false;
        boolean sawList = false, sawMap = false, sawOther = false;
        for (Map<String, Object> r : rows) {
            Object v = r.get(col);
            if (v == null) continue;
            if (v instanceof Boolean) sawBool = true;
            else if (v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte) sawLong = true;
            else if (v instanceof Double || v instanceof Float) sawDouble = true;
            else if (v instanceof List || v instanceof long[] || v instanceof int[]
                    || v instanceof float[] || v instanceof double[]) sawList = true;
            else if (v instanceof Map) sawMap = true;
            else { sawOther = true; break; }
        }
        if (sawOther) return org.bytedeco.pytorch.data.dataframe.Column.DType.STRING;
        if (sawMap) return org.bytedeco.pytorch.data.dataframe.Column.DType.MAP;
        if (sawList) return org.bytedeco.pytorch.data.dataframe.Column.DType.LIST;
        if (sawDouble) return org.bytedeco.pytorch.data.dataframe.Column.DType.FLOAT64;
        if (sawLong) return org.bytedeco.pytorch.data.dataframe.Column.DType.INT64;
        if (sawBool) return org.bytedeco.pytorch.data.dataframe.Column.DType.BOOLEAN;
        return org.bytedeco.pytorch.data.dataframe.Column.DType.STRING;
    }

    // ---- javacpp-pytorch Dataset / DataLoader interop -----------------------

    /**
     * Convert to a {@link org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset}
     * with automatic column roles:
     * <ul>
     *   <li>label-like columns ({@code label}, {@code labels}, {@code target}, {@code y},
     *       {@code class}, {@code class_label}) → labels</li>
     *   <li>remaining numeric / bool / list columns → features (lists → sequence features)</li>
     *   <li>string columns stay in the frame but are not packed as float features
     *       (tokenize first for NLP training)</li>
     * </ul>
     */
    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset asDataFrameDataset()
            throws Exception {
        return asDataFrameDataset(null, null);
    }

    /**
     * Convert with explicit feature / label column names.
     * {@code null} features → all non-label columns; {@code null} labels → auto-detect.
     */
    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset asDataFrameDataset(
            String[] featureCols, String[] labelCols) throws Exception {
        org.bytedeco.pytorch.data.dataframe.DataFrame df = toDataFrame();
        var b = org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataset.builder(df);
        String[] labs = labelCols;
        if (labs == null || labs.length == 0) {
            labs = detectLabelColumns(df.getColumnNames()).toArray(new String[0]);
        }
        if (labs.length > 0) b.labels(labs);
        if (featureCols != null && featureCols.length > 0) {
            // split scalar vs sequence by dtype
            List<String> scalars = new ArrayList<>();
            List<String> seqs = new ArrayList<>();
            for (String c : featureCols) {
                if (!df.hasColumn(c)) continue;
                org.bytedeco.pytorch.data.dataframe.Column.DType dt = df.column(c).dtype();
                if (dt == org.bytedeco.pytorch.data.dataframe.Column.DType.LIST
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.VECTOR
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.EMBEDDING
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.TENSOR) {
                    seqs.add(c);
                } else if (dt != org.bytedeco.pytorch.data.dataframe.Column.DType.STRING) {
                    scalars.add(c);
                }
            }
            if (!scalars.isEmpty()) b.features(scalars.toArray(new String[0]));
            if (!seqs.isEmpty()) b.sequenceFeatures(seqs.toArray(new String[0]));
            // if user only passed string cols, still register them so builder doesn't
            // re-default — pack as empty features + keep labels
            if (scalars.isEmpty() && seqs.isEmpty() && featureCols.length > 0) {
                b.features(featureCols); // may zero-fill strings; documented
            }
        } else {
            // numeric / list features only (exclude strings + labels)
            java.util.Set<String> labSet = new java.util.HashSet<>(java.util.Arrays.asList(labs));
            List<String> scalars = new ArrayList<>();
            List<String> seqs = new ArrayList<>();
            for (String c : df.getColumnNames()) {
                if (labSet.contains(c)) continue;
                org.bytedeco.pytorch.data.dataframe.Column.DType dt = df.column(c).dtype();
                if (dt == org.bytedeco.pytorch.data.dataframe.Column.DType.LIST
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.VECTOR
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.EMBEDDING
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.TENSOR) {
                    seqs.add(c);
                } else if (dt == org.bytedeco.pytorch.data.dataframe.Column.DType.INT32
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.INT64
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.FLOAT32
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.FLOAT64
                        || dt == org.bytedeco.pytorch.data.dataframe.Column.DType.BOOLEAN) {
                    scalars.add(c);
                }
            }
            // DataFrameDataset.Builder re-defaults to "all non-label columns" when both
            // feature and sequence arrays are empty — that would pack STRING text as 0f.
            // If we only have text (common for HF NLP sets), inject a synthetic row id
            // so the Dataset/DataLoader path stays valid without polluting features.
            if (scalars.isEmpty() && seqs.isEmpty()) {
                String rid = "__row_id";
                if (!df.hasColumn(rid)) {
                    // addColumn pads with nulls to current rowCount — set in place.
                    df.addColumn(rid, org.bytedeco.pytorch.data.dataframe.Column.DType.INT64);
                    var col = df.column(rid);
                    for (int i = 0; i < df.rowCount(); i++) col.set(i, (long) i);
                }
                scalars.add(rid);
            }
            b.features(scalars.toArray(new String[0]));
            if (!seqs.isEmpty()) b.sequenceFeatures(seqs.toArray(new String[0]));
        }
        return b.build();
    }

    /**
     * Native {@link org.bytedeco.pytorch.data.Dataset} adapter
     * ({@code Example(data, target)}) for {@code RandomDataLoader} /
     * {@code SequentialDataLoader}.
     */
    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameNativeDataset asDataset()
            throws Exception {
        return asDataFrameDataset().asDataset();
    }

    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameNativeDataset asDataset(
            String[] featureCols, String[] labelCols) throws Exception {
        return asDataFrameDataset(featureCols, labelCols).asDataset();
    }

    /** Features-only {@link org.bytedeco.pytorch.data.datasets.JavaTensorDataset}. */
    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameJavaTensorDataset asJavaTensorDataset()
            throws Exception {
        return asDataFrameDataset().asJavaTensorDataset();
    }

    /**
     * Pure-Java multi-feature {@link org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataLoader}
     * builder (named batches, shuffle, dropLast).
     */
    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataLoader.Builder dataloader()
            throws Exception {
        return asDataFrameDataset().dataloader();
    }

    public org.bytedeco.pytorch.data.dataframe.dataset.DataFrameDataLoader dataloader(int batchSize)
            throws Exception {
        return asDataFrameDataset().dataloader(batchSize);
    }

    /**
     * Native DataLoader builder over {@link #asDataset()}.
     *
     * <pre>{@code
     * SequentialDataLoader loader = ds.nativeDataLoader().batchSize(32).buildSequential();
     * }</pre>
     */
    public org.bytedeco.pytorch.data.dataframe.dataset.NativeDataLoaderBuilder nativeDataLoader()
            throws Exception {
        return asDataFrameDataset().nativeDataLoader();
    }

    /** Materialize a {@link org.bytedeco.pytorch.data.datasets.TensorDataset}. */
    public org.bytedeco.pytorch.data.datasets.TensorDataset toTensorDataset() throws Exception {
        return asDataFrameDataset().toTensorDataset();
    }

    /**
     * Encode a string column to contiguous integer ids (for classification labels or
     * categorical features). Returns a new dataset with column {@code outCol}.
     */
    public HfDataset encodeColumn(String column, String outCol) {
        Objects.requireNonNull(column, "column");
        String out = outCol == null || outCol.isBlank() ? column + "_id" : outCol;
        Map<String, Integer> vocab = new LinkedHashMap<>();
        List<Map<String, Object>> outRows = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) {
            Map<String, Object> n = new LinkedHashMap<>(r);
            Object v = r.get(column);
            String key = v == null ? "<null>" : String.valueOf(v);
            int id = vocab.computeIfAbsent(key, k -> vocab.size());
            n.put(out, (long) id);
            outRows.add(n);
        }
        return new HfDataset(outRows, false, info + "/encode(" + column + ")");
    }

    /** Pretty-print first {@code n} rows for verification. */
    public String headString(int n) {
        int m = Math.min(Math.max(0, n), rows.size());
        StringBuilder sb = new StringBuilder();
        sb.append(this).append('\n');
        for (int i = 0; i < m; i++) {
            sb.append("  [").append(i).append("] ").append(formatRow(rows.get(i))).append('\n');
        }
        if (rows.size() > m) sb.append("  ... (").append(rows.size() - m).append(" more)\n");
        return sb.toString();
    }

    private static String formatRow(Map<String, Object> row) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : row.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            sb.append(e.getKey()).append('=');
            Object v = e.getValue();
            if (v == null) sb.append("null");
            else if (v instanceof String s) {
                String t = s.length() > 80 ? s.substring(0, 77) + "..." : s;
                sb.append('"').append(t.replace("\n", "\\n")).append('"');
            } else if (v instanceof Map || v instanceof List) {
                String t = String.valueOf(v);
                if (t.length() > 100) t = t.substring(0, 97) + "...";
                sb.append(t);
            } else {
                sb.append(v);
            }
        }
        return sb.append('}').toString();
    }

    static List<String> detectLabelColumns(List<String> cols) {
        List<String> preferred = List.of(
                "label", "labels", "target", "targets", "y",
                "class", "class_label", "label_id", "cls");
        List<String> found = new ArrayList<>();
        for (String p : preferred) {
            for (String c : cols) {
                if (c.equalsIgnoreCase(p) && !found.contains(c)) found.add(c);
            }
        }
        return found;
    }

    /** Concatenate datasets (vertical stack). */
    public static HfDataset concatenate(HfDataset... parts) {
        List<Map<String, Object>> all = new ArrayList<>();
        StringBuilder info = new StringBuilder("concat");
        if (parts != null) {
            for (HfDataset p : parts) {
                if (p == null) continue;
                for (Map<String, Object> r : p.rows) all.add(new LinkedHashMap<>(r));
            }
        }
        return new HfDataset(all, false, info.toString());
    }

    public static HfDataset concatenate(List<HfDataset> parts) {
        if (parts == null || parts.isEmpty()) return empty();
        return concatenate(parts.toArray(new HfDataset[0]));
    }

    // ---- pure-Java Parquet Group → Map (no DataFrame / Tensor) -------------

    @SuppressWarnings("unchecked")
    private static HfDataset fromDataFrameReflect(Object df, String info) throws Exception {
        if (df == null) return empty();
        // Prefer typed path when the concrete class is already loaded.
        if (df instanceof org.bytedeco.pytorch.data.dataframe.DataFrame typed) {
            return fromDataFrame(typed, info);
        }
        List<String> cols = (List<String>) df.getClass().getMethod("getColumnNames").invoke(df);
        int n = ((Number) df.getClass().getMethod("rowCount").invoke(df)).intValue();
        List<Map<String, Object>> rows = new ArrayList<>(n);
        var get = df.getClass().getMethod("get", int.class, String.class);
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>(cols.size());
            for (String c : cols) row.put(c, get.invoke(df, i, c));
            rows.add(row);
        }
        return new HfDataset(rows, false, info == null ? "dataframe" : info);
    }

    static Object parquetGroupValue(org.apache.parquet.example.data.Group row,
                                    String field, org.apache.parquet.schema.Type ft) {
        int fieldIndex;
        try {
            fieldIndex = row.getType().getFieldIndex(field);
        } catch (Exception e) {
            return null;
        }
        int n = row.getFieldRepetitionCount(fieldIndex);
        if (n == 0) return null;

        if (ft.isPrimitive()) {
            if (ft.getRepetition() == org.apache.parquet.schema.Type.Repetition.REPEATED) {
                List<Object> out = new ArrayList<>(n);
                for (int i = 0; i < n; i++) out.add(parquetPrimitiveAt(row, fieldIndex, i, ft));
                return out;
            }
            return parquetPrimitiveAt(row, fieldIndex, 0, ft);
        }

        org.apache.parquet.schema.LogicalTypeAnnotation lta = ft.getLogicalTypeAnnotation();
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.ListLogicalTypeAnnotation
                || ft.getRepetition() == org.apache.parquet.schema.Type.Repetition.REPEATED) {
            return parquetReadList(row, fieldIndex, ft);
        }
        if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapLogicalTypeAnnotation
                || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.MapKeyValueTypeAnnotation) {
            return parquetReadMap(row, fieldIndex, ft);
        }
        // STRUCT
        org.apache.parquet.example.data.Group g = row.getGroup(fieldIndex, 0);
        return parquetReadStruct(g);
    }

    private static Object parquetPrimitiveAt(org.apache.parquet.example.data.Group row,
                                             int fieldIndex, int idx,
                                             org.apache.parquet.schema.Type ft) {
        org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName ptn =
                ft.asPrimitiveType().getPrimitiveTypeName();
        try {
            return switch (ptn) {
                case INT32 -> row.getInteger(fieldIndex, idx);
                case INT64 -> row.getLong(fieldIndex, idx);
                case FLOAT -> Float.valueOf(row.getFloat(fieldIndex, idx));
                case DOUBLE -> row.getDouble(fieldIndex, idx);
                case BOOLEAN -> Boolean.valueOf(row.getBoolean(fieldIndex, idx));
                case BINARY, FIXED_LEN_BYTE_ARRAY -> {
                    org.apache.parquet.schema.LogicalTypeAnnotation lta = ft.getLogicalTypeAnnotation();
                    org.apache.parquet.io.api.Binary bin = row.getBinary(fieldIndex, idx);
                    if (lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.StringLogicalTypeAnnotation
                            || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.EnumLogicalTypeAnnotation
                            || lta instanceof org.apache.parquet.schema.LogicalTypeAnnotation.JsonLogicalTypeAnnotation
                            || lta == null) {
                        yield bin.toStringUsingUTF8();
                    }
                    yield bin.getBytes();
                }
                case INT96 -> row.getInt96(fieldIndex, idx).toStringUsingUTF8();
                default -> row.getValueToString(fieldIndex, idx);
            };
        } catch (Exception e) {
            try { return row.getValueToString(fieldIndex, idx); }
            catch (Exception e2) { return null; }
        }
    }

    private static Object parquetReadList(org.apache.parquet.example.data.Group row,
                                          int fieldIndex, org.apache.parquet.schema.Type ft) {
        org.apache.parquet.schema.GroupType listType = ft.asGroupType();
        if (row.getFieldRepetitionCount(fieldIndex) == 0) return null;
        List<Object> elems = new ArrayList<>();

        // Standard 3-level LIST: outer optional group → repeated list → element
        // or 2-level: outer → repeated element
        if (ft.getRepetition() == org.apache.parquet.schema.Type.Repetition.REPEATED) {
            int n = row.getFieldRepetitionCount(fieldIndex);
            for (int i = 0; i < n; i++) {
                if (listType.getFieldCount() == 0) continue;
                org.apache.parquet.schema.Type child = listType.getType(0);
                if (listType.getFieldCount() == 1 && child.isPrimitive()) {
                    org.apache.parquet.example.data.Group g = row.getGroup(fieldIndex, i);
                    if (g.getFieldRepetitionCount(0) == 0) elems.add(null);
                    else elems.add(parquetPrimitiveAt(g, 0, 0, child));
                } else {
                    elems.add(parquetReadStruct(row.getGroup(fieldIndex, i)));
                }
            }
            return elems;
        }

        org.apache.parquet.example.data.Group outer = row.getGroup(fieldIndex, 0);
        if (listType.getFieldCount() == 0) return elems;
        org.apache.parquet.schema.Type mid = listType.getType(0);
        int midIdx = 0;
        int rep = outer.getFieldRepetitionCount(midIdx);
        if (mid.isPrimitive()) {
            for (int i = 0; i < rep; i++) elems.add(parquetPrimitiveAt(outer, midIdx, i, mid));
            return elems;
        }
        // mid is repeated group "list" { element }
        org.apache.parquet.schema.GroupType midG = mid.asGroupType();
        for (int i = 0; i < rep; i++) {
            org.apache.parquet.example.data.Group listItem = outer.getGroup(midIdx, i);
            if (midG.getFieldCount() == 0) {
                elems.add(null);
                continue;
            }
            org.apache.parquet.schema.Type elem = midG.getType(0);
            if (listItem.getFieldRepetitionCount(0) == 0) {
                elems.add(null);
            } else if (elem.isPrimitive()) {
                elems.add(parquetPrimitiveAt(listItem, 0, 0, elem));
            } else {
                elems.add(parquetReadStruct(listItem.getGroup(0, 0)));
            }
        }
        return elems;
    }

    private static Map<String, Object> parquetReadMap(org.apache.parquet.example.data.Group row,
                                                      int fieldIndex, org.apache.parquet.schema.Type ft) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (row.getFieldRepetitionCount(fieldIndex) == 0) return out;
        org.apache.parquet.example.data.Group outer = row.getGroup(fieldIndex, 0);
        org.apache.parquet.schema.GroupType outerType = ft.asGroupType();
        if (outerType.getFieldCount() == 0) return out;
        // key_value repeated group
        int kvIdx = 0;
        int rep = outer.getFieldRepetitionCount(kvIdx);
        org.apache.parquet.schema.GroupType kvType = outerType.getType(0).asGroupType();
        for (int i = 0; i < rep; i++) {
            org.apache.parquet.example.data.Group kv = outer.getGroup(kvIdx, i);
            Object key = null;
            Object val = null;
            if (kvType.getFieldCount() >= 1 && kv.getFieldRepetitionCount(0) > 0) {
                org.apache.parquet.schema.Type kt = kvType.getType(0);
                key = kt.isPrimitive() ? parquetPrimitiveAt(kv, 0, 0, kt) : parquetReadStruct(kv.getGroup(0, 0));
            }
            if (kvType.getFieldCount() >= 2 && kv.getFieldRepetitionCount(1) > 0) {
                org.apache.parquet.schema.Type vt = kvType.getType(1);
                val = vt.isPrimitive() ? parquetPrimitiveAt(kv, 1, 0, vt) : parquetReadStruct(kv.getGroup(1, 0));
            }
            out.put(String.valueOf(key), val);
        }
        return out;
    }

    private static Map<String, Object> parquetReadStruct(org.apache.parquet.example.data.Group g) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (g == null) return out;
        org.apache.parquet.schema.GroupType gt = g.getType();
        for (int i = 0; i < gt.getFieldCount(); i++) {
            org.apache.parquet.schema.Type t = gt.getType(i);
            out.put(t.getName(), parquetGroupValue(g, t.getName(), t));
        }
        return out;
    }

    public int size() {
        return rows.size();
    }

    public int numRows() {
        return rows.size();
    }

    public List<String> columnNames() {
        return Collections.unmodifiableList(columnNames);
    }

    public boolean isStreaming() {
        return streaming;
    }

    public String info() {
        return info;
    }

    public Map<String, Object> get(int index) {
        return Collections.unmodifiableMap(rows.get(index));
    }

    public Map<String, Object> getMutable(int index) {
        return rows.get(index);
    }

    @Override
    public Iterator<Map<String, Object>> iterator() {
        return rows.stream().map(Collections::unmodifiableMap).iterator();
    }

    public Stream<Map<String, Object>> stream() {
        return rows.stream().map(Collections::unmodifiableMap);
    }

    public HfDataset select(int... indices) {
        List<Map<String, Object>> out = new ArrayList<>(indices.length);
        for (int i : indices) {
            out.add(new LinkedHashMap<>(rows.get(i)));
        }
        return new HfDataset(out, false, info + "/select");
    }

    public HfDataset selectColumns(String... cols) {
        List<String> keep = Arrays.asList(cols);
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) {
            Map<String, Object> n = new LinkedHashMap<>();
            for (String c : keep) {
                if (r.containsKey(c)) n.put(c, r.get(c));
            }
            out.add(n);
        }
        return new HfDataset(out, false, info + "/selectColumns");
    }

    public HfDataset removeColumns(String... cols) {
        List<String> drop = Arrays.asList(cols);
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) {
            Map<String, Object> n = new LinkedHashMap<>(r);
            for (String c : drop) n.remove(c);
            out.add(n);
        }
        return new HfDataset(out, false, info + "/removeColumns");
    }

    public HfDataset renameColumn(String oldName, String newName) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) {
            Map<String, Object> n = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : r.entrySet()) {
                n.put(e.getKey().equals(oldName) ? newName : e.getKey(), e.getValue());
            }
            out.add(n);
        }
        return new HfDataset(out, false, info + "/rename");
    }

    public HfDataset filter(Predicate<Map<String, Object>> pred) {
        List<Map<String, Object>> out = rows.stream()
                .filter(pred)
                .map(LinkedHashMap::new)
                .collect(Collectors.toCollection(ArrayList::new));
        return new HfDataset(out, false, info + "/filter");
    }

    public HfDataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) {
            Map<String, Object> mapped = fn.apply(new LinkedHashMap<>(r));
            out.add(mapped == null ? new LinkedHashMap<>() : new LinkedHashMap<>(mapped));
        }
        return new HfDataset(out, false, info + "/map");
    }

    public HfDataset mapColumn(String column, Function<Object, Object> fn) {
        return map(r -> {
            if (r.containsKey(column)) {
                r.put(column, fn.apply(r.get(column)));
            }
            return r;
        });
    }

    public HfDataset shuffle(long seed) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) out.add(new LinkedHashMap<>(r));
        Collections.shuffle(out, new Random(seed));
        return new HfDataset(out, false, info + "/shuffle");
    }

    public HfDataset shard(int numShards, int index) {
        if (numShards <= 0) throw new IllegalArgumentException("numShards must be > 0");
        if (index < 0 || index >= numShards) throw new IllegalArgumentException("index out of range");
        List<Map<String, Object>> out = new ArrayList<>();
        for (int i = index; i < rows.size(); i += numShards) {
            out.add(new LinkedHashMap<>(rows.get(i)));
        }
        return new HfDataset(out, false, info + "/shard(" + index + "/" + numShards + ")");
    }

    public HfDataset take(int n) {
        int m = Math.min(n, rows.size());
        List<Map<String, Object>> out = new ArrayList<>(m);
        for (int i = 0; i < m; i++) out.add(new LinkedHashMap<>(rows.get(i)));
        return new HfDataset(out, false, info + "/take");
    }

    public HfDataset skip(int n) {
        int start = Math.min(Math.max(0, n), rows.size());
        List<Map<String, Object>> out = new ArrayList<>(rows.size() - start);
        for (int i = start; i < rows.size(); i++) out.add(new LinkedHashMap<>(rows.get(i)));
        return new HfDataset(out, false, info + "/skip");
    }

    public DatasetDict trainTestSplit(double testSize, long seed) {
        if (testSize < 0 || testSize > 1) throw new IllegalArgumentException("testSize in [0,1]");
        HfDataset shuffled = shuffle(seed);
        int nTest = (int) Math.round(shuffled.size() * testSize);
        nTest = Math.min(nTest, shuffled.size());
        int nTrain = shuffled.size() - nTest;
        return new DatasetDict(Map.of(
                "train", shuffled.take(nTrain),
                "test", shuffled.skip(nTrain)
        ));
    }

    public List<?> column(String name) {
        List<Object> col = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) col.add(r.get(name));
        return col;
    }

    public void toJsonl(Path path) throws IOException {
        Files.createDirectories(path.getParent() == null ? Path.of(".") : path.getParent());
        StringBuilder sb = new StringBuilder();
        for (Map<String, Object> r : rows) {
            sb.append(toJsonObject(r)).append('\n');
        }
        Files.writeString(path, sb.toString(), StandardCharsets.UTF_8);
    }

    /** Persist a simple cache marker + jsonl under {@code cacheDir/key}. */
    public HfDataset saveToDisk(Path dir) throws IOException {
        Files.createDirectories(dir);
        toJsonl(dir.resolve("data.jsonl"));
        Files.writeString(dir.resolve("dataset_info.json"),
                "{\"info\":" + jsonEscape(info) + ",\"num_rows\":" + rows.size() + "}",
                StandardCharsets.UTF_8);
        return this;
    }

    public static HfDataset loadFromDisk(Path dir) throws IOException {
        Path data = dir.resolve("data.jsonl");
        if (!Files.isRegularFile(data)) {
            throw new IOException("Not an HfDataset dir (missing data.jsonl): " + dir);
        }
        HfDataset ds = fromJsonl(data);
        return new HfDataset(ds.rows, false, "loadFromDisk:" + dir.getFileName());
    }

    public Map<String, Object> features() {
        Map<String, Object> f = new LinkedHashMap<>();
        if (rows.isEmpty()) return f;
        Map<String, Object> sample = rows.get(0);
        for (Map.Entry<String, Object> e : sample.entrySet()) {
            Object v = e.getValue();
            String t = v == null ? "null"
                    : v instanceof Number ? "numeric"
                    : v instanceof Boolean ? "bool"
                    : v instanceof List ? "sequence"
                    : "string";
            f.put(e.getKey(), t);
        }
        return f;
    }

    @Override
    public String toString() {
        return "HfDataset{rows=" + rows.size() + ", cols=" + columnNames + ", info=" + info + "}";
    }

    // ---- DatasetDict -----------------------------------------------------

    public static final class DatasetDict {
        private final Map<String, HfDataset> splits;

        public DatasetDict(Map<String, HfDataset> splits) {
            this.splits = new LinkedHashMap<>(Objects.requireNonNull(splits, "splits"));
        }

        public HfDataset get(String split) {
            HfDataset ds = splits.get(split);
            if (ds == null) throw new IllegalArgumentException("No split: " + split);
            return ds;
        }

        public HfDataset train() { return get("train"); }
        public HfDataset test() { return get("test"); }
        public HfDataset validation() {
            if (splits.containsKey("validation")) return get("validation");
            return get("val");
        }

        public Map<String, HfDataset> splits() {
            return Collections.unmodifiableMap(splits);
        }

        public DatasetDict map(Function<Map<String, Object>, Map<String, Object>> fn) {
            Map<String, HfDataset> out = new LinkedHashMap<>();
            for (Map.Entry<String, HfDataset> e : splits.entrySet()) {
                out.put(e.getKey(), e.getValue().map(fn));
            }
            return new DatasetDict(out);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("DatasetDict({");
            boolean first = true;
            for (Map.Entry<String, HfDataset> e : splits.entrySet()) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(e.getKey()).append(": ").append(e.getValue().size()).append(" rows");
            }
            return sb.append("})").toString();
        }
    }

    // ---- json / csv helpers (minimal, no external deps) ------------------

    private static List<String> splitDelimited(String line, char delimiter) {
        List<String> cells = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQ = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                // RFC-style escaped quote ""
                if (inQ && i + 1 < line.length() && line.charAt(i + 1) == '"') {
                    cur.append('"');
                    i++;
                } else {
                    inQ = !inQ;
                }
            } else if (c == delimiter && !inQ) {
                cells.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        cells.add(cur.toString());
        return cells;
    }

    @Deprecated
    private static List<String> splitCsv(String line) {
        return splitDelimited(line, ',');
    }

    static String stripCompressionSuffix(String name) {
        if (name == null) return "";
        String n = name.toLowerCase();
        for (String c : new String[]{".gz", ".bz2", ".zst", ".xz"}) {
            if (n.endsWith(c)) return n.substring(0, n.length() - c.length());
        }
        return n;
    }

    static List<Map<String, Object>> parseJsonArrayOfObjects(String json) {
        List<Map<String, Object>> rows = new ArrayList<>();
        String s = json.trim();
        if (!s.startsWith("[")) throw new IllegalArgumentException("Expected JSON array");
        int i = 1;
        while (i < s.length()) {
            while (i < s.length() && (Character.isWhitespace(s.charAt(i)) || s.charAt(i) == ',')) i++;
            if (i >= s.length() || s.charAt(i) == ']') break;
            if (s.charAt(i) != '{') {
                // skip non-object values
                ParseResult pr = parseValue(s, i);
                i = pr.next;
                continue;
            }
            int start = i;
            int depth = 0;
            for (; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == '{') depth++;
                else if (c == '}') {
                    depth--;
                    if (depth == 0) { i++; break; }
                } else if (c == '"') {
                    i++;
                    while (i < s.length() && s.charAt(i) != '"') {
                        if (s.charAt(i) == '\\') i++;
                        i++;
                    }
                }
            }
            rows.add(parseJsonObject(s.substring(start, i)));
        }
        return rows;
    }

    static int countTopLevelObjects(String raw) {
        int n = 0;
        for (String line : raw.split("\n")) {
            String t = line.trim();
            if (t.startsWith("{")) n++;
        }
        return n;
    }

    private static Object autoType(String s) {
        if (s == null) return null;
        String t = s.trim();
        if (t.isEmpty()) return "";
        if ("true".equalsIgnoreCase(t) || "false".equalsIgnoreCase(t)) {
            return Boolean.parseBoolean(t);
        }
        try {
            if (t.indexOf('.') >= 0 || t.indexOf('e') >= 0 || t.indexOf('E') >= 0) {
                return Double.parseDouble(t);
            }
            return Long.parseLong(t);
        } catch (NumberFormatException e) {
            return t;
        }
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseJsonObject(String json) {
        // Extremely small subset parser for flat / one-level objects & arrays of scalars.
        json = json.trim();
        if (!json.startsWith("{")) throw new IllegalArgumentException("Expected object: " + json);
        Map<String, Object> map = new LinkedHashMap<>();
        int i = 1;
        while (i < json.length()) {
            while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
            if (i < json.length() && json.charAt(i) == '}') break;
            if (json.charAt(i) != '"') throw new IllegalArgumentException("Expected key at " + i);
            int keyStart = ++i;
            while (i < json.length() && json.charAt(i) != '"') {
                if (json.charAt(i) == '\\') i++;
                i++;
            }
            String key = json.substring(keyStart, i);
            i++; // skip "
            while (i < json.length() && json.charAt(i) != ':') i++;
            i++; // skip :
            while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
            ParseResult pr = parseValue(json, i);
            map.put(key, pr.value);
            i = pr.next;
            while (i < json.length() && (Character.isWhitespace(json.charAt(i)) || json.charAt(i) == ',')) i++;
        }
        return map;
    }

    private static final class ParseResult {
        final Object value;
        final int next;
        ParseResult(Object value, int next) { this.value = value; this.next = next; }
    }

    private static ParseResult parseValue(String json, int i) {
        while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
        if (i >= json.length()) return new ParseResult(null, i);
        char c = json.charAt(i);
        if (c == '"') {
            int start = ++i;
            StringBuilder sb = new StringBuilder();
            while (i < json.length() && json.charAt(i) != '"') {
                if (json.charAt(i) == '\\' && i + 1 < json.length()) {
                    sb.append(json.charAt(++i));
                    i++;
                } else {
                    sb.append(json.charAt(i++));
                }
            }
            return new ParseResult(sb.toString(), i + 1);
        }
        if (c == '{') {
            // nested object — find matching brace naively
            int depth = 0, start = i;
            for (; i < json.length(); i++) {
                if (json.charAt(i) == '{') depth++;
                else if (json.charAt(i) == '}') {
                    depth--;
                    if (depth == 0) {
                        Map<String, Object> nested = parseJsonObject(json.substring(start, i + 1));
                        return new ParseResult(nested, i + 1);
                    }
                }
            }
        }
        if (c == '[') {
            List<Object> list = new ArrayList<>();
            i++;
            while (i < json.length()) {
                while (i < json.length() && (Character.isWhitespace(json.charAt(i)) || json.charAt(i) == ',')) i++;
                if (i < json.length() && json.charAt(i) == ']') return new ParseResult(list, i + 1);
                ParseResult pr = parseValue(json, i);
                list.add(pr.value);
                i = pr.next;
            }
            return new ParseResult(list, i);
        }
        if (json.startsWith("null", i)) return new ParseResult(null, i + 4);
        if (json.startsWith("true", i)) return new ParseResult(Boolean.TRUE, i + 4);
        if (json.startsWith("false", i)) return new ParseResult(Boolean.FALSE, i + 5);
        int start = i;
        while (i < json.length() && "+-0123456789.eE".indexOf(json.charAt(i)) >= 0) i++;
        String num = json.substring(start, i);
        if (num.indexOf('.') >= 0 || num.indexOf('e') >= 0 || num.indexOf('E') >= 0) {
            return new ParseResult(Double.parseDouble(num), i);
        }
        return new ParseResult(Long.parseLong(num), i);
    }

    static String toJsonObject(Map<String, Object> row) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> e : row.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append(jsonEscape(e.getKey())).append(':').append(toJsonValue(e.getValue()));
        }
        sb.append('}');
        return sb.toString();
    }

    private static String toJsonValue(Object v) {
        if (v == null) return "null";
        if (v instanceof Boolean || v instanceof Number) return v.toString();
        if (v instanceof List<?> list) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(toJsonValue(list.get(i)));
            }
            return sb.append(']').toString();
        }
        if (v instanceof Map<?, ?> m) {
            @SuppressWarnings("unchecked")
            Map<String, Object> mm = (Map<String, Object>) m;
            return toJsonObject(mm);
        }
        return jsonEscape(String.valueOf(v));
    }

    private static String jsonEscape(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
