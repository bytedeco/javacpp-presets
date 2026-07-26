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
import org.bytedeco.pytorch.jit.*;

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
 * HuggingFace {@code datasets}-style in-memory / streaming dataset.
 *
 * <p>Rows are {@code Map&lt;String, Object&gt;} feature dicts. Supports map/filter,
 * train/test split, sharding, JSONL/CSV load, and simple disk cache.
 *
 * <pre>{@code
 * HfDataset ds = HfDataset.fromList(rows);
 * ds = ds.map(r -> { r.put("len", r.get("text").toString().length()); return r; });
 * DatasetDict split = ds.trainTestSplit(0.2, 42L);
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
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        if (lines.isEmpty()) return empty();
        List<String> headers;
        int start;
        if (hasHeader) {
            headers = splitCsv(lines.get(0));
            start = 1;
        } else {
            int cols = splitCsv(lines.get(0)).size();
            headers = new ArrayList<>(cols);
            for (int i = 0; i < cols; i++) headers.add("col" + i);
            start = 0;
        }
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = start; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;
            List<String> cells = splitCsv(line);
            Map<String, Object> row = new LinkedHashMap<>();
            for (int c = 0; c < headers.size(); c++) {
                row.put(headers.get(c), c < cells.size() ? autoType(cells.get(c)) : null);
            }
            rows.add(row);
        }
        return new HfDataset(rows, false, "csv:" + path.getFileName());
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

    // ---- json helpers (minimal, no external deps) ------------------------

    private static List<String> splitCsv(String line) {
        List<String> cells = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQ = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                inQ = !inQ;
            } else if (c == ',' && !inQ) {
                cells.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        cells.add(cur.toString());
        return cells;
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
    static Map<String, Object> parseJsonObject(String json) {
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
