/*
 * In-memory + optional file-backed offline store.
 * File mode writes JSONL under {root}/{project}/{view}/data.jsonl for demos.
 */
package org.bytedeco.pytorch.feature.offline;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/** Default offline store used by FeaturePlatform. */
public final class FileOfflineStore implements OfflineStore {

    private final ConcurrentHashMap<String, CopyOnWriteArrayList<Map<String, Object>>> tables =
            new ConcurrentHashMap<>();
    private final Path root; // nullable → pure memory

    public FileOfflineStore() {
        this.root = null;
    }

    public FileOfflineStore(Path root) {
        this.root = root;
        if (root != null) {
            try {
                Files.createDirectories(root);
                loadAll();
            } catch (IOException e) {
                throw new IllegalStateException("FileOfflineStore init failed: " + root, e);
            }
        }
    }

    public static FileOfflineStore inMemory() {
        return new FileOfflineStore();
    }

    private static String key(String project, String viewName) {
        return (project == null || project.isEmpty() ? "default" : project) + "/" + viewName;
    }

    private CopyOnWriteArrayList<Map<String, Object>> table(String project, String viewName) {
        return tables.computeIfAbsent(key(project, viewName), k -> new CopyOnWriteArrayList<>());
    }

    @Override
    public void put(String project, String viewName, List<Map<String, Object>> rows) {
        if (rows == null || rows.isEmpty()) return;
        CopyOnWriteArrayList<Map<String, Object>> t = table(project, viewName);
        for (Map<String, Object> r : rows) {
            t.add(Collections.unmodifiableMap(new LinkedHashMap<>(r)));
        }
        persistAppend(project, viewName, rows);
    }

    @Override
    public void replace(String project, String viewName, List<Map<String, Object>> rows) {
        CopyOnWriteArrayList<Map<String, Object>> t = new CopyOnWriteArrayList<>();
        if (rows != null) {
            for (Map<String, Object> r : rows) {
                t.add(Collections.unmodifiableMap(new LinkedHashMap<>(r)));
            }
        }
        tables.put(key(project, viewName), t);
        persistReplace(project, viewName, rows != null ? rows : List.of());
    }

    @Override
    public List<Map<String, Object>> readAll(String project, String viewName) {
        return new ArrayList<>(table(project, viewName));
    }

    @Override
    public List<Map<String, Object>> readRange(String project, String viewName,
                                               Instant start, Instant end,
                                               String timestampColumn) {
        String tsCol = timestampColumn != null ? timestampColumn : "event_timestamp";
        long s = start != null ? start.toEpochMilli() : Long.MIN_VALUE;
        long e = end != null ? end.toEpochMilli() : Long.MAX_VALUE;
        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> row : table(project, viewName)) {
            long ts = toEpochMillis(row.get(tsCol));
            if (ts >= s && ts <= e) out.add(row);
        }
        return out;
    }

    @Override
    public Optional<Long> latestTimestamp(String project, String viewName, String timestampColumn) {
        String tsCol = timestampColumn != null ? timestampColumn : "event_timestamp";
        long max = Long.MIN_VALUE;
        boolean any = false;
        for (Map<String, Object> row : table(project, viewName)) {
            long ts = toEpochMillis(row.get(tsCol));
            if (ts > max) {
                max = ts;
                any = true;
            }
        }
        return any ? Optional.of(max) : Optional.empty();
    }

    @Override
    public long rowCount(String project, String viewName) {
        return table(project, viewName).size();
    }

    public static long toEpochMillis(Object v) {
        if (v == null) return 0L;
        if (v instanceof Number) return ((Number) v).longValue();
        if (v instanceof Instant) return ((Instant) v).toEpochMilli();
        if (v instanceof String) {
            try {
                return Long.parseLong((String) v);
            } catch (NumberFormatException e) {
                try {
                    return Instant.parse((String) v).toEpochMilli();
                } catch (Exception e2) {
                    return 0L;
                }
            }
        }
        return 0L;
    }

    private Path dataFile(String project, String viewName) {
        if (root == null) return null;
        String p = project == null || project.isEmpty() ? "default" : project;
        return root.resolve(p).resolve(viewName).resolve("data.jsonl");
    }

    private void persistAppend(String project, String viewName, List<Map<String, Object>> rows) {
        Path f = dataFile(project, viewName);
        if (f == null) return;
        try {
            Files.createDirectories(f.getParent());
            try (BufferedWriter w = Files.newBufferedWriter(f, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                for (Map<String, Object> r : rows) {
                    w.write(toJsonLine(r));
                    w.newLine();
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("offline persist append failed", e);
        }
    }

    private void persistReplace(String project, String viewName, List<Map<String, Object>> rows) {
        Path f = dataFile(project, viewName);
        if (f == null) return;
        try {
            Files.createDirectories(f.getParent());
            try (BufferedWriter w = Files.newBufferedWriter(f, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
                for (Map<String, Object> r : rows) {
                    w.write(toJsonLine(r));
                    w.newLine();
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException("offline persist replace failed", e);
        }
    }

    private void loadAll() throws IOException {
        if (root == null || !Files.isDirectory(root)) return;
        try (var projects = Files.list(root)) {
            for (Path proj : (Iterable<Path>) projects.filter(Files::isDirectory)::iterator) {
                String project = proj.getFileName().toString();
                try (var views = Files.list(proj)) {
                    for (Path viewDir : (Iterable<Path>) views.filter(Files::isDirectory)::iterator) {
                        Path data = viewDir.resolve("data.jsonl");
                        if (!Files.isRegularFile(data)) continue;
                        String viewName = viewDir.getFileName().toString();
                        List<Map<String, Object>> rows = new ArrayList<>();
                        try (BufferedReader br = Files.newBufferedReader(data, StandardCharsets.UTF_8)) {
                            String line;
                            while ((line = br.readLine()) != null) {
                                if (line.isBlank()) continue;
                                rows.add(parseJsonLine(line));
                            }
                        }
                        tables.put(key(project, viewName), new CopyOnWriteArrayList<>(rows));
                    }
                }
            }
        }
    }

    /** Minimal JSON object line writer (string/number/bool/list-of-number). */
    public static String toJsonLine(Map<String, Object> row) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> e : row.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(esc(e.getKey())).append('"').append(':');
            sb.append(jsonValue(e.getValue()));
        }
        sb.append('}');
        return sb.toString();
    }

    private static String jsonValue(Object v) {
        if (v == null) return "null";
        if (v instanceof Number || v instanceof Boolean) return v.toString();
        if (v instanceof long[]) {
            long[] a = (long[]) v;
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (v instanceof int[]) {
            int[] a = (int[]) v;
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (v instanceof float[]) {
            float[] a = (float[]) v;
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (v instanceof double[]) {
            double[] a = (double[]) v;
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < a.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(a[i]);
            }
            return sb.append(']').toString();
        }
        if (v instanceof List) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object o : (List<?>) v) {
                if (!first) sb.append(',');
                first = false;
                sb.append(jsonValue(o));
            }
            return sb.append(']').toString();
        }
        return "\"" + esc(String.valueOf(v)) + "\"";
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    public static Map<String, Object> parseJsonLine(String line) {
        Map<String, Object> out = new LinkedHashMap<>();
        String s = line.trim();
        if (s.startsWith("{")) s = s.substring(1);
        if (s.endsWith("}")) s = s.substring(0, s.length() - 1);
        List<String> parts = splitTopLevel(s, ',');
        for (String part : parts) {
            int colon = indexOfColon(part);
            if (colon < 0) continue;
            String k = unquote(part.substring(0, colon).trim());
            String rawVal = part.substring(colon + 1).trim();
            out.put(k, parseValue(rawVal));
        }
        return out;
    }

    private static Object parseValue(String raw) {
        if (raw == null || raw.equals("null")) return null;
        if (raw.equals("true")) return Boolean.TRUE;
        if (raw.equals("false")) return Boolean.FALSE;
        if (raw.startsWith("\"") && raw.endsWith("\"")) return unquote(raw);
        if (raw.startsWith("[") && raw.endsWith("]")) {
            String inner = raw.substring(1, raw.length() - 1).trim();
            if (inner.isEmpty()) return new ArrayList<>();
            List<String> elems = splitTopLevel(inner, ',');
            List<Object> list = new ArrayList<>(elems.size());
            boolean allLong = true;
            boolean allDouble = true;
            for (String e : elems) {
                Object v = parseValue(e.trim());
                list.add(v);
                if (!(v instanceof Long) && !(v instanceof Integer)) allLong = false;
                if (!(v instanceof Number)) allDouble = false;
            }
            if (allLong) {
                long[] a = new long[list.size()];
                for (int i = 0; i < list.size(); i++) a[i] = ((Number) list.get(i)).longValue();
                return a;
            }
            if (allDouble) {
                double[] a = new double[list.size()];
                for (int i = 0; i < list.size(); i++) a[i] = ((Number) list.get(i)).doubleValue();
                return a;
            }
            return list;
        }
        try {
            if (raw.contains(".") || raw.contains("e") || raw.contains("E")) {
                return Double.parseDouble(raw);
            }
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            return raw;
        }
    }

    private static List<String> splitTopLevel(String s, char sep) {
        List<String> parts = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQ = false;
        int depth = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQ = !inQ;
            if (!inQ) {
                if (c == '[') depth++;
                if (c == ']') depth--;
                if (c == sep && depth == 0) {
                    parts.add(cur.toString());
                    cur.setLength(0);
                    continue;
                }
            }
            cur.append(c);
        }
        if (cur.length() > 0) parts.add(cur.toString());
        return parts;
    }

    private static int indexOfColon(String s) {
        boolean inQ = false;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQ = !inQ;
            if (c == ':' && !inQ) return i;
        }
        return -1;
    }

    private static String unquote(String s) {
        s = s.trim();
        if (s.startsWith("\"") && s.endsWith("\"") && s.length() >= 2) {
            s = s.substring(1, s.length() - 1);
        }
        return s.replace("\\\"", "\"").replace("\\\\", "\\");
    }
}
