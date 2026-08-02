/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.utils.paimon;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeSchema;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Lightweight Apache Paimon metadata: schema + snapshot markers + parquet discovery.
 *
 * <p>Does not depend on paimon-core. Snapshot JSON is best-effort parsed for id/time
 * and file hints; data files are discovered under the table path.</p>
 *
 * @see <a href="https://paimon.apache.org/docs/master/concepts/spec/">Paimon Spec</a>
 */
public final class PaimonSnapshot {

    public record Snapshot(long id, long timeMillis, Path metaPath, List<String> dataFileHints) {
        public Snapshot {
            dataFileHints = dataFileHints == null ? List.of() : List.copyOf(dataFileHints);
        }
    }

    private static final Pattern SNAPSHOT_FILE = Pattern.compile("^snapshot-(\\d+)$");
    private static final Pattern JSON_LONG = Pattern.compile("\"(id|snapshotId|timeMillis|timestamp)\"\\s*:\\s*(\\d+)");
    private static final Pattern JSON_PATH = Pattern.compile(
            "\"(?:fileName|filePath|path|dataFilePath)\"\\s*:\\s*\"([^\"]+)\"");

    private final Path tablePath;
    private final List<Snapshot> snapshots;
    private final LakeSchema schema;

    private PaimonSnapshot(Path tablePath, List<Snapshot> snapshots, LakeSchema schema) {
        this.tablePath = tablePath;
        this.snapshots = List.copyOf(snapshots);
        this.schema = schema;
    }

    public static PaimonSnapshot load(Path tablePath) {
        Objects.requireNonNull(tablePath, "tablePath");
        Path root = tablePath.toAbsolutePath().normalize();
        LakeSchema schema = loadSchema(root);
        List<Snapshot> snaps = loadSnapshots(root);
        return new PaimonSnapshot(root, snaps, schema);
    }

    public Path tablePath() { return tablePath; }
    public List<Snapshot> snapshots() { return snapshots; }
    public LakeSchema schema() { return schema; }

    public Snapshot latest() {
        return snapshots.isEmpty() ? null : snapshots.get(snapshots.size() - 1);
    }

    public Snapshot earliest() {
        return snapshots.isEmpty() ? null : snapshots.get(0);
    }

    public Snapshot resolve(Long snapshotId, Long asOfTimeMs) {
        if (snapshots.isEmpty()) return null;
        if (snapshotId != null) {
            for (Snapshot s : snapshots) {
                if (s.id() == snapshotId) return s;
            }
            throw new LakeException(LakeFormat.PAIMON, "snapshot",
                    "snapshot id not found: " + snapshotId);
        }
        if (asOfTimeMs != null) {
            Snapshot best = null;
            for (Snapshot s : snapshots) {
                if (s.timeMillis() <= asOfTimeMs) best = s;
            }
            return best;
        }
        return latest();
    }

    public List<Snapshot> after(long fromExclusiveId) {
        List<Snapshot> out = new ArrayList<>();
        for (Snapshot s : snapshots) {
            if (s.id() > fromExclusiveId) out.add(s);
        }
        return out;
    }

    public static LakeSchema loadSchema(Path tablePath) {
        Path schemaDir = tablePath.resolve("schema");
        Path best = null;
        long bestId = -1;
        if (Files.isDirectory(schemaDir)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(schemaDir)) {
                for (Path p : stream) {
                    String n = p.getFileName().toString();
                    if (n.startsWith("schema-")) {
                        try {
                            long id = Long.parseLong(n.substring("schema-".length()));
                            if (id >= bestId) {
                                bestId = id;
                                best = p;
                            }
                        } catch (NumberFormatException ignored) {
                        }
                    }
                }
            } catch (IOException ignored) {
            }
        }
        if (best != null && Files.isRegularFile(best)) {
            try {
                return parseSchemaJson(Files.readString(best, StandardCharsets.UTF_8));
            } catch (Exception ignored) {
            }
        }
        // fallback: infer from first parquet
        List<Path> files = discoverParquetFiles(tablePath);
        if (!files.isEmpty()) {
            try {
                var df = org.bytedeco.pytorch.dataframe.DataFrame.readParquet(files.get(0).toString());
                LakeSchema.Builder b = LakeSchema.builder();
                for (Column c : df.columns()) {
                    b.add(c.name(), c.dtype());
                }
                return b.build();
            } catch (Exception ignored) {
            }
        }
        return LakeSchema.builder().add("value", Column.DType.STRING).build();
    }

    static LakeSchema parseSchemaJson(String json) {
        LakeSchema.Builder b = LakeSchema.builder();
        // fields array: "name":"x" ... "type":"..."
        Pattern field = Pattern.compile(
                "\\{\\s*\"id\"\\s*:\\s*\\d+\\s*,\\s*\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"type\"\\s*:\\s*\"([^\"]+)\"");
        Matcher m = field.matcher(json);
        boolean any = false;
        while (m.find()) {
            b.add(m.group(1), mapType(m.group(2)));
            any = true;
        }
        if (!any) {
            // simpler: "name":"col"
            Pattern names = Pattern.compile("\"name\"\\s*:\\s*\"([^\"]+)\"");
            Matcher nm = names.matcher(json);
            while (nm.find()) {
                String name = nm.group(1);
                if ("type".equals(name) || "fields".equals(name)) continue;
                b.add(name, Column.DType.STRING);
                any = true;
            }
        }
        if (!any) b.add("value", Column.DType.STRING);
        return b.build();
    }

    static Column.DType mapType(String paimonType) {
        if (paimonType == null) return Column.DType.STRING;
        String t = paimonType.toLowerCase();
        if (t.contains("int64") || t.equals("bigint") || t.equals("long")) return Column.DType.INT64;
        if (t.contains("int32") || t.equals("int") || t.equals("integer")) return Column.DType.INT32;
        if (t.contains("float64") || t.equals("double")) return Column.DType.FLOAT64;
        if (t.contains("float32") || t.equals("float")) return Column.DType.FLOAT32;
        if (t.contains("bool")) return Column.DType.BOOLEAN;
        if (t.contains("timestamp") || t.contains("date")) return Column.DType.STRING;
        if (t.contains("binary") || t.contains("bytes")) return Column.DType.BINARY;
        return Column.DType.STRING;
    }

    static List<Snapshot> loadSnapshots(Path tablePath) {
        Path snapDir = tablePath.resolve("snapshot");
        List<Snapshot> list = new ArrayList<>();
        if (!Files.isDirectory(snapDir)) return list;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(snapDir)) {
            for (Path p : stream) {
                if (!Files.isRegularFile(p)) continue;
                String n = p.getFileName().toString();
                if ("EARLIEST".equals(n) || "LATEST".equals(n)) continue;
                Matcher m = SNAPSHOT_FILE.matcher(n);
                if (!m.matches()) continue;
                long id = Long.parseLong(m.group(1));
                list.add(parseSnapshotFile(p, id));
            }
        } catch (IOException e) {
            throw new LakeException(LakeFormat.PAIMON, "snapshot.load",
                    "failed to list snapshots under " + snapDir, e);
        }
        list.sort(Comparator.comparingLong(Snapshot::id));
        return list;
    }

    static Snapshot parseSnapshotFile(Path file, long fallbackId) {
        long id = fallbackId;
        long time = Files.isRegularFile(file) ? file.toFile().lastModified() : 0L;
        List<String> hints = new ArrayList<>();
        try {
            String raw = Files.readString(file, StandardCharsets.UTF_8);
            Matcher lm = JSON_LONG.matcher(raw);
            while (lm.find()) {
                String key = lm.group(1);
                long val = Long.parseLong(lm.group(2));
                if ("id".equals(key) || "snapshotId".equals(key)) id = val;
                if ("timeMillis".equals(key) || "timestamp".equals(key)) time = val;
            }
            Matcher pm = JSON_PATH.matcher(raw);
            while (pm.find()) {
                String path = pm.group(1);
                if (path.endsWith(".parquet") || path.endsWith(".parq")) hints.add(path);
            }
            // bare parquet
            Matcher bare = Pattern.compile("([\\w./\\-]+\\.parquet)").matcher(raw);
            while (bare.find()) {
                String path = bare.group(1);
                if (!hints.contains(path)) hints.add(path);
            }
        } catch (IOException ignored) {
        }
        return new Snapshot(id, time, file, hints);
    }

    public static List<Path> discoverParquetFiles(Path tablePath) {
        List<Path> out = new ArrayList<>();
        if (!Files.isDirectory(tablePath)) return out;
        try (Stream<Path> walk = Files.walk(tablePath)) {
            walk.filter(Files::isRegularFile)
                    .filter(p -> {
                        String n = p.getFileName().toString().toLowerCase();
                        return n.endsWith(".parquet") || n.endsWith(".parq");
                    })
                    .filter(p -> {
                        String s = p.toString().replace('\\', '/');
                        return !s.contains("/schema/") && !s.contains("/snapshot/")
                                && !s.contains("/manifest/") && !s.contains("/changelog/");
                    })
                    .sorted()
                    .forEach(out::add);
        } catch (IOException e) {
            throw new LakeException(LakeFormat.PAIMON, "discover",
                    "failed to walk " + tablePath, e);
        }
        return out;
    }

    public static String partitionPathOf(Path tableRoot, Path file) {
        Path parent = file.getParent();
        if (parent == null) return "";
        try {
            Path rel = tableRoot.toAbsolutePath().normalize()
                    .relativize(parent.toAbsolutePath().normalize());
            String s = rel.toString().replace('\\', '/');
            if (s.equals(".") || s.isEmpty()) return "";
            // strip bucket-N prefix noise for matching but keep hive keys
            return s;
        } catch (Exception e) {
            return "";
        }
    }

    public static boolean partitionMatches(String partitionPath, PartitionFilter filter) {
        if (filter == null || filter.isEmpty()) return true;
        Map<String, String> parts = parseHivePartition(partitionPath);
        for (var pred : filter.predicates()) {
            String v = parts.get(pred.column());
            if (v == null) {
                if (pred.op() == PartitionFilter.Op.EQ
                        && partitionPath != null
                        && partitionPath.contains(pred.column() + "=" + pred.values().get(0))) {
                    continue;
                }
                return false;
            }
            switch (pred.op()) {
                case EQ -> {
                    if (pred.values().isEmpty() || !v.equals(pred.values().get(0))) return false;
                }
                case IN -> {
                    if (!pred.values().contains(v)) return false;
                }
                case GT -> {
                    if (pred.values().isEmpty() || v.compareTo(pred.values().get(0)) <= 0) return false;
                }
                case GTE -> {
                    if (pred.values().isEmpty() || v.compareTo(pred.values().get(0)) < 0) return false;
                }
                case LT -> {
                    if (pred.values().isEmpty() || v.compareTo(pred.values().get(0)) >= 0) return false;
                }
                case LTE -> {
                    if (pred.values().isEmpty() || v.compareTo(pred.values().get(0)) > 0) return false;
                }
            }
        }
        return true;
    }

    public static Map<String, String> parseHivePartition(String partitionPath) {
        Map<String, String> m = new LinkedHashMap<>();
        if (partitionPath == null || partitionPath.isBlank()) return m;
        for (String seg : partitionPath.replace('\\', '/').split("/")) {
            int eq = seg.indexOf('=');
            if (eq > 0) m.put(seg.substring(0, eq), seg.substring(eq + 1));
        }
        return m;
    }

    public static void initTable(Path tablePath, LakeSchema schema) throws IOException {
        Files.createDirectories(tablePath.resolve("schema"));
        Files.createDirectories(tablePath.resolve("snapshot"));
        Files.createDirectories(tablePath.resolve("manifest"));
        Path schema0 = tablePath.resolve("schema").resolve("schema-0");
        if (!Files.exists(schema0) && schema != null) {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"version\":1,\"id\":0,\"fields\":[");
            for (int i = 0; i < schema.fields().size(); i++) {
                var f = schema.fields().get(i);
                if (i > 0) sb.append(',');
                sb.append("{\"id\":").append(i)
                        .append(",\"name\":\"").append(f.name())
                        .append("\",\"type\":\"").append(f.dtype()).append("\"}");
            }
            sb.append("],\"highestFieldId\":").append(Math.max(0, schema.size() - 1)).append('}');
            Files.writeString(schema0, sb.toString(), StandardCharsets.UTF_8);
        }
        Path earliest = tablePath.resolve("snapshot").resolve("EARLIEST");
        Path latest = tablePath.resolve("snapshot").resolve("LATEST");
        if (!Files.exists(earliest)) Files.writeString(earliest, "1", StandardCharsets.UTF_8);
        if (!Files.exists(latest)) Files.writeString(latest, "0", StandardCharsets.UTF_8);
    }

    public static Snapshot writeSnapshot(Path tablePath, long id, List<Path> dataFiles, long totalRecords)
            throws IOException {
        Path snapDir = tablePath.resolve("snapshot");
        Files.createDirectories(snapDir);
        long now = System.currentTimeMillis();
        Path meta = snapDir.resolve("snapshot-" + id);
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"version\": 3,\n");
        sb.append("  \"id\": ").append(id).append(",\n");
        sb.append("  \"schemaId\": 0,\n");
        sb.append("  \"timeMillis\": ").append(now).append(",\n");
        sb.append("  \"totalRecordCount\": ").append(totalRecords).append(",\n");
        sb.append("  \"filePaths\": [");
        for (int i = 0; i < dataFiles.size(); i++) {
            if (i > 0) sb.append(',');
            String loc = dataFiles.get(i).toAbsolutePath().normalize().toString().replace('\\', '/');
            sb.append("\n    \"").append(loc.replace("\\", "\\\\").replace("\"", "\\\"")).append('"');
        }
        sb.append("\n  ]\n}\n");
        Files.writeString(meta, sb.toString(), StandardCharsets.UTF_8);
        Files.writeString(snapDir.resolve("LATEST"), String.valueOf(id), StandardCharsets.UTF_8);
        Path earliest = snapDir.resolve("EARLIEST");
        if (!Files.exists(earliest)) {
            Files.writeString(earliest, String.valueOf(id), StandardCharsets.UTF_8);
        }
        List<String> hints = new ArrayList<>();
        for (Path p : dataFiles) hints.add(p.toString());
        return new Snapshot(id, now, meta, hints);
    }

    public static long nextSnapshotId(Path tablePath) {
        Path latest = tablePath.resolve("snapshot").resolve("LATEST");
        if (Files.isRegularFile(latest)) {
            try {
                String s = Files.readString(latest, StandardCharsets.UTF_8).trim();
                return Long.parseLong(s) + 1;
            } catch (Exception ignored) {
            }
        }
        List<Snapshot> snaps = loadSnapshots(tablePath);
        return snaps.isEmpty() ? 1L : snaps.get(snaps.size() - 1).id() + 1;
    }
}
