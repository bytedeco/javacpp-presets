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
package org.bytedeco.pytorch.utils.hudi;

import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Lightweight Apache Hudi timeline parser over {@code .hoodie/} metadata.
 *
 * <p>Supports commit / deltacommit / replacecommit instant files without
 * Hadoop or hudi-client. Commit metadata is best-effort JSON/properties parse
 * for file lists; when metadata lacks file paths, data files are discovered
 * by walking the table path for {@code *.parquet} (COW base).</p>
 *
 * @see <a href="https://hudi.apache.org/docs/timeline">Hudi Timeline</a>
 */
public final class HudiTimeline {

    public enum Action {
        COMMIT,
        DELTACOMMIT,
        REPLACECOMMIT,
        CLEAN,
        COMPACTION,
        ROLLBACK,
        SAVEPOINT,
        RESTORE,
        UNKNOWN
    }

    public enum State {
        REQUESTED,
        INFLIGHT,
        COMPLETED,
        UNKNOWN
    }

    /**
     * One timeline instant.
     *
     * @param instantTime Hudi instant time string (yyyyMMddHHmmssSSS or similar)
     * @param action      commit action
     * @param state       completion state
     * @param metaPath    path to the timeline file if present
     */
    public record Instant(String instantTime, Action action, State state, Path metaPath)
            implements Comparable<Instant> {
        @Override
        public int compareTo(Instant o) {
            return this.instantTime.compareTo(o.instantTime);
        }
    }

    /**
     * File slice reference from a completed commit (path relative or absolute).
     */
    public record FileSlice(String partitionPath, String fileId, String path, long recordCount) {}

    private static final Pattern INSTANT_FILE = Pattern.compile(
            "^(\\d{14,17})(?:\\.(\\w+))?(?:\\.(\\w+))?$");

    private static final DateTimeFormatter INSTANT_FMT =
            DateTimeFormatter.ofPattern("yyyyMMddHHmmss");

    private final Path tablePath;
    private final Path hoodiePath;
    private final List<Instant> instants;

    private HudiTimeline(Path tablePath, Path hoodiePath, List<Instant> instants) {
        this.tablePath = tablePath;
        this.hoodiePath = hoodiePath;
        this.instants = List.copyOf(instants);
    }

    public static HudiTimeline load(Path tablePath) {
        Objects.requireNonNull(tablePath, "tablePath");
        Path root = tablePath.toAbsolutePath().normalize();
        Path hoodie = root.resolve(".hoodie");
        List<Instant> list = new ArrayList<>();
        if (Files.isDirectory(hoodie)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(hoodie)) {
                for (Path p : stream) {
                    if (!Files.isRegularFile(p)) continue;
                    Instant inst = parseInstantFile(p);
                    if (inst != null) list.add(inst);
                }
            } catch (IOException e) {
                throw new LakeException(LakeFormat.HUDI, "timeline.load",
                        "failed to list .hoodie under " + hoodie, e);
            }
        }
        list.sort(Comparator.naturalOrder());
        return new HudiTimeline(root, hoodie, list);
    }

    static Instant parseInstantFile(Path file) {
        String name = file.getFileName().toString();
        // Skip non-instant metadata: hoodie.properties, archived, schema, etc.
        if (name.startsWith("hoodie.") || name.equals("archived") || name.endsWith(".properties")) {
            return null;
        }
        if (name.contains("archived") || Files.isDirectory(file)) return null;

        Matcher m = INSTANT_FILE.matcher(name);
        if (!m.matches()) {
            // also accept: <ts>.commit, <ts>.deltacommit.inflight, <ts>.commit.requested
            int dot = name.indexOf('.');
            if (dot < 14) return null;
            String ts = name.substring(0, dot);
            if (!ts.chars().allMatch(Character::isDigit) || ts.length() < 14) return null;
            String rest = name.substring(dot + 1).toLowerCase();
            Action action = actionFrom(rest);
            State state = stateFrom(rest);
            return new Instant(ts, action, state, file);
        }
        String ts = m.group(1);
        String g2 = m.group(2);
        String g3 = m.group(3);
        String rest = ((g2 == null ? "" : g2) + (g3 == null ? "" : "." + g3)).toLowerCase();
        return new Instant(ts, actionFrom(rest), stateFrom(rest), file);
    }

    private static Action actionFrom(String rest) {
        if (rest == null || rest.isBlank()) return Action.UNKNOWN;
        if (rest.contains("deltacommit")) return Action.DELTACOMMIT;
        if (rest.contains("replacecommit")) return Action.REPLACECOMMIT;
        if (rest.contains("commit")) return Action.COMMIT;
        if (rest.contains("clean")) return Action.CLEAN;
        if (rest.contains("compaction")) return Action.COMPACTION;
        if (rest.contains("rollback")) return Action.ROLLBACK;
        if (rest.contains("savepoint")) return Action.SAVEPOINT;
        if (rest.contains("restore")) return Action.RESTORE;
        return Action.UNKNOWN;
    }

    private static State stateFrom(String rest) {
        if (rest == null) return State.UNKNOWN;
        if (rest.endsWith("requested") || rest.contains(".requested")) return State.REQUESTED;
        if (rest.endsWith("inflight") || rest.contains(".inflight")) return State.INFLIGHT;
        // bare .commit / .deltacommit means completed
        if (rest.contains("commit") || rest.contains("clean") || rest.contains("compaction")
                || rest.contains("savepoint") || rest.contains("restore") || rest.contains("rollback")) {
            if (!rest.contains("inflight") && !rest.contains("requested")) return State.COMPLETED;
        }
        return State.UNKNOWN;
    }

    public Path tablePath() { return tablePath; }
    public Path hoodiePath() { return hoodiePath; }
    public List<Instant> instants() { return instants; }

    public boolean hasTimeline() {
        return Files.isDirectory(hoodiePath);
    }

    public List<Instant> completedCommits() {
        return instants.stream()
                .filter(i -> i.state() == State.COMPLETED)
                .filter(i -> i.action() == Action.COMMIT
                        || i.action() == Action.DELTACOMMIT
                        || i.action() == Action.REPLACECOMMIT)
                .sorted()
                .collect(Collectors.toList());
    }

    public Instant latestCompleted() {
        List<Instant> c = completedCommits();
        return c.isEmpty() ? null : c.get(c.size() - 1);
    }

    /**
     * Resolve as-of instant: explicit instantTime, or last completed with
     * timestamp &lt;= asOfTimeMs, or latest completed.
     */
    public Instant resolveInstant(String instantTime, Long asOfTimeMs) {
        List<Instant> completed = completedCommits();
        if (completed.isEmpty()) return null;
        if (instantTime != null && !instantTime.isBlank()) {
            for (Instant i : completed) {
                if (i.instantTime().equals(instantTime)) return i;
            }
            // prefix match (14 vs 17 digit)
            for (int k = completed.size() - 1; k >= 0; k--) {
                Instant i = completed.get(k);
                if (i.instantTime().startsWith(instantTime) || instantTime.startsWith(i.instantTime())) {
                    return i;
                }
            }
            throw new LakeException(LakeFormat.HUDI, "timeline",
                    "instant not found: " + instantTime);
        }
        if (asOfTimeMs != null) {
            String bound = formatInstant(asOfTimeMs);
            Instant best = null;
            for (Instant i : completed) {
                if (i.instantTime().compareTo(bound) <= 0) best = i;
            }
            return best;
        }
        return completed.get(completed.size() - 1);
    }

    /**
     * Instants strictly after {@code fromExclusive} up to and including latest (or toInstant).
     */
    public List<Instant> instantsAfter(String fromExclusive, String toInclusive) {
        List<Instant> completed = completedCommits();
        List<Instant> out = new ArrayList<>();
        for (Instant i : completed) {
            if (fromExclusive != null && i.instantTime().compareTo(fromExclusive) <= 0) continue;
            if (toInclusive != null && i.instantTime().compareTo(toInclusive) > 0) continue;
            out.add(i);
        }
        return out;
    }

    public static String formatInstant(long epochMs) {
        // Fully-qualified: nested record Instant shadows java.time.Instant
        LocalDateTime ldt = LocalDateTime.ofInstant(
                java.time.Instant.ofEpochMilli(epochMs), ZoneOffset.UTC);
        return ldt.format(INSTANT_FMT) + String.format("%03d", Math.floorMod(epochMs, 1000));
    }

    public static String nowInstant() {
        return formatInstant(System.currentTimeMillis());
    }

    /**
     * Read hoodie.properties if present.
     */
    public Map<String, String> tableProperties() {
        Path props = hoodiePath.resolve("hoodie.properties");
        if (!Files.isRegularFile(props)) return Map.of();
        Map<String, String> m = new LinkedHashMap<>();
        try {
            for (String line : Files.readAllLines(props, StandardCharsets.UTF_8)) {
                String t = line.trim();
                if (t.isEmpty() || t.startsWith("#")) continue;
                int eq = t.indexOf('=');
                if (eq <= 0) continue;
                m.put(t.substring(0, eq).trim(), t.substring(eq + 1).trim());
            }
        } catch (IOException e) {
            throw new LakeException(LakeFormat.HUDI, "timeline.properties",
                    "failed to read " + props, e);
        }
        return Collections.unmodifiableMap(m);
    }

    /**
     * Best-effort extract of write file paths from a completed commit metadata file.
     * Falls back to empty list (caller should walk parquet files).
     */
    public List<String> dataFilesFromCommit(Instant instant) {
        if (instant == null || instant.metaPath() == null) return List.of();
        if (!Files.isRegularFile(instant.metaPath())) return List.of();
        try {
            String raw = Files.readString(instant.metaPath(), StandardCharsets.UTF_8);
            return extractPathsFromMetadata(raw);
        } catch (IOException e) {
            return List.of();
        }
    }

    /**
     * Extract path-like strings from Hudi commit metadata JSON / avro-json dumps.
     */
    static List<String> extractPathsFromMetadata(String raw) {
        if (raw == null || raw.isBlank()) return List.of();
        List<String> paths = new ArrayList<>();
        // common keys: "path":"...", "filePath":"...", "dataFilePath"
        Pattern p = Pattern.compile(
                "\"(?:path|filePath|dataFilePath|baseFilePath)\"\\s*:\\s*\"([^\"]+)\"");
        Matcher m = p.matcher(raw);
        while (m.find()) {
            String path = m.group(1);
            if (path.endsWith(".parquet") || path.endsWith(".parq")) {
                paths.add(path);
            }
        }
        // also bare parquet paths in properties-style commits
        Pattern bare = Pattern.compile("([\\w./\\-]+\\.parquet)");
        Matcher bm = bare.matcher(raw);
        while (bm.find()) {
            String path = bm.group(1);
            if (!paths.contains(path)) paths.add(path);
        }
        return paths;
    }

    /**
     * Discover all base Parquet files under the table (COW / base-only MOR).
     * Skips {@code .hoodie} and hidden dirs.
     */
    public List<Path> discoverParquetFiles() {
        List<Path> out = new ArrayList<>();
        if (!Files.isDirectory(tablePath)) return out;
        try (Stream<Path> walk = Files.walk(tablePath)) {
            walk.filter(Files::isRegularFile)
                    .filter(p -> {
                        String n = p.getFileName().toString().toLowerCase();
                        return n.endsWith(".parquet") || n.endsWith(".parq");
                    })
                    .filter(p -> !p.toString().contains("/.hoodie/") && !p.toString().contains("\\.hoodie\\"))
                    .sorted()
                    .forEach(out::add);
        } catch (IOException e) {
            throw new LakeException(LakeFormat.HUDI, "timeline.discover",
                    "failed to walk parquet under " + tablePath, e);
        }
        return out;
    }

    /**
     * Partition-relative path for a data file (Hive-style {@code k=v/} segments), or empty.
     */
    public static String partitionPathOf(Path tableRoot, Path file) {
        Path parent = file.getParent();
        if (parent == null) return "";
        Path rel;
        try {
            rel = tableRoot.toAbsolutePath().normalize().relativize(parent.toAbsolutePath().normalize());
        } catch (Exception e) {
            return "";
        }
        String s = rel.toString().replace('\\', '/');
        if (s.equals(".") || s.isEmpty()) return "";
        // drop leading data/ if present
        if (s.startsWith("data/")) s = s.substring(5);
        return s;
    }

    /**
     * Whether a partition path matches the filter (EQ/IN on hive-style keys).
     */
    public static boolean partitionMatches(String partitionPath, org.bytedeco.pytorch.utils.lake.PartitionFilter filter) {
        if (filter == null || filter.isEmpty()) return true;
        Map<String, String> parts = parseHivePartition(partitionPath);
        for (var pred : filter.predicates()) {
            String v = parts.get(pred.column());
            if (v == null) {
                // also allow matching full path contains
                if (pred.op() == org.bytedeco.pytorch.utils.lake.PartitionFilter.Op.EQ
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
            if (eq > 0) {
                m.put(seg.substring(0, eq), seg.substring(eq + 1));
            }
        }
        return m;
    }

    /**
     * Ensure {@code .hoodie} exists and write minimal hoodie.properties for a new light table.
     */
    public static void initTable(Path tablePath, Map<String, String> extraProps) throws IOException {
        Path hoodie = tablePath.resolve(".hoodie");
        Files.createDirectories(hoodie);
        Path props = hoodie.resolve("hoodie.properties");
        if (!Files.exists(props)) {
            StringBuilder sb = new StringBuilder();
            sb.append("# Generated by jnitorch Hudi light adapter\n");
            sb.append("hoodie.table.name=").append(tablePath.getFileName()).append('\n');
            sb.append("hoodie.table.type=COPY_ON_WRITE\n");
            sb.append("hoodie.archivelog.folder=archived\n");
            sb.append("hoodie.timeline.layout.version=1\n");
            if (extraProps != null) {
                for (var e : extraProps.entrySet()) {
                    sb.append(e.getKey()).append('=').append(e.getValue()).append('\n');
                }
            }
            Files.writeString(props, sb.toString(), StandardCharsets.UTF_8);
        }
    }

    /**
     * Write a completed commit marker + simple metadata listing written files.
     */
    public static Instant writeCommit(Path tablePath, String instantTime, List<Path> dataFiles,
                                      long totalRecords) throws IOException {
        Path hoodie = tablePath.resolve(".hoodie");
        Files.createDirectories(hoodie);
        Path meta = hoodie.resolve(instantTime + ".commit");
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"partitionToWriteStats\": {},\n");
        sb.append("  \"compacted\": false,\n");
        sb.append("  \"extraMetadata\": {\"jnitorch.records\":\"").append(totalRecords).append("\"},\n");
        sb.append("  \"fileIds\": [");
        for (int i = 0; i < dataFiles.size(); i++) {
            Path f = dataFiles.get(i);
            String loc = f.toAbsolutePath().normalize().toString().replace('\\', '/');
            if (i > 0) sb.append(',');
            sb.append("\n    {\"path\":\"").append(escapeJson(loc)).append("\"}");
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        Files.writeString(meta, sb.toString(), StandardCharsets.UTF_8);
        return new Instant(instantTime, Action.COMMIT, State.COMPLETED, meta);
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
