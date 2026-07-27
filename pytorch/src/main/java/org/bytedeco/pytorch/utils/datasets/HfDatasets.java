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

import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.hub.HfToken;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

/**
 * HuggingFace {@code datasets.load_dataset}-style entry points for this framework.
 *
 * <p>Downloads dataset repos from the Hub (using {@link HfHub} + resolved
 * {@link HfToken}), selects config/split from common on-disk layouts
 * ({@code {config}/{split}-*.parquet}, {@code data/{split}.csv}, …), and
 * materialises rows into {@link HfDataset} / {@link HfDataset.DatasetDict}.
 *
 * <pre>{@code
 * // full DatasetDict (all splits of default / chosen config)
 * HfDataset.DatasetDict glue = HfDatasets.loadDataset("glue", "cola");
 *
 * // single split
 * HfDataset train = HfDatasets.loadDataset("imdb", null, "train");
 *
 * // local files
 * HfDataset local = HfDatasets.loadFromDisk(Path.of("./my_ds"));
 * HfDataset csv   = HfDatasets.load("csv", Map.of("data_files", "train.csv"));
 * }</pre>
 *
 * <p>Token resolution order matches {@link HfToken}: explicit builder arg →
 * {@code HF_TOKEN} / {@code HUGGING_FACE_HUB_TOKEN} → token files. Endpoint
 * honours {@code HF_ENDPOINT} / {@code HF_MIRROR} (e.g. {@code https://hf-mirror.com}).
 */
public final class HfDatasets {

    private HfDatasets() {}

    // Common split name tokens appearing in filenames.
    private static final Set<String> KNOWN_SPLITS = Set.of(
            "train", "test", "validation", "val", "dev", "eval",
            "unsupervised", "preview", "sample",
            "train_sft", "test_sft", "train_gen", "test_gen",
            "test_matched", "test_mismatched",
            "validation_matched", "validation_mismatched",
            "auxiliary_train"
    );

    // ---- public load API ---------------------------------------------------

    /**
     * Load a Hub dataset into a {@link HfDataset.DatasetDict} (all discovered splits
     * of the default or only config).
     */
    public static HfDataset.DatasetDict loadDataset(String path) throws IOException {
        return loadDataset(path, null, null, LoadConfig.defaults());
    }

    /** Load a specific config (subset) of a Hub dataset (all splits). */
    public static HfDataset.DatasetDict loadDataset(String path, String name) throws IOException {
        return loadDataset(path, name, null, LoadConfig.defaults());
    }

    /**
     * Load one split of a Hub dataset. When {@code split} is null, returns all
     * splits as a DatasetDict via {@link #loadDataset(String, String)}; this
     * overload always returns a single {@link HfDataset}.
     */
    public static HfDataset loadDataset(String path, String name, String split) throws IOException {
        Objects.requireNonNull(split, "split");
        HfDataset.DatasetDict dict = loadDataset(path, name, split, LoadConfig.defaults());
        // Prefer exact split; otherwise first available.
        if (dict.splits().containsKey(split)) return dict.get(split);
        if (dict.splits().isEmpty()) return HfDataset.empty();
        return dict.splits().values().iterator().next();
    }

    public static HfDataset.DatasetDict loadDataset(String path, String name, String split,
                                                     LoadConfig cfg) throws IOException {
        Objects.requireNonNull(path, "path");
        LoadConfig c = cfg == null ? LoadConfig.defaults() : cfg;

        // Local path short-circuit: directory or file on disk.
        Path local = Path.of(path);
        if (Files.exists(local)) {
            return loadLocal(local, name, split, c);
        }

        HfHub hub = c.buildHub();
        c.log("[HfDatasets] load_dataset path=" + path
                + " name=" + name + " split=" + split
                + " endpoint=" + hub.endpoint()
                + " token=" + HfToken.mask(hub.token()));

        // 1) List remote tree (or use cache)
        List<HfHub.RepoFile> tree;
        try {
            tree = hub.listDatasetFiles(path, c.revision);
        } catch (IOException e) {
            // Offline / network failure: try cache snapshot
            c.log("[HfDatasets] tree failed: " + e.getMessage() + " — trying cache");
            Path snap = tryCachedSnapshot(hub, path, c.revision);
            if (snap == null) throw e;
            return loadFromSnapshot(snap, path, name, split, c);
        }

        List<String> dataFiles = new ArrayList<>();
        for (HfHub.RepoFile rf : tree) {
            if (rf.isFile() && HfHub.isDatasetDataFile(rf.path)) {
                if (c.allowPatterns != null && !c.allowPatterns.isEmpty()) {
                    boolean ok = false;
                    for (String pat : c.allowPatterns) {
                        if (HfHub.matchPattern(rf.path, pat)) { ok = true; break; }
                    }
                    if (!ok) continue;
                }
                dataFiles.add(rf.path);
            }
        }
        if (dataFiles.isEmpty()) {
            throw new IOException("No data files found in dataset repo: " + path
                    + " (tree size=" + tree.size()
                    + (c.allowPatterns != null ? ", allowPatterns=" + c.allowPatterns : "")
                    + ")");
        }

        // 2) Infer configs + splits from paths
        DatasetLayout layout = DatasetLayout.infer(dataFiles);
        c.log("[HfDatasets] layout configs=" + layout.configs()
                + " splits=" + layout.splits()
                + " files=" + dataFiles.size());

        String config = resolveConfig(name, layout, c);
        List<String> wantedSplits = resolveSplits(split, layout, config, c);

        // 3) Select files to download
        List<String> toDownload = layout.filesFor(config, wantedSplits);
        // Prefer columnar / structured shards over giant plain-text dumps when capping
        // (TinyStories ships both data/train-*.parquet AND TinyStories-train.txt ~1.8GB).
        toDownload = preferCompactDataFiles(toDownload);
        if (c.maxFiles > 0 && toDownload.size() > c.maxFiles) {
            c.log("[HfDatasets] truncating file list " + toDownload.size() + " -> " + c.maxFiles
                    + " (first=" + toDownload.get(0) + ")");
            toDownload = new ArrayList<>(toDownload.subList(0, c.maxFiles));
        }
        if (toDownload.isEmpty()) {
            throw new IOException("No files for config=" + config + " splits=" + wantedSplits
                    + " in " + path + ". Available configs=" + layout.configs()
                    + " splits=" + layout.splits());
        }

        // 4) Download
        Path snap = hub.downloadRepoFiles(path, c.revision, "datasets", toDownload, null);
        c.log("[HfDatasets] snapshot=" + snap);

        // 5) Materialise
        return materialise(snap, toDownload, layout, config, wantedSplits, c);
    }

    /** Python-style alias. */
    public static HfDataset.DatasetDict load_dataset(String path) throws IOException {
        return loadDataset(path);
    }

    public static HfDataset.DatasetDict load_dataset(String path, String name) throws IOException {
        return loadDataset(path, name);
    }

    public static HfDataset load_dataset(String path, String name, String split) throws IOException {
        return loadDataset(path, name, split);
    }

    /**
     * Builder-style load: {@code HfDatasets.load("parquet", Map.of("data_files", "..."))}
     * or {@code load("json", ...)} / {@code load("csv", ...)} / {@code load("text", ...)}.
     */
    public static HfDataset load(String format, Map<String, Object> kwargs) throws IOException {
        Objects.requireNonNull(format, "format");
        Map<String, Object> kw = kwargs == null ? Map.of() : kwargs;
        Object df = kw.get("data_files");
        if (df == null) df = kw.get("path");
        if (df == null) throw new IllegalArgumentException("data_files or path required");

        List<Path> files = new ArrayList<>();
        if (df instanceof Path p) files.add(p);
        else if (df instanceof String s) {
            if (s.contains(",") && !Files.exists(Path.of(s))) {
                for (String part : s.split(",")) files.add(Path.of(part.trim()));
            } else {
                files.add(Path.of(s));
            }
        } else if (df instanceof List<?> list) {
            for (Object o : list) files.add(Path.of(String.valueOf(o)));
        } else {
            files.add(Path.of(String.valueOf(df)));
        }

        String fmt = format.toLowerCase(Locale.ROOT);
        List<HfDataset> parts = new ArrayList<>();
        for (Path f : files) {
            if (Files.isDirectory(f)) {
                parts.add(HfDataset.fromDirectory(f, true));
                continue;
            }
            switch (fmt) {
                case "parquet", "pq" -> parts.add(HfDataset.fromParquet(f));
                case "arrow", "feather", "ipc" -> parts.add(HfDataset.fromArrow(f));
                case "orc" -> parts.add(HfDataset.fromOrc(f));
                case "avro" -> parts.add(HfDataset.fromAvro(f));
                case "csv" -> parts.add(HfDataset.fromCsv(f, true));
                case "tsv" -> parts.add(HfDataset.fromTsv(f, true));
                case "json" -> parts.add(HfDataset.fromJson(f));
                case "jsonl", "ndjson" -> parts.add(HfDataset.fromJsonl(f));
                case "text", "txt" -> parts.add(HfDataset.fromText(f));
                default -> parts.add(HfDataset.fromFile(f));
            }
        }
        HfDataset out = HfDataset.concatenate(parts);
        int take = intArg(kw, "split_take", intArg(kw, "take", -1));
        if (take > 0) out = out.take(take);
        return out;
    }

    public static HfDataset loadFromDisk(Path dir) throws IOException {
        return HfDataset.loadFromDisk(dir);
    }

    public static HfDataset.DatasetDict loadLocal(Path path) throws IOException {
        return loadLocal(path, null, null, LoadConfig.defaults());
    }

    // ---- layout inference --------------------------------------------------

    /**
     * Infer config / split structure from relative file paths inside a dataset repo.
     *
     * <p>Recognises:
     * <ul>
     *   <li>{@code {config}/{split}-00000-of-00001.parquet} (glue, mmlu, imdb, squad…)</li>
     *   <li>{@code data/{split}.csv} / {@code data/{split}-*.parquet}</li>
     *   <li>{@code {split}.jsonl} at repo root</li>
     *   <li>flat single-config multi-split files</li>
     * </ul>
     */
    public static final class DatasetLayout {
        /** config -> split -> files */
        private final Map<String, Map<String, List<String>>> tree = new LinkedHashMap<>();
        private final List<String> allFiles = new ArrayList<>();

        public static DatasetLayout infer(List<String> files) {
            DatasetLayout layout = new DatasetLayout();
            for (String f : files) {
                layout.allFiles.add(f);
                Parsed p = parsePath(f);
                layout.tree
                        .computeIfAbsent(p.config, k -> new LinkedHashMap<>())
                        .computeIfAbsent(p.split, k -> new ArrayList<>())
                        .add(f);
            }
            // sort file lists for determinism
            for (Map<String, List<String>> splits : layout.tree.values()) {
                for (List<String> fl : splits.values()) Collections.sort(fl);
            }
            return layout;
        }

        public Set<String> configs() {
            return Collections.unmodifiableSet(tree.keySet());
        }

        public Set<String> splits() {
            Set<String> s = new LinkedHashSet<>();
            for (Map<String, List<String>> m : tree.values()) s.addAll(m.keySet());
            return s;
        }

        public Set<String> splits(String config) {
            Map<String, List<String>> m = tree.get(config);
            return m == null ? Set.of() : Collections.unmodifiableSet(m.keySet());
        }

        public List<String> filesFor(String config, List<String> splits) {
            List<String> out = new ArrayList<>();
            if (config != null && tree.containsKey(config)) {
                Map<String, List<String>> sm = tree.get(config);
                if (splits == null || splits.isEmpty()) {
                    for (List<String> fl : sm.values()) out.addAll(fl);
                } else {
                    for (String sp : splits) {
                        List<String> fl = sm.get(sp);
                        if (fl != null) out.addAll(fl);
                    }
                }
                return out;
            }
            // config null → union all configs (rare)
            for (Map<String, List<String>> sm : tree.values()) {
                if (splits == null || splits.isEmpty()) {
                    for (List<String> fl : sm.values()) out.addAll(fl);
                } else {
                    for (String sp : splits) {
                        List<String> fl = sm.get(sp);
                        if (fl != null) out.addAll(fl);
                    }
                }
            }
            return out;
        }

        public Map<String, List<String>> splitFiles(String config) {
            Map<String, List<String>> sm = tree.get(config);
            return sm == null ? Map.of() : Collections.unmodifiableMap(sm);
        }

        public List<String> allFiles() {
            return Collections.unmodifiableList(allFiles);
        }

        static final class Parsed {
            final String config;
            final String split;
            Parsed(String config, String split) {
                this.config = config;
                this.split = split;
            }
        }

        static Parsed parsePath(String path) {
            String p = path.replace('\\', '/');
            String[] parts = p.split("/");
            String file = parts[parts.length - 1];
            String parent = parts.length >= 2 ? parts[parts.length - 2] : null;

            String split = detectSplitInName(file);
            String config;

            if (parts.length >= 3) {
                // e.g. plain_text/train-00000.parquet  OR  data/train.csv
                config = parent;
                if (isGenericDir(config)) {
                    config = "default";
                }
                // mmlu-style: abstract_algebra/test-....parquet → config=abstract_algebra
            } else if (parts.length == 2) {
                config = parent;
                if (isGenericDir(config)) {
                    config = "default";
                }
            } else {
                config = "default";
            }

            if (split == null) {
                // parent directory is the split: subsamples/ur0.01/train/week_00.parquet
                if (parent != null && KNOWN_SPLITS.contains(parent.toLowerCase(Locale.ROOT))) {
                    split = parent.toLowerCase(Locale.ROOT);
                    config = parts.length >= 3 ? parts[parts.length - 3] : "default";
                    if (isGenericDir(config) && parts.length >= 4) {
                        config = parts[parts.length - 4];
                    }
                } else {
                    split = "train"; // default bucket for unrecognised
                }
            }

            // Deep recsys / multi-domain layouts:
            //   dataset/small/marketplace/events/01216.pq  → config=small
            //   dataset/full/retail/events/....pq          → config=full
            //   subsamples/ur0.01_ip0.01/train/...         → config=ur0.01_ip0.01 (already)
            String versionConfig = detectVersionConfig(parts);
            if (versionConfig != null) {
                config = versionConfig;
            }

            // Generic leaf dirs (events/files/raw) are poor config names — climb one level
            // when we still have a deeper path and no version marker won.
            if (isGenericDir(config) && parts.length >= 4) {
                String up = parts[parts.length - 3];
                if (!isGenericDir(up) && !KNOWN_SPLITS.contains(up.toLowerCase(Locale.ROOT))) {
                    config = up;
                }
            }

            // normalise common aliases
            split = normaliseSplit(split);
            if (config == null || config.isBlank() || isGenericDir(config)) {
                // last resort: prefer non-generic ancestor, else default
                config = firstMeaningfulAncestor(parts);
            }
            return new Parsed(config, split);
        }

        /** dirs that are containers, not dataset configs */
        private static boolean isGenericDir(String name) {
            if (name == null || name.isBlank()) return true;
            String n = name.toLowerCase(Locale.ROOT);
            return n.equals("data") || n.equals("dataset") || n.equals("datasets")
                    || n.equals("events") || n.equals("files") || n.equals("raw")
                    || n.equals("input") || n.equals("output") || n.equals("outputs")
                    || n.equals("images") || n.equals("image") || n.equals("audio")
                    || n.equals("videos") || n.equals("video") || n.equals("annotations")
                    || n.equals("meta") || n.equals("metadata") || n.equals("tables");
        }

        /**
         * {@code dataset/small/...} or {@code dataset/full/...} → {@code small}/{@code full}.
         * Also {@code subsamples/<name>/...} is handled by the parent-is-split branch.
         */
        private static String detectVersionConfig(String[] parts) {
            for (int i = 0; i < parts.length - 1; i++) {
                String s = parts[i].toLowerCase(Locale.ROOT);
                if ("small".equals(s) || "full".equals(s) || "tiny".equals(s)
                        || "mini".equals(s) || "sample".equals(s) || "preview".equals(s)) {
                    return s;
                }
                // subsample folder names: ur0.01_ip0.01, up0.01_ip0.01, …
                if ((s.startsWith("ur") || s.startsWith("up") || s.startsWith("ip"))
                        && (s.contains("0.") || s.contains("_"))) {
                    return parts[i]; // preserve original spelling
                }
            }
            return null;
        }

        private static String firstMeaningfulAncestor(String[] parts) {
            // walk from parent toward root; skip file name and generic dirs
            for (int i = parts.length - 2; i >= 0; i--) {
                if (!isGenericDir(parts[i])
                        && !KNOWN_SPLITS.contains(parts[i].toLowerCase(Locale.ROOT))) {
                    return parts[i];
                }
            }
            return "default";
        }

        static String detectSplitInName(String fileName) {
            String base = fileName;
            int dot = base.lastIndexOf('.');
            if (dot > 0) base = base.substring(0, dot);
            // strip compression already handled by extension; strip shard suffix
            base = base.replaceAll("-\\d{5}-of-\\d{5}$", "");
            // strip trailing hash suffixes like -3d4cd8309148a71f
            base = base.replaceAll("-[0-9a-f]{8,}$", "");

            String lower = base.toLowerCase(Locale.ROOT);
            // exact known split
            if (KNOWN_SPLITS.contains(lower)) return lower;
            // common alias not in KNOWN_SPLITS set
            if ("valid".equals(lower)) return "valid";

            // suffix match: All_Beauty.train / foo_test / bar-validation
            // (McAuley-Lab Amazon Reviews benchmark CSVs use this style)
            String best = null;
            for (String sp : KNOWN_SPLITS) {
                if (lower.endsWith("." + sp) || lower.endsWith("_" + sp) || lower.endsWith("-" + sp)) {
                    if (best == null || sp.length() > best.length()) best = sp;
                }
            }
            if (lower.endsWith(".valid") || lower.endsWith("_valid") || lower.endsWith("-valid")) {
                if (best == null || "valid".length() > best.length()) best = "valid";
            }
            if (best != null) return best;

            // prefix match: train_sft, train-gen, validation_matched, …
            // pick the longest known split that is a prefix (with separator) or exact
            best = null;
            for (String sp : KNOWN_SPLITS) {
                if (lower.equals(sp) || lower.startsWith(sp + "-") || lower.startsWith(sp + "_")
                        || lower.startsWith(sp + ".")) {
                    if (best == null || sp.length() > best.length()) best = sp;
                }
            }
            if (best != null) return best;

            // first token before - or _
            int cut = indexOfSep(lower);
            if (cut > 0) {
                String tok = lower.substring(0, cut);
                if (KNOWN_SPLITS.contains(tok)) return tok;
                if ("valid".equals(tok)) return "valid";
            }
            return null;
        }

        private static int indexOfSep(String s) {
            int a = s.indexOf('-');
            int b = s.indexOf('_');
            if (a < 0) return b;
            if (b < 0) return a;
            return Math.min(a, b);
        }

        static String normaliseSplit(String split) {
            if (split == null) return "train";
            String s = split.toLowerCase(Locale.ROOT);
            // Keep "dev" distinct (MMLU few-shot prompts ≠ validation).
            // Alias short forms: val / valid → validation; eval → test.
            return switch (s) {
                case "val", "valid" -> "validation";
                case "eval" -> "test";
                default -> s;
            };
        }
    }

    // ---- materialisation ---------------------------------------------------

    private static HfDataset.DatasetDict materialise(Path snap, List<String> files,
                                                      DatasetLayout layout, String config,
                                                      List<String> wantedSplits, LoadConfig c)
            throws IOException {
        // Only group the files that were actually selected/downloaded (respects maxFiles).
        // Do NOT re-expand from full layout — that logs false "missing" after truncation.
        Set<String> wanted = files == null ? Set.of() : new LinkedHashSet<>(files);
        Map<String, List<String>> bySplit = new LinkedHashMap<>();
        for (String f : wanted) {
            DatasetLayout.Parsed p = DatasetLayout.parsePath(f);
            if (wantedSplits != null && !wantedSplits.isEmpty()
                    && !wantedSplits.contains(p.split)
                    && !wantedSplits.contains("*")) {
                continue;
            }
            // Prefer layout's split assignment when the path is known there
            String sp = p.split;
            Map<String, List<String>> splitMap = layout.splitFiles(config);
            if (!splitMap.isEmpty()) {
                for (Map.Entry<String, List<String>> e : splitMap.entrySet()) {
                    if (e.getValue().contains(f)) {
                        sp = e.getKey();
                        break;
                    }
                }
            }
            bySplit.computeIfAbsent(sp, k -> new ArrayList<>()).add(f);
        }

        Map<String, HfDataset> splits = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> e : bySplit.entrySet()) {
            List<Path> paths = new ArrayList<>();
            for (String rel : e.getValue()) {
                Path p = snap.resolve(rel);
                if (Files.isRegularFile(p)) paths.add(p);
                else c.log("[HfDatasets] missing file in snapshot: " + rel);
            }
            if (paths.isEmpty()) continue;
            HfDataset ds = loadPaths(paths, c);
            if (c.take > 0) ds = ds.take(c.take);
            splits.put(e.getKey(), ds);
            c.log("[HfDatasets] split " + e.getKey() + " -> " + ds.size() + " rows from "
                    + paths.size() + " file(s)");
        }
        if (splits.isEmpty()) {
            throw new IOException("Materialised 0 splits from " + snap);
        }
        return new HfDataset.DatasetDict(splits);
    }

    private static HfDataset loadPaths(List<Path> paths, LoadConfig c) throws IOException {
        if (paths.size() == 1) {
            return loadOne(paths.get(0), c);
        }
        List<HfDataset> parts = new ArrayList<>(paths.size());
        for (Path p : paths) {
            parts.add(loadOne(p, c));
            if (c.take > 0) {
                int total = parts.stream().mapToInt(HfDataset::size).sum();
                if (total >= c.take) break;
            }
        }
        HfDataset cat = HfDataset.concatenate(parts);
        return c.take > 0 ? cat.take(c.take) : cat;
    }

    private static HfDataset loadOne(Path path, LoadConfig c) throws IOException {
        c.log("[HfDatasets] read " + path.getFileName()
                + (c.take > 0 ? " (take=" + c.take + ")" : ""));
        // Forward take so Parquet early-stops instead of materialising multi-M row shards.
        return HfDataset.fromFile(path, c.take);
    }

    // ---- local / cache helpers ---------------------------------------------

    public static HfDataset.DatasetDict loadLocal(Path path, String name, String split,
                                                   LoadConfig c) throws IOException {
        if (Files.isRegularFile(path)) {
            HfDataset ds = HfDataset.fromFile(path);
            if (c.take > 0) ds = ds.take(c.take);
            String sp = split == null ? "train" : split;
            return new HfDataset.DatasetDict(Map.of(sp, ds));
        }
        // directory: either saveToDisk layout or multi-file dataset folder
        Path marker = path.resolve("data.jsonl");
        if (Files.isRegularFile(marker)) {
            HfDataset ds = HfDataset.loadFromDisk(path);
            if (c.take > 0) ds = ds.take(c.take);
            return new HfDataset.DatasetDict(Map.of(split == null ? "train" : split, ds));
        }

        List<String> relFiles = new ArrayList<>();
        try (var walk = Files.walk(path)) {
            Path root = path;
            walk.filter(Files::isRegularFile)
                    .filter(p -> HfHub.isDatasetDataFile(p.getFileName().toString()))
                    .forEach(p -> relFiles.add(root.relativize(p).toString().replace('\\', '/')));
        }
        Collections.sort(relFiles);
        if (relFiles.isEmpty()) {
            throw new IOException("No data files under local path: " + path);
        }
        DatasetLayout layout = DatasetLayout.infer(relFiles);
        String config = resolveConfig(name, layout, c);
        List<String> wanted = resolveSplits(split, layout, config, c);
        List<String> files = layout.filesFor(config, wanted);
        return materialise(path, files, layout, config, wanted, c);
    }

    private static Path tryCachedSnapshot(HfHub hub, String datasetId, String revision) {
        try {
            String rev = revision == null ? "main" : revision;
            if (hub.cache().hasSnapshot("datasets", datasetId, rev)) {
                return hub.cache().snapshotPath("datasets", datasetId, rev);
            }
        } catch (Exception ignored) {}
        return null;
    }

    private static HfDataset.DatasetDict loadFromSnapshot(Path snap, String path, String name,
                                                           String split, LoadConfig c)
            throws IOException {
        List<String> relFiles = new ArrayList<>();
        try (var walk = Files.walk(snap)) {
            walk.filter(Files::isRegularFile)
                    .filter(p -> HfHub.isDatasetDataFile(p.getFileName().toString()))
                    .forEach(p -> relFiles.add(snap.relativize(p).toString().replace('\\', '/')));
        }
        Collections.sort(relFiles);
        if (relFiles.isEmpty()) {
            throw new IOException("Cached snapshot has no data files: " + snap);
        }
        DatasetLayout layout = DatasetLayout.infer(relFiles);
        String config = resolveConfig(name, layout, c);
        List<String> wanted = resolveSplits(split, layout, config, c);
        List<String> files = layout.filesFor(config, wanted);
        return materialise(snap, files, layout, config, wanted, c);
    }

    private static String resolveConfig(String name, DatasetLayout layout, LoadConfig c) {
        if (name != null && !name.isBlank()) {
            if (!layout.configs().contains(name)) {
                // allow case-insensitive
                for (String cfg : layout.configs()) {
                    if (cfg.equalsIgnoreCase(name)) return cfg;
                }
                throw new IllegalArgumentException("Unknown config '" + name
                        + "'. Available: " + layout.configs());
            }
            return name;
        }
        if (c.defaultConfig != null && layout.configs().contains(c.defaultConfig)) {
            return c.defaultConfig;
        }
        // Prefer small/default configs for large multi-domain repos (T-ECD etc.)
        for (String preferred : List.of("default", "plain_text", "all", "en",
                "small", "tiny", "mini", "sample", "preview")) {
            if (layout.configs().contains(preferred)) return preferred;
        }
        return layout.configs().iterator().next();
    }

    private static List<String> resolveSplits(String split, DatasetLayout layout,
                                               String config, LoadConfig c) {
        Set<String> available = layout.splits(config);
        if (available.isEmpty()) available = layout.splits();
        if (split == null || split.isBlank() || "*".equals(split) || "all".equalsIgnoreCase(split)) {
            return new ArrayList<>(available);
        }
        // support "train[:100]" style lightly → just name
        String sp = split;
        int bracket = sp.indexOf('[');
        if (bracket > 0) sp = sp.substring(0, bracket);
        sp = DatasetLayout.normaliseSplit(sp);
        if (available.contains(sp)) return List.of(sp);
        // fuzzy: train_sft when user asked train
        List<String> fuzzy = new ArrayList<>();
        for (String a : available) {
            if (a.equals(sp) || a.startsWith(sp + "_") || a.startsWith(sp + "-")) fuzzy.add(a);
        }
        if (!fuzzy.isEmpty()) return fuzzy;
        throw new IllegalArgumentException("Unknown split '" + split + "' for config '"
                + config + "'. Available: " + available);
    }

    private static int intArg(Map<String, Object> kw, String key, int def) {
        Object v = kw.get(key);
        if (v == null) return def;
        if (v instanceof Number n) return n.intValue();
        try { return Integer.parseInt(String.valueOf(v)); }
        catch (NumberFormatException e) { return def; }
    }

    /**
     * Re-order selected data files so columnar / structured shards come before
     * plain text (and raw JSON blobs after structured formats). Keeps relative
     * order within each priority band. Used before {@code maxFiles} truncation
     * so stress loads pick {@code data/train-*.parquet} over multi-GB
     * {@code *.txt} dumps that share the same split name.
     */
    static List<String> preferCompactDataFiles(List<String> files) {
        if (files == null || files.size() <= 1) return files;
        List<String> ranked = new ArrayList<>(files);
        ranked.sort((a, b) -> Integer.compare(dataFilePriority(a), dataFilePriority(b)));
        return ranked;
    }

    private static int dataFilePriority(String path) {
        if (path == null) return 99;
        String p = path.toLowerCase(Locale.ROOT);
        // strip compression
        for (String c : new String[]{".gz", ".bz2", ".zst", ".xz"}) {
            if (p.endsWith(c)) {
                p = p.substring(0, p.length() - c.length());
                break;
            }
        }
        if (p.endsWith(".parquet") || p.endsWith(".pq")
                || p.endsWith(".arrow") || p.endsWith(".feather") || p.endsWith(".ipc")
                || p.endsWith(".orc") || p.endsWith(".avro")) return 0;
        if (p.endsWith(".jsonl") || p.endsWith(".ndjson") || p.endsWith(".csv") || p.endsWith(".tsv")) return 1;
        if (p.endsWith(".json")) return 2;
        if (p.endsWith(".txt") || p.endsWith(".text")) return 3;
        return 4;
    }

    // ---- LoadConfig --------------------------------------------------------

    /**
     * Options for {@link #loadDataset(String, String, String, LoadConfig)}.
     *
     * <pre>{@code
     * LoadConfig cfg = LoadConfig.builder()
     *     .token(System.getenv("HF_TOKEN"))
     *     .endpoint("https://hf-mirror.com")
     *     .take(1000)
     *     .logger(System.out::println)
     *     .build();
     * }</pre>
     */
    public static final class LoadConfig {
        public final String token;
        public final String endpoint;
        public final String revision;
        public final boolean preferMirror;
        public final boolean offline;
        public final Path cacheDir;
        public final int take;
        public final int maxFiles;
        public final String defaultConfig;
        /** Optional path filters applied before layout inference (prefix or glob, e.g. dataset/small/). */
        public final List<String> allowPatterns;
        public final Consumer<String> logger;

        private LoadConfig(Builder b) {
            this.token = b.token;
            this.endpoint = b.endpoint;
            this.revision = b.revision == null ? "main" : b.revision;
            this.preferMirror = b.preferMirror;
            this.offline = b.offline;
            this.cacheDir = b.cacheDir;
            this.take = b.take;
            this.maxFiles = b.maxFiles;
            this.defaultConfig = b.defaultConfig;
            this.allowPatterns = b.allowPatterns == null || b.allowPatterns.isEmpty()
                    ? null : List.copyOf(b.allowPatterns);
            this.logger = b.logger == null ? s -> {} : b.logger;
        }

        public static LoadConfig defaults() {
            return builder().build();
        }

        public static Builder builder() {
            return new Builder();
        }

        HfHub buildHub() {
            HfHub.Builder hb = HfHub.create()
                    .offline(offline)
                    .logger(logger);
            if (token != null) hb.token(token);
            // else auto-resolve from env inside HfHub
            if (endpoint != null && !endpoint.isBlank()) {
                hb.endpoint(endpoint);
            } else if (preferMirror) {
                hb.endpoint(HfToken.resolveEndpoint(true));
            }
            if (cacheDir != null) hb.cacheDir(cacheDir);
            return hb.build();
        }

        void log(String msg) {
            logger.accept(msg);
        }

        public static final class Builder {
            private String token;
            private String endpoint;
            private String revision = "main";
            private boolean preferMirror;
            private boolean offline;
            private Path cacheDir;
            private int take = -1;
            private int maxFiles = -1;
            private String defaultConfig;
            private List<String> allowPatterns;
            private Consumer<String> logger;

            public Builder token(String token) { this.token = token; return this; }
            public Builder endpoint(String endpoint) { this.endpoint = endpoint; return this; }
            public Builder revision(String revision) { this.revision = revision; return this; }
            public Builder preferMirror(boolean preferMirror) {
                this.preferMirror = preferMirror;
                return this;
            }
            public Builder offline(boolean offline) { this.offline = offline; return this; }
            public Builder cacheDir(Path cacheDir) { this.cacheDir = cacheDir; return this; }
            /** Keep at most N rows per split after load (stress / smoke). */
            public Builder take(int take) { this.take = take; return this; }
            /** Download at most N data files (shard cap). */
            public Builder maxFiles(int maxFiles) { this.maxFiles = maxFiles; return this; }
            public Builder defaultConfig(String defaultConfig) {
                this.defaultConfig = defaultConfig;
                return this;
            }
            /** Restrict remote files before layout inference (prefix or glob, e.g. dataset/small/). */
            public Builder allowPatterns(List<String> allowPatterns) {
                this.allowPatterns = allowPatterns;
                return this;
            }
            public Builder allowPatterns(String... allowPatterns) {
                this.allowPatterns = allowPatterns == null ? null : List.of(allowPatterns);
                return this;
            }
            public Builder logger(Consumer<String> logger) { this.logger = logger; return this; }

            public LoadConfig build() { return new LoadConfig(this); }
        }
    }

    // ---- info helpers ------------------------------------------------------

    /** List configs inferred from remote tree (no download of data files). */
    public static List<String> listConfigs(String datasetId) throws IOException {
        return listConfigs(datasetId, LoadConfig.defaults());
    }

    public static List<String> listConfigs(String datasetId, LoadConfig cfg) throws IOException {
        LoadConfig c = cfg == null ? LoadConfig.defaults() : cfg;
        HfHub hub = c.buildHub();
        List<HfHub.RepoFile> tree = hub.listDatasetFiles(datasetId, c.revision);
        List<String> data = new ArrayList<>();
        for (HfHub.RepoFile rf : tree) {
            if (rf.isFile() && HfHub.isDatasetDataFile(rf.path)) data.add(rf.path);
        }
        return new ArrayList<>(DatasetLayout.infer(data).configs());
    }

    public static List<String> listSplits(String datasetId, String config) throws IOException {
        return listSplits(datasetId, config, LoadConfig.defaults());
    }

    public static List<String> listSplits(String datasetId, String config, LoadConfig cfg)
            throws IOException {
        LoadConfig c = cfg == null ? LoadConfig.defaults() : cfg;
        HfHub hub = c.buildHub();
        List<HfHub.RepoFile> tree = hub.listDatasetFiles(datasetId, c.revision);
        List<String> data = new ArrayList<>();
        for (HfHub.RepoFile rf : tree) {
            if (rf.isFile() && HfHub.isDatasetDataFile(rf.path)) data.add(rf.path);
        }
        DatasetLayout layout = DatasetLayout.infer(data);
        String conf = resolveConfig(config, layout, c);
        return new ArrayList<>(layout.splits(conf));
    }
}
