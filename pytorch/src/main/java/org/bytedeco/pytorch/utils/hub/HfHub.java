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
package org.bytedeco.pytorch.utils.hub;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Hugging Face Hub client (Java port of {@code huggingface_hub}).
 *
 * <p>Supports local offline seeding, cache-aware snapshot resolution, and optional
 * HTTP download of public files. Network is never required for unit / benchmark
 * paths that seed the cache via {@link #seedLocal}.
 *
 * <pre>{@code
 * HfHub hub = HfHub.create().cache(HfCache.of(tmp)).build();
 * hub.seedLocal("models", "acme/tiny-gpt", "main", Map.of(
 *     "config.json", "{\"model_type\":\"gpt2\"}",
 *     "tokenizer.json", "{\"version\":\"1.0\"}"
 * ));
 * Path snap = hub.snapshotDownload("acme/tiny-gpt");
 * }</pre>
 */
public final class HfHub {

    public static final String DEFAULT_ENDPOINT = "https://huggingface.co";

    private final HfCache cache;
    private final String endpoint;
    private final String token;
    private final boolean offline;
    private final HfTransfer transfer;
    private final Consumer<String> logger;

    private HfHub(Builder b) {
        this.cache = b.cache == null ? HfCache.defaultCache() : b.cache;
        String ep = b.endpoint == null ? HfToken.resolveEndpoint() : b.endpoint;
        while (ep != null && ep.endsWith("/")) ep = ep.substring(0, ep.length() - 1);
        this.endpoint = ep == null || ep.isBlank() ? DEFAULT_ENDPOINT : ep;
        // Auto-resolve token from env / token files when builder left it null.
        this.token = b.tokenExplicit ? b.token : HfToken.resolve(b.token);
        this.offline = b.offline;
        this.transfer = b.transfer == null ? HfTransfer.create().build() : b.transfer;
        this.logger = b.logger == null ? s -> {} : b.logger;
    }

    public String token() {
        return token;
    }

    /** Convenience: hub pre-wired with env token + default cache. */
    public static HfHub fromEnv() {
        return create().build();
    }

    /** Convenience: hub with env token and optional China mirror. */
    public static HfHub fromEnv(boolean preferMirror) {
        return create().endpoint(HfToken.resolveEndpoint(preferMirror)).build();
    }

    public static Builder create() {
        return new Builder();
    }

    public static Builder builder() {
        return new Builder();
    }

    public HfCache cache() {
        return cache;
    }

    public String endpoint() {
        return endpoint;
    }

    public boolean offline() {
        return offline;
    }

    public HfTransfer transfer() {
        return transfer;
    }

    // ---- local seed / offline helpers ------------------------------------

    /**
     * Seed a local snapshot without network — used by tests/benchmarks and for
     * packing private checkpoints into the HF cache layout.
     */
    public Path seedLocal(String repoType, String repoId, String revision,
                          Map<String, String> files) throws IOException {
        Objects.requireNonNull(repoId, "repoId");
        Objects.requireNonNull(files, "files");
        String rev = revision == null ? "main" : revision;
        cache.ensureLayout(repoType, repoId);
        // pin a deterministic commit from content hashes
        StringBuilder material = new StringBuilder();
        List<String> keys = new ArrayList<>(files.keySet());
        Collections.sort(keys);
        for (String k : keys) {
            material.append(k).append('=').append(HfCache.sha256(files.get(k))).append('\n');
        }
        String commit = HfCache.sha256(material.toString()).substring(0, 40);
        cache.writeRef(repoType, repoId, rev, commit);
        Path last = null;
        for (Map.Entry<String, String> e : files.entrySet()) {
            last = cache.storeText(repoType, repoId, rev, e.getKey(), e.getValue());
        }
        logger.accept("[HfHub] seeded " + repoType + "/" + repoId + "@" + rev
                + " (" + files.size() + " files, commit=" + commit + ")");
        return last == null ? cache.snapshotPath(repoType, repoId, rev) : last.getParent();
    }

    public Path seedLocal(String repoId, Map<String, String> files) throws IOException {
        return seedLocal("models", repoId, "main", files);
    }

    public Path seedBytes(String repoType, String repoId, String revision,
                          String relativePath, byte[] content) throws IOException {
        cache.ensureLayout(repoType, repoId);
        String rev = revision == null ? "main" : revision;
        if (!cache.hasSnapshot(repoType, repoId, rev)) {
            String commit = HfCache.sha256(content).substring(0, 40);
            cache.writeRef(repoType, repoId, rev, commit);
        }
        return cache.storeBlob(repoType, repoId, rev, relativePath, content);
    }

    // ---- download / snapshot ---------------------------------------------

    /**
     * Resolve a local snapshot directory. If offline or already cached, returns
     * the cache path. Otherwise attempts HTTP download of listed files.
     */
    public Path snapshotDownload(String repoId) throws IOException {
        return snapshotDownload(repoId, "main", "models", null);
    }

    public Path snapshotDownload(String repoId, String revision) throws IOException {
        return snapshotDownload(repoId, revision, "models", null);
    }

    public Path snapshotDownload(String repoId, String revision, String repoType,
                                 List<String> allowPatterns) throws IOException {
        Objects.requireNonNull(repoId, "repoId");
        String rev = revision == null ? "main" : revision;
        String type = repoType == null ? "models" : repoType;

        if (cache.hasSnapshot(type, repoId, rev)) {
            Path snap = cache.snapshotPath(type, repoId, rev);
            // If weights were requested but missing from an older config-only cache, continue fetch.
            boolean wantWeights = allowPatterns == null
                    || allowPatterns.stream().anyMatch(p -> p.contains("safetensors") || p.contains("*"));
            boolean hasWeights = false;
            try (var walk = Files.walk(snap)) {
                hasWeights = walk.anyMatch(p -> p.getFileName() != null
                        && p.getFileName().toString().endsWith(".safetensors"));
            } catch (IOException ignored) {}
            if (hasWeights || !wantWeights) {
                logger.accept("[HfHub] cache hit " + type + "/" + repoId + " -> " + snap);
                return snap;
            }
            logger.accept("[HfHub] cache hit without weights, fetching remaining files for " + repoId);
        }
        if (offline) {
            throw new IOException("Offline mode and snapshot not cached: " + type + "/" + repoId + "@" + rev);
        }
        // Default: config + tokenizer + generation + safetensors (+ index for shards).
        List<String> files = allowPatterns == null
                ? new ArrayList<>(Arrays.asList(
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
                "model.safetensors.index.json",
                "model.safetensors",
                "README.md"))
                : new ArrayList<>(allowPatterns);
        cache.ensureLayout(type, repoId);
        // Only mint a new commit when no snapshot exists yet — otherwise keep the
        // existing ref so config-only caches can be filled with weights in-place.
        if (!cache.hasSnapshot(type, repoId, rev)) {
            String commit = ("dl-" + System.currentTimeMillis());
            cache.writeRef(type, repoId, rev, commit);
        }
        int ok = 0;
        // Phase 1: small metadata files (byte[] ok)
        List<String> meta = new ArrayList<>();
        List<String> large = new ArrayList<>();
        for (String f : files) {
            if (f.endsWith(".safetensors") || f.endsWith(".bin") || f.endsWith(".pt") || f.endsWith(".gguf")) {
                large.add(f);
            } else {
                meta.add(f);
            }
        }
        for (String f : meta) {
            try {
                byte[] body = downloadFile(type, repoId, rev, f);
                if (body != null && body.length > 0) {
                    cache.storeBlob(type, repoId, rev, f, body);
                    ok++;
                }
            } catch (IOException e) {
                logger.accept("[HfHub] skip " + f + ": " + e.getMessage());
            }
        }
        // Phase 2: if index present, expand shard list
        try {
            Path snap = cache.snapshotPath(type, repoId, rev);
            Path index = snap.resolve("model.safetensors.index.json");
            if (Files.isRegularFile(index)) {
                String raw = Files.readString(index, StandardCharsets.UTF_8);
                // lightweight extract of "weight_map" values
                for (String shard : extractIndexShards(raw)) {
                    if (!large.contains(shard)) large.add(shard);
                }
                // single-file model.safetensors may 404 when sharded — drop it
                large.remove("model.safetensors");
            }
        } catch (IOException e) {
            logger.accept("[HfHub] index parse: " + e.getMessage());
        }
        // Phase 3: stream large weight files to disk
        for (String f : large) {
            try {
                Path stored = downloadFileToCache(type, repoId, rev, f);
                if (stored != null) ok++;
            } catch (IOException e) {
                logger.accept("[HfHub] skip " + f + ": " + e.getMessage());
            }
        }
        if (ok == 0) {
            throw new IOException("No files downloaded for " + repoId
                    + " — seed locally with seedLocal() or check network/token");
        }
        return cache.snapshotPath(type, repoId, rev);
    }

    /** Stream a (possibly multi-GB) file into the hub cache without full byte[]. */
    private Path downloadFileToCache(String repoType, String repoId, String revision,
                                     String filename) throws IOException {
        String url = resolveUrl(repoType, repoId, revision, filename);
        logger.accept("[HfHub] GET(stream) " + url);
        Path tmpDir = cache.blobsDir(repoType, repoId);
        Files.createDirectories(tmpDir);
        Path tmp = tmpDir.resolve("dl-" + filename.replace('/', '_') + ".part");
        final long[] lastLog = { -1L };
        transfer.downloadToFile(url, token, tmp, (done, total) -> {
            if (done == null) return;
            long d = done;
            long t = total == null ? -1L : total;
            // Log every ~8 MiB or at completion — avoid spamming 0/N lines.
            long step = 8L << 20;
            if (lastLog[0] < 0 || d - lastLog[0] >= step || (t > 0 && d >= t)) {
                lastLog[0] = d;
                if (t > 0) {
                    logger.accept("[HfHub] " + filename + " " + (d >> 20) + "/" + (t >> 20) + " MiB");
                } else {
                    logger.accept("[HfHub] " + filename + " " + (d >> 20) + " MiB");
                }
            }
        });
        Path stored = cache.storeFile(repoType, repoId, revision, filename, tmp);
        try { Files.deleteIfExists(tmp); } catch (IOException ignored) {}
        return stored;
    }

    private static List<String> extractIndexShards(String indexJson) {
        List<String> shards = new ArrayList<>();
        // values look like: "model-00001-of-00002.safetensors"
        int i = 0;
        while (i < indexJson.length()) {
            int p = indexJson.indexOf(".safetensors\"", i);
            if (p < 0) break;
            int start = indexJson.lastIndexOf('"', p);
            if (start >= 0 && start < p) {
                String name = indexJson.substring(start + 1, p + ".safetensors".length());
                if (!shards.contains(name) && !name.contains("index")) shards.add(name);
            }
            i = p + 1;
        }
        return shards;
    }

    private String resolveUrl(String repoType, String repoId, String revision, String filename) {
        if ("datasets".equals(repoType) || "dataset".equals(repoType)) {
            return endpoint + "/datasets/" + repoId + "/resolve/" + revision + "/" + filename;
        } else if ("spaces".equals(repoType) || "space".equals(repoType)) {
            return endpoint + "/spaces/" + repoId + "/resolve/" + revision + "/" + filename;
        }
        return endpoint + "/" + repoId + "/resolve/" + revision + "/" + filename;
    }

    public Path hfHubDownload(String repoId, String filename) throws IOException {
        return hfHubDownload(repoId, filename, "main", "models");
    }

    public Path hfHubDownload(String repoId, String filename, String revision,
                              String repoType) throws IOException {
        Objects.requireNonNull(filename, "filename");
        String rev = revision == null ? "main" : revision;
        String type = repoType == null ? "models" : repoType;
        Path snap = cache.snapshotPath(type, repoId, rev);
        Path local = snap.resolve(filename);
        if (Files.isRegularFile(local)) {
            return local;
        }
        if (offline) {
            throw new IOException("Offline and file missing: " + filename);
        }
        byte[] body = downloadFile(type, repoId, rev, filename);
        return cache.storeBlob(type, repoId, rev, filename, body);
    }

    private byte[] downloadFile(String repoType, String repoId, String revision,
                                String filename) throws IOException {
        String base = resolveUrl(repoType, repoId, revision, filename);
        logger.accept("[HfHub] GET " + base);
        return transfer.download(base, token);
    }

    // ---- metadata helpers ------------------------------------------------

    public ModelCard readModelCard(String repoId) throws IOException {
        return readModelCard(repoId, "main");
    }

    public ModelCard readModelCard(String repoId, String revision) throws IOException {
        Path snap = snapshotDownload(repoId, revision);
        Path readme = snap.resolve("README.md");
        if (!Files.isRegularFile(readme)) {
            return ModelCard.empty(repoId);
        }
        return ModelCard.parse(repoId, Files.readString(readme, StandardCharsets.UTF_8));
    }

    public Map<String, Object> modelInfo(String repoId) throws IOException {
        Map<String, Object> info = new LinkedHashMap<>(cache.info("models", repoId));
        Path snap = cache.hasSnapshot("models", repoId, "main")
                ? cache.snapshotPath("models", repoId, "main")
                : null;
        if (snap != null) {
            Path cfg = snap.resolve("config.json");
            if (Files.isRegularFile(cfg)) {
                info.put("config.json", Files.readString(cfg, StandardCharsets.UTF_8));
            }
            List<String> files = new ArrayList<>();
            try (var walk = Files.walk(snap)) {
                Path finalSnap = snap;
                walk.filter(Files::isRegularFile)
                        .forEach(p -> files.add(finalSnap.relativize(p).toString()));
            }
            info.put("files", files);
        }
        return info;
    }

    public List<String> listRepoFiles(String repoId) throws IOException {
        return listRepoFiles(repoId, "main", "models");
    }

    public List<String> listRepoFiles(String repoId, String revision, String repoType) throws IOException {
        // Prefer Hub tree API (no full download) when online; fall back to local snapshot walk.
        try {
            if (!offline) {
                List<RepoFile> remote = listRepoTree(repoId, revision, repoType, true);
                if (!remote.isEmpty()) {
                    List<String> files = new ArrayList<>(remote.size());
                    for (RepoFile f : remote) {
                        if (f.isFile()) files.add(f.path);
                    }
                    Collections.sort(files);
                    return files;
                }
            }
        } catch (IOException e) {
            logger.accept("[HfHub] tree API unavailable, falling back to local: " + e.getMessage());
        }
        Path snap = cache.hasSnapshot(repoType == null ? "models" : repoType, repoId, revision == null ? "main" : revision)
                ? cache.snapshotPath(repoType == null ? "models" : repoType, repoId, revision == null ? "main" : revision)
                : snapshotDownload(repoId, revision, repoType, null);
        List<String> files = new ArrayList<>();
        if (snap != null && Files.isDirectory(snap)) {
            try (var walk = Files.walk(snap)) {
                walk.filter(Files::isRegularFile)
                        .forEach(p -> files.add(snap.relativize(p).toString().replace('\\', '/')));
            }
        }
        Collections.sort(files);
        return files;
    }

    // ---- Hub REST / tree API (datasets + models) ---------------------------

    /**
     * List files/dirs in a Hub repo via {@code /api/{repoType}/{repoId}/tree/{revision}}.
     * Handles pagination via the {@code Link: &lt;...&gt;; rel="next"} header.
     */
    public List<RepoFile> listRepoTree(String repoId, String revision, String repoType,
                                       boolean recursive) throws IOException {
        Objects.requireNonNull(repoId, "repoId");
        if (offline) throw new IOException("Offline: cannot list remote tree for " + repoId);
        String rev = revision == null || revision.isBlank() ? "main" : revision;
        String type = normalizeRepoType(repoType);
        String apiBase = endpoint + "/api/" + type + "/" + repoId + "/tree/" + rev
                + (recursive ? "?recursive=1" : "");
        List<RepoFile> all = new ArrayList<>();
        String url = apiBase;
        int pages = 0;
        while (url != null && pages++ < 200) {
            logger.accept("[HfHub] TREE " + url);
            ApiResponse resp = apiGet(url);
            if (resp.code == 401 || resp.code == 403) {
                throw new IOException("Hub tree auth failed HTTP " + resp.code
                        + " for " + repoId + " (token=" + HfToken.mask(token) + "): " + resp.body);
            }
            if (resp.code == 404) {
                throw new IOException("Repo not found: " + type + "/" + repoId + "@" + rev);
            }
            if (resp.code < 200 || resp.code >= 300) {
                throw new IOException("Hub tree HTTP " + resp.code + " for " + url + ": "
                        + truncate(resp.body, 300));
            }
            all.addAll(parseTreeJson(resp.body));
            url = resp.nextLink;
        }
        return all;
    }

    /** Dataset-specific tree listing. */
    public List<RepoFile> listDatasetFiles(String datasetId) throws IOException {
        return listDatasetFiles(datasetId, "main");
    }

    public List<RepoFile> listDatasetFiles(String datasetId, String revision) throws IOException {
        return listRepoTree(datasetId, revision, "datasets", true);
    }

    /**
     * Fetch {@code /api/datasets/{id}} metadata (siblings, cardData.configs, …).
     * Returns a loosely-typed map parsed from JSON.
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> datasetInfo(String datasetId) throws IOException {
        return datasetInfo(datasetId, "main");
    }

    public Map<String, Object> datasetInfo(String datasetId, String revision) throws IOException {
        Objects.requireNonNull(datasetId, "datasetId");
        if (offline) throw new IOException("Offline: cannot fetch dataset info for " + datasetId);
        String rev = revision == null || revision.isBlank() ? "main" : revision;
        String url = endpoint + "/api/datasets/" + datasetId + "?revision=" + rev;
        logger.accept("[HfHub] GET " + url);
        ApiResponse resp = apiGet(url);
        if (resp.code < 200 || resp.code >= 300) {
            throw new IOException("datasetInfo HTTP " + resp.code + ": " + truncate(resp.body, 300));
        }
        return parseJsonObjectLoose(resp.body);
    }

    /**
     * Download selected files from a Hub repo into the local cache and return the
     * snapshot directory. Unlike {@link #snapshotDownload} (model-oriented defaults),
     * this takes an explicit file list — ideal for datasets.
     *
     * @param allowPatterns optional glob-ish filters ({@code *.parquet}, {@code cola/*},
     *                      {@code **}/{@code train-*}); {@code null} downloads every listed file.
     */
    public Path downloadRepoFiles(String repoId, String revision, String repoType,
                                   List<String> files, List<String> allowPatterns)
            throws IOException {
        Objects.requireNonNull(repoId, "repoId");
        String rev = revision == null || revision.isBlank() ? "main" : revision;
        String type = normalizeRepoType(repoType);
        List<String> wanted = filterByPatterns(files, allowPatterns);
        if (wanted.isEmpty()) {
            throw new IOException("No files matched patterns " + allowPatterns + " in " + repoId);
        }
        cache.ensureLayout(type, repoId);
        if (!cache.hasSnapshot(type, repoId, rev)) {
            String commit = "dl-" + System.currentTimeMillis();
            cache.writeRef(type, repoId, rev, commit);
        }
        int ok = 0;
        List<String> errors = new ArrayList<>();
        for (String f : wanted) {
            try {
                Path local = cache.snapshotPath(type, repoId, rev).resolve(f);
                if (Files.isRegularFile(local) && Files.size(local) > 0) {
                    ok++;
                    continue;
                }
                Path stored = downloadFileToCache(type, repoId, rev, f);
                if (stored != null) ok++;
            } catch (IOException e) {
                errors.add(f + ": " + e.getMessage());
                logger.accept("[HfHub] skip " + f + ": " + e.getMessage());
            }
        }
        if (ok == 0) {
            throw new IOException("No files downloaded for " + type + "/" + repoId
                    + " — errors: " + errors);
        }
        return cache.snapshotPath(type, repoId, rev);
    }

    /**
     * Download a dataset snapshot, optionally filtered by path patterns.
     * When {@code allowPatterns} is null, lists the remote tree and downloads all
     * data-like files (parquet/arrow/csv/json/jsonl/tsv/txt + compressed variants).
     */
    public Path downloadDataset(String datasetId, String revision,
                                 List<String> allowPatterns) throws IOException {
        Objects.requireNonNull(datasetId, "datasetId");
        String rev = revision == null || revision.isBlank() ? "main" : revision;
        List<String> files;
        if (offline) {
            Path snap = cache.snapshotPath("datasets", datasetId, rev);
            if (!Files.isDirectory(snap)) {
                throw new IOException("Offline and dataset not cached: " + datasetId);
            }
            return snap;
        }
        List<RepoFile> tree = listDatasetFiles(datasetId, rev);
        files = new ArrayList<>();
        for (RepoFile rf : tree) {
            if (rf.isFile()) files.add(rf.path);
        }
        if (allowPatterns == null || allowPatterns.isEmpty()) {
            // Prefer data files; still keep README/dataset_infos if present for metadata.
            List<String> data = new ArrayList<>();
            for (String f : files) {
                if (isDatasetDataFile(f) || isDatasetMetaFile(f)) data.add(f);
            }
            if (!data.isEmpty()) files = data;
        }
        return downloadRepoFiles(datasetId, rev, "datasets", files, allowPatterns);
    }

    public Path downloadDataset(String datasetId) throws IOException {
        return downloadDataset(datasetId, "main", null);
    }

    /** True for common tabular / text dataset payload extensions. */
    public static boolean isDatasetDataFile(String path) {
        if (path == null) return false;
        String p = path.toLowerCase();
        // strip compression suffix for extension check
        if (p.endsWith(".gz") || p.endsWith(".bz2") || p.endsWith(".zst") || p.endsWith(".xz")) {
            int dot = p.lastIndexOf('.', p.length() - 4);
            if (dot > 0) p = p.substring(0, p.length() - (p.length() - p.lastIndexOf('.')));
            // recompute on basename
            p = path.toLowerCase();
            for (String c : new String[]{".gz", ".bz2", ".zst", ".xz"}) {
                if (p.endsWith(c)) {
                    p = p.substring(0, p.length() - c.length());
                    break;
                }
            }
        }
        return p.endsWith(".parquet") || p.endsWith(".pq")
                || p.endsWith(".arrow") || p.endsWith(".feather") || p.endsWith(".ipc")
                || p.endsWith(".orc") || p.endsWith(".avro")
                || p.endsWith(".csv") || p.endsWith(".tsv")
                || p.endsWith(".json") || p.endsWith(".jsonl") || p.endsWith(".ndjson")
                || p.endsWith(".txt") || p.endsWith(".text");
    }

    public static boolean isDatasetMetaFile(String path) {
        if (path == null) return false;
        String name = path.contains("/") ? path.substring(path.lastIndexOf('/') + 1) : path;
        return "README.md".equalsIgnoreCase(name)
                || "dataset_infos.json".equalsIgnoreCase(name)
                || "dataset_info.json".equalsIgnoreCase(name)
                || ".gitattributes".equalsIgnoreCase(name);
    }

    // ---- pattern / glob helpers --------------------------------------------

    /**
     * Simple glob matching for Hub paths:
     * {@code *} any within segment, {@code **} any path, {@code ?} one char.
     * Also accepts plain substring prefixes like {@code cola/}.
     */
    public static boolean matchPattern(String path, String pattern) {
        if (pattern == null || pattern.isEmpty() || "*".equals(pattern) || "**".equals(pattern)) {
            return true;
        }
        if (path == null) return false;
        String p = path.replace('\\', '/');
        String g = pattern.replace('\\', '/');
        // plain prefix directory
        if (!g.contains("*") && !g.contains("?") && g.endsWith("/")) {
            return p.startsWith(g) || p.startsWith(g.substring(0, g.length() - 1) + "/");
        }
        if (!g.contains("*") && !g.contains("?")) {
            return p.equals(g) || p.endsWith("/" + g) || p.startsWith(g + "/");
        }
        return globMatch(p, g);
    }

    static boolean globMatch(String text, String pattern) {
        // Convert glob to regex
        StringBuilder re = new StringBuilder("^");
        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            if (c == '*') {
                if (i + 1 < pattern.length() && pattern.charAt(i + 1) == '*') {
                    re.append(".*");
                    i++;
                    if (i + 1 < pattern.length() && pattern.charAt(i + 1) == '/') i++;
                } else {
                    re.append("[^/]*");
                }
            } else if (c == '?') {
                re.append("[^/]");
            } else if (".+()[]{}^$|\\".indexOf(c) >= 0) {
                re.append('\\').append(c);
            } else {
                re.append(c);
            }
        }
        re.append('$');
        return text.matches(re.toString());
    }

    static List<String> filterByPatterns(List<String> files, List<String> patterns) {
        if (files == null) return List.of();
        if (patterns == null || patterns.isEmpty()) return new ArrayList<>(files);
        List<String> out = new ArrayList<>();
        for (String f : files) {
            for (String pat : patterns) {
                if (matchPattern(f, pat)) {
                    out.add(f);
                    break;
                }
            }
        }
        return out;
    }

    static String normalizeRepoType(String repoType) {
        if (repoType == null || repoType.isBlank()) return "models";
        String t = repoType.toLowerCase();
        if ("dataset".equals(t)) return "datasets";
        if ("model".equals(t)) return "models";
        if ("space".equals(t)) return "spaces";
        if (!t.endsWith("s") && ("dataset".equals(t) || "model".equals(t) || "space".equals(t))) {
            return t + "s";
        }
        return t;
    }

    // ---- HTTP helpers for Hub API ------------------------------------------

    private static final class ApiResponse {
        final int code;
        final String body;
        final String nextLink;
        ApiResponse(int code, String body, String nextLink) {
            this.code = code;
            this.body = body;
            this.nextLink = nextLink;
        }
    }

    private ApiResponse apiGet(String url) throws IOException {
        // Follow a few redirects manually so mirrors (307) work with auth headers.
        String current = url;
        for (int hop = 0; hop < 8; hop++) {
            java.net.HttpURLConnection conn =
                    (java.net.HttpURLConnection) URI.create(current).toURL().openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(15_000);
            conn.setReadTimeout(60_000);
            conn.setInstanceFollowRedirects(false);
            conn.setRequestProperty("User-Agent", "javacpp-pytorch-hf-hub/1.0");
            conn.setRequestProperty("Accept", "application/json");
            if (token != null && !token.isBlank()) {
                conn.setRequestProperty("Authorization", "Bearer " + token);
            }
            int code;
            try {
                code = conn.getResponseCode();
            } catch (IOException e) {
                conn.disconnect();
                throw e;
            }
            if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
                String loc = conn.getHeaderField("Location");
                conn.disconnect();
                if (loc == null || loc.isBlank()) {
                    throw new IOException("Redirect without Location from " + current);
                }
                if (loc.startsWith("/")) {
                    URI base = URI.create(current);
                    current = base.getScheme() + "://" + base.getHost()
                            + (base.getPort() > 0 ? ":" + base.getPort() : "") + loc;
                } else if (!loc.startsWith("http")) {
                    current = URI.create(current).resolve(loc).toString();
                } else {
                    current = loc;
                }
                continue;
            }
            String body;
            try (InputStream in = code >= 400 ? conn.getErrorStream() : conn.getInputStream()) {
                if (in == null) body = "";
                else body = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            } finally {
                // capture Link before disconnect
            }
            String link = conn.getHeaderField("Link");
            conn.disconnect();
            String next = parseNextLink(link);
            return new ApiResponse(code, body, next);
        }
        throw new IOException("Too many redirects for " + url);
    }

    private static String parseNextLink(String linkHeader) {
        if (linkHeader == null || linkHeader.isBlank()) return null;
        // e.g. <https://...&cursor=...>; rel="next", <...>; rel="prev"
        for (String part : linkHeader.split(",")) {
            String p = part.trim();
            if (!p.contains("rel=\"next\"") && !p.contains("rel='next'")) continue;
            int lt = p.indexOf('<');
            int gt = p.indexOf('>');
            if (lt >= 0 && gt > lt) return p.substring(lt + 1, gt);
        }
        return null;
    }

    static List<RepoFile> parseTreeJson(String json) {
        List<RepoFile> out = new ArrayList<>();
        if (json == null || json.isBlank()) return out;
        // Expect a JSON array of objects with path/type/size/oid.
        String s = json.trim();
        if (!s.startsWith("[")) {
            // sometimes wrapped
            int i = s.indexOf('[');
            if (i < 0) return out;
            s = s.substring(i);
        }
        int i = 1;
        while (i < s.length()) {
            while (i < s.length() && (Character.isWhitespace(s.charAt(i)) || s.charAt(i) == ',')) i++;
            if (i >= s.length() || s.charAt(i) == ']') break;
            if (s.charAt(i) != '{') { i++; continue; }
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
            String obj = s.substring(start, i);
            String path = extractJsonString(obj, "path");
            if (path == null) path = extractJsonString(obj, "rfilename");
            String type = extractJsonString(obj, "type");
            long size = extractJsonLong(obj, "size");
            String oid = extractJsonString(obj, "oid");
            if (path != null) out.add(new RepoFile(path, type == null ? "file" : type, size, oid));
        }
        return out;
    }

    static String extractJsonString(String obj, String key) {
        String pat = "\"" + key + "\"";
        int k = obj.indexOf(pat);
        if (k < 0) return null;
        int colon = obj.indexOf(':', k + pat.length());
        if (colon < 0) return null;
        int i = colon + 1;
        while (i < obj.length() && Character.isWhitespace(obj.charAt(i))) i++;
        if (i >= obj.length()) return null;
        if (obj.charAt(i) == 'n' && obj.startsWith("null", i)) return null;
        if (obj.charAt(i) != '"') return null;
        i++;
        StringBuilder sb = new StringBuilder();
        while (i < obj.length() && obj.charAt(i) != '"') {
            if (obj.charAt(i) == '\\' && i + 1 < obj.length()) {
                sb.append(obj.charAt(++i));
                i++;
            } else {
                sb.append(obj.charAt(i++));
            }
        }
        return sb.toString();
    }

    static long extractJsonLong(String obj, String key) {
        String pat = "\"" + key + "\"";
        int k = obj.indexOf(pat);
        if (k < 0) return -1L;
        int colon = obj.indexOf(':', k + pat.length());
        if (colon < 0) return -1L;
        int i = colon + 1;
        while (i < obj.length() && Character.isWhitespace(obj.charAt(i))) i++;
        int start = i;
        while (i < obj.length() && "+-0123456789".indexOf(obj.charAt(i)) >= 0) i++;
        if (start == i) return -1L;
        try { return Long.parseLong(obj.substring(start, i)); }
        catch (NumberFormatException e) { return -1L; }
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> parseJsonObjectLoose(String json) {
        // Keep hub free of datasets-package dependency: extract common fields loosely.
        Map<String, Object> m = new LinkedHashMap<>();
        if (json == null) return m;
        String id = extractJsonString(json, "id");
        if (id != null) m.put("id", id);
        String sha = extractJsonString(json, "sha");
        if (sha != null) m.put("sha", sha);
        String author = extractJsonString(json, "author");
        if (author != null) m.put("author", author);
        List<String> siblings = new ArrayList<>();
        int idx = 0;
        while (true) {
            int p = json.indexOf("\"rfilename\"", idx);
            if (p < 0) p = json.indexOf("\"path\"", idx);
            if (p < 0) break;
            String key = json.regionMatches(p + 1, "rfilename", 0, 9) ? "rfilename" : "path";
            String v = extractJsonString(json.substring(p, Math.min(json.length(), p + 400)), key);
            if (v != null && !siblings.contains(v)) siblings.add(v);
            idx = p + 10;
        }
        if (!siblings.isEmpty()) m.put("siblings", siblings);
        // configs from cardData
        List<String> configs = new ArrayList<>();
        int cidx = 0;
        while (true) {
            int p = json.indexOf("\"config_name\"", cidx);
            if (p < 0) break;
            String v = extractJsonString(json.substring(p, Math.min(json.length(), p + 200)), "config_name");
            if (v != null && !configs.contains(v)) configs.add(v);
            cidx = p + 12;
        }
        if (!configs.isEmpty()) m.put("configs", configs);
        if (m.isEmpty()) m.put("_raw", truncate(json, 500));
        return m;
    }

    private static String truncate(String s, int n) {
        if (s == null) return "";
        return s.length() <= n ? s : s.substring(0, n) + "...";
    }

    /** One entry from Hub tree / siblings listing. */
    public static final class RepoFile {
        public final String path;
        public final String type;
        public final long size;
        public final String oid;

        public RepoFile(String path, String type, long size, String oid) {
            this.path = path;
            this.type = type == null ? "file" : type;
            this.size = size;
            this.oid = oid;
        }

        public boolean isFile() {
            return !"directory".equalsIgnoreCase(type) && !"dir".equalsIgnoreCase(type);
        }

        public boolean isDirectory() {
            return "directory".equalsIgnoreCase(type) || "dir".equalsIgnoreCase(type);
        }

        public String fileName() {
            if (path == null) return "";
            int s = path.lastIndexOf('/');
            return s < 0 ? path : path.substring(s + 1);
        }

        @Override
        public String toString() {
            return type + ":" + path + (size >= 0 ? "(" + size + ")" : "");
        }
    }

    // ---- builder ---------------------------------------------------------

    public static final class Builder {
        private HfCache cache;
        private String endpoint;
        private String token;
        /** When true, do not fall back to env/file token (caller set null deliberately). */
        private boolean tokenExplicit;
        private boolean offline;
        private HfTransfer transfer;
        private Consumer<String> logger;

        public Builder cache(HfCache cache) {
            this.cache = cache;
            return this;
        }

        public Builder cacheDir(Path dir) {
            this.cache = HfCache.of(dir);
            return this;
        }

        public Builder endpoint(String endpoint) {
            this.endpoint = endpoint;
            return this;
        }

        public Builder token(String token) {
            this.token = token;
            this.tokenExplicit = true;
            return this;
        }

        /** Use env/file token resolution (default). */
        public Builder tokenFromEnv() {
            this.token = null;
            this.tokenExplicit = false;
            return this;
        }

        public Builder offline(boolean offline) {
            this.offline = offline;
            return this;
        }

        public Builder transfer(HfTransfer transfer) {
            this.transfer = transfer;
            return this;
        }

        public Builder logger(Consumer<String> logger) {
            this.logger = logger;
            return this;
        }

        public HfHub build() {
            return new HfHub(this);
        }
    }

    // ---- model card ------------------------------------------------------

    /** Minimal model card with YAML front-matter extraction. */
    public static final class ModelCard {
        private final String repoId;
        private final String raw;
        private final Map<String, String> meta;
        private final String body;

        private ModelCard(String repoId, String raw, Map<String, String> meta, String body) {
            this.repoId = repoId;
            this.raw = raw;
            this.meta = meta;
            this.body = body;
        }

        public static ModelCard empty(String repoId) {
            return new ModelCard(repoId, "", Map.of(), "");
        }

        public static ModelCard parse(String repoId, String markdown) {
            Map<String, String> meta = new LinkedHashMap<>();
            String body = markdown == null ? "" : markdown;
            if (body.startsWith("---")) {
                int end = body.indexOf("---", 3);
                if (end > 0) {
                    String yaml = body.substring(3, end).trim();
                    body = body.substring(end + 3).trim();
                    for (String line : yaml.split("\n")) {
                        int c = line.indexOf(':');
                        if (c > 0) {
                            meta.put(line.substring(0, c).trim(), line.substring(c + 1).trim());
                        }
                    }
                }
            }
            return new ModelCard(repoId, markdown == null ? "" : markdown, meta, body);
        }

        public String repoId() { return repoId; }
        public String raw() { return raw; }
        public Map<String, String> meta() { return Collections.unmodifiableMap(meta); }
        public String body() { return body; }
        public String license() { return meta.getOrDefault("license", ""); }
        public String libraryName() { return meta.getOrDefault("library_name", ""); }
    }
}
