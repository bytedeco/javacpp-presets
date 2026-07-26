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
        this.endpoint = b.endpoint == null ? DEFAULT_ENDPOINT : b.endpoint;
        this.token = b.token;
        this.offline = b.offline;
        this.transfer = b.transfer == null ? HfTransfer.create().build() : b.transfer;
        this.logger = b.logger == null ? s -> {} : b.logger;
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
        transfer.downloadToFile(url, token, tmp, (done, total) -> {
            if (done == null || total == null || total <= 0) return;
            if (done % (64L << 20) < (1L << 20)) {
                logger.accept("[HfHub] " + filename + " " + (done >> 20) + "/" + (total >> 20) + " MiB");
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
        Path snap = snapshotDownload(repoId, revision, repoType, null);
        List<String> files = new ArrayList<>();
        try (var walk = Files.walk(snap)) {
            walk.filter(Files::isRegularFile)
                    .forEach(p -> files.add(snap.relativize(p).toString().replace('\\', '/')));
        }
        Collections.sort(files);
        return files;
    }

    // ---- builder ---------------------------------------------------------

    public static final class Builder {
        private HfCache cache;
        private String endpoint = DEFAULT_ENDPOINT;
        private String token;
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
