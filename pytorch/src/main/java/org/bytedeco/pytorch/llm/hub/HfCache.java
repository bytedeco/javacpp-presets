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
package org.bytedeco.pytorch.llm.hub;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

/**
 * Local Hugging Face Hub cache layout (mirrors {@code HF_HOME}/{@code huggingface_hub}).
 *
 * <pre>
 *   cache/
 *     models--org--name/
 *       refs/main
 *       snapshots/&lt;rev&gt;/config.json
 *       blobs/&lt;sha&gt;
 * </pre>
 */
public final class HfCache {

    public static final String DEFAULT_ENV = "HF_HOME";
    public static final String DEFAULT_HUB_SUBDIR = "hub";

    private final Path root;
    private final Map<String, Path> snapshotIndex = new ConcurrentHashMap<>();

    public HfCache(Path root) {
        this.root = Objects.requireNonNull(root, "root").toAbsolutePath().normalize();
    }

    public static HfCache defaultCache() {
        String home = System.getenv(DEFAULT_ENV);
        if (home == null || home.isBlank()) {
            home = System.getProperty("user.home") + "/.cache/huggingface";
        }
        return new HfCache(Path.of(home, DEFAULT_HUB_SUBDIR));
    }

    public static HfCache of(Path root) {
        return new HfCache(root);
    }

    public Path root() {
        return root;
    }

    /** Repo folder name: {@code models--org--name} / {@code datasets--org--name}. */
    public static String repoFolderName(String repoType, String repoId) {
        Objects.requireNonNull(repoId, "repoId");
        String type = (repoType == null || repoType.isBlank()) ? "models" : repoType;
        if (!type.endsWith("s")) {
            type = type + "s";
        }
        return type + "--" + repoId.replace("/", "--");
    }

    public Path repoDir(String repoType, String repoId) {
        return root.resolve(repoFolderName(repoType, repoId));
    }

    public Path snapshotsDir(String repoType, String repoId) {
        return repoDir(repoType, repoId).resolve("snapshots");
    }

    public Path blobsDir(String repoType, String repoId) {
        return repoDir(repoType, repoId).resolve("blobs");
    }

    public Path refsDir(String repoType, String repoId) {
        return repoDir(repoType, repoId).resolve("refs");
    }

    public void ensureLayout(String repoType, String repoId) throws IOException {
        Files.createDirectories(snapshotsDir(repoType, repoId));
        Files.createDirectories(blobsDir(repoType, repoId));
        Files.createDirectories(refsDir(repoType, repoId));
    }

    /** Resolve revision pointer under refs/, defaulting to {@code main}. */
    public String resolveRevision(String repoType, String repoId, String revision) throws IOException {
        String rev = (revision == null || revision.isBlank()) ? "main" : revision;
        Path ref = refsDir(repoType, repoId).resolve(rev);
        if (Files.isRegularFile(ref)) {
            return Files.readString(ref, StandardCharsets.UTF_8).trim();
        }
        return rev;
    }

    public void writeRef(String repoType, String repoId, String revision, String commitHash) throws IOException {
        ensureLayout(repoType, repoId);
        String rev = (revision == null || revision.isBlank()) ? "main" : revision;
        Path ref = refsDir(repoType, repoId).resolve(rev);
        Files.createDirectories(ref.getParent());
        Files.writeString(ref, commitHash, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }

    public Path snapshotPath(String repoType, String repoId, String revision) throws IOException {
        String commit = resolveRevision(repoType, repoId, revision);
        Path snap = snapshotsDir(repoType, repoId).resolve(commit);
        snapshotIndex.put(repoFolderName(repoType, repoId) + "@" + commit, snap);
        return snap;
    }

    /**
     * Store raw bytes under blobs/ and hardlink (or copy) into a snapshot path.
     *
     * @return path inside the snapshot
     */
    public Path storeBlob(String repoType, String repoId, String revision,
                          String relativePath, byte[] content) throws IOException {
        ensureLayout(repoType, repoId);
        String sha = sha256(content);
        Path blob = blobsDir(repoType, repoId).resolve(sha);
        if (!Files.exists(blob)) {
            Path tmp = blob.resolveSibling(sha + ".tmp");
            Files.write(tmp, content, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
            Files.move(tmp, blob, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        }
        String commit = resolveRevision(repoType, repoId, revision);
        // first store: pin revision -> commit if ref missing
        Path ref = refsDir(repoType, repoId).resolve(
                (revision == null || revision.isBlank()) ? "main" : revision);
        if (!Files.exists(ref)) {
            writeRef(repoType, repoId, revision, commit.equals(
                    (revision == null || revision.isBlank()) ? "main" : revision)
                    ? sha.substring(0, Math.min(40, sha.length()))
                    : commit);
            commit = resolveRevision(repoType, repoId, revision);
        }
        Path snap = snapshotsDir(repoType, repoId).resolve(commit);
        Path dest = snap.resolve(relativePath);
        Files.createDirectories(dest.getParent() == null ? snap : dest.getParent());
        if (!Files.exists(dest)) {
            try {
                Files.createLink(dest, blob);
            } catch (UnsupportedOperationException | IOException e) {
                Files.copy(blob, dest, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        return dest;
    }

    /** Write text content into cache snapshot (convenience). */
    public Path storeText(String repoType, String repoId, String revision,
                          String relativePath, String text) throws IOException {
        return storeBlob(repoType, repoId, revision, relativePath,
                text.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Place an already-downloaded file into the snapshot (for multi-GB weights).
     * Copies into {@code blobs/} then hardlinks into the snapshot path.
     * Prefer this over {@link #storeBlob} when content must not fit in a {@code byte[]}.
     */
    public Path storeFile(String repoType, String repoId, String revision,
                          String relativePath, Path sourceFile) throws IOException {
        ensureLayout(repoType, repoId);
        Objects.requireNonNull(sourceFile, "sourceFile");
        if (!Files.isRegularFile(sourceFile)) {
            throw new IOException("Not a regular file: " + sourceFile);
        }
        // hash via streaming
        String sha;
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (var in = Files.newInputStream(sourceFile)) {
                byte[] buf = new byte[1 << 20];
                int n;
                while ((n = in.read(buf)) >= 0) {
                    if (n > 0) md.update(buf, 0, n);
                }
            }
            sha = HexFormat.of().formatHex(md.digest());
        } catch (Exception e) {
            throw new IOException("SHA-256 failed for " + sourceFile, e);
        }
        Path blob = blobsDir(repoType, repoId).resolve(sha);
        if (!Files.exists(blob)) {
            Path tmp = blob.resolveSibling(sha + ".tmp");
            Files.copy(sourceFile, tmp, StandardCopyOption.REPLACE_EXISTING);
            Files.move(tmp, blob, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        }
        String commit = resolveRevision(repoType, repoId, revision);
        Path ref = refsDir(repoType, repoId).resolve(
                (revision == null || revision.isBlank()) ? "main" : revision);
        if (!Files.exists(ref)) {
            writeRef(repoType, repoId, revision, sha.substring(0, Math.min(40, sha.length())));
            commit = resolveRevision(repoType, repoId, revision);
        }
        Path snap = snapshotsDir(repoType, repoId).resolve(commit);
        Path dest = snap.resolve(relativePath);
        Files.createDirectories(dest.getParent() == null ? snap : dest.getParent());
        if (!Files.exists(dest)) {
            try {
                Files.createLink(dest, blob);
            } catch (UnsupportedOperationException | IOException e) {
                Files.copy(blob, dest, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        return dest;
    }

    public boolean hasSnapshot(String repoType, String repoId, String revision) {
        try {
            Path snap = snapshotPath(repoType, repoId, revision);
            return Files.isDirectory(snap) && Files.list(snap).findAny().isPresent();
        } catch (IOException e) {
            return false;
        }
    }

    public List<String> listCachedRepos() throws IOException {
        if (!Files.isDirectory(root)) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        try (Stream<Path> s = Files.list(root)) {
            s.filter(Files::isDirectory)
                    .map(p -> p.getFileName().toString())
                    .forEach(out::add);
        }
        Collections.sort(out);
        return out;
    }

    /** Approximate disk usage in bytes of a repo cache tree. */
    public long diskUsage(String repoType, String repoId) throws IOException {
        Path dir = repoDir(repoType, repoId);
        if (!Files.isDirectory(dir)) {
            return 0L;
        }
        long[] total = {0L};
        try (Stream<Path> walk = Files.walk(dir)) {
            walk.filter(Files::isRegularFile).forEach(p -> {
                try {
                    total[0] += Files.size(p);
                } catch (IOException ignored) {
                }
            });
        }
        return total[0];
    }

    public Map<String, Object> info(String repoType, String repoId) throws IOException {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("repoType", repoType);
        m.put("repoId", repoId);
        m.put("path", repoDir(repoType, repoId).toString());
        m.put("diskUsage", diskUsage(repoType, repoId));
        m.put("cached", hasSnapshot(repoType, repoId, "main"));
        return m;
    }

    public static String sha256(byte[] data) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            return HexFormat.of().formatHex(md.digest(data));
        } catch (Exception e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    public static String sha256(String text) {
        return sha256(text.getBytes(StandardCharsets.UTF_8));
    }
}
