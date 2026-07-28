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
package org.bytedeco.pytorch.distributed;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.chrono.Milliseconds;
import org.bytedeco.pytorch.ByteVector;
import org.bytedeco.pytorch.StringVector;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Rank-prefixed key/value store used for process-group rendezvous.
 *
 * <p>Supports {@link FileStore} (single-machine multi-process), {@link TCPStore}
 * (multi-machine), and {@link HashStore} (single-process in-memory — default when
 * {@code worldSize==1} so smoke tests never block on FileStore rendezvous).
 *
 * <p><b>Hang prevention:</b> FileStore paths are unique per process group
 * (UUID / {@code ACCELERATE_FILE_STORE} env / Options.fileStorePath). A fixed
 * shared path like {@code /tmp/pytorch_ddp_store_$USER} deadlocks when leftover
 * ranks or concurrent runs share the same file.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DistributedStore implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final Set<DistributedStore> INSTANCES =
            ConcurrentHashMap.newKeySet();

    /** Optional override from MultiProcessLauncher ({@code ACCELERATE_FILE_STORE}). */
    public static final String ENV_FILE_STORE = "ACCELERATE_FILE_STORE";

    private final Options options;
    private final int rank;
    private final int worldSize;
    private final String prefix;
    private final StoreType selectedType;
    private final Store store;
    private final Path fileStorePath; // non-null only for FILE
    private final boolean ownsFileStorePath;

    public DistributedStore(Options options, int rank, int worldSize) {
        this.options = Objects.requireNonNull(options, "options");
        this.rank = rank;
        this.worldSize = worldSize;
        this.prefix = "_rank_" + rank + "_";
        this.selectedType = resolveType(options, worldSize);
        Path fsp = null;
        boolean owns = false;
        if (selectedType == StoreType.TCP) {
            this.store = createTcpStore(options, worldSize, rank);
            this.fileStorePath = null;
            this.ownsFileStorePath = false;
        } else if (selectedType == StoreType.HASH) {
            this.store = new HashStore();
            this.fileStorePath = null;
            this.ownsFileStorePath = false;
        } else {
            AtomicReference<Boolean> ownsRef = new AtomicReference<>(false);
            fsp = resolveFileStorePath(options, ownsRef);
            owns = Boolean.TRUE.equals(ownsRef.get());
            this.store = new FileStore(fsp.toString(), worldSize);
            this.fileStorePath = fsp;
            this.ownsFileStorePath = owns;
        }
        INSTANCES.add(this);
        System.out.printf(
                "[DistributedStore] type=%s rank=%d worldSize=%d path=%s%n",
                selectedType, rank, worldSize,
                fileStorePath != null ? fileStorePath : "(in-memory/tcp)");
    }

    public static DistributedStore create(int rank, int worldSize) {
        return create(new Options(), rank, worldSize);
    }

    public static DistributedStore create(Options options, int rank, int worldSize) {
        return new DistributedStore(options, rank, worldSize);
    }

    /** Single-process smoke helper: always HashStore, never FileStore. */
    public static DistributedStore createSingleProcess() {
        return create(new Options().type(StoreType.HASH), 0, 1);
    }

    public void set(String key, String value) {
        store.set(prefix + key, value);
    }

    public void set(String key, byte[] value) {
        store.set(prefix + key, new ByteVector(value));
    }

    public String getString(String key) {
        try {
            ByteVector bv = store.get(prefix + key);
            if (bv == null || bv.empty()) {
                return null;
            }
            return new String(bv.get(), StandardCharsets.UTF_8);
        } catch (RuntimeException e) {
            return null;
        }
    }

    public byte[] getBytes(String key) {
        try {
            ByteVector bv = store.get(prefix + key);
            if (bv == null || bv.empty()) {
                return null;
            }
            return bv.get();
        } catch (RuntimeException e) {
            return null;
        }
    }

    public long add(String key, long value) {
        return store.add(prefix + key, value);
    }

    public int getInteger(String key) {
        return (int) store.add(prefix + key, 0L);
    }

    public void delete(String key) {
        store.deleteKey(prefix + key);
    }

    public boolean exists(String key) {
        return store.check(new StringVector(prefix + key));
    }

    public void waitFor(String key) {
        store._wait(new StringVector(prefix + key));
    }

    public void waitFor(String key, int timeoutMs) {
        store._wait(new StringVector(prefix + key), new Milliseconds(timeoutMs));
    }

    public Store getNativeStore() { return store; }
    public int getRank() { return rank; }
    public int getWorldSize() { return worldSize; }
    public StoreType getType() { return selectedType; }
    public Path getFileStorePath() { return fileStorePath; }

    @Override
    public void close() {
        INSTANCES.remove(this);
        // Best-effort cleanup of unique FileStore directories we created.
        if (ownsFileStorePath && fileStorePath != null && rank == 0) {
            try {
                Files.deleteIfExists(fileStorePath);
                Path parent = fileStorePath.getParent();
                if (parent != null && parent.getFileName() != null
                        && parent.getFileName().toString().startsWith("pytorch_ddp_")) {
                    // only remove empty UUID dirs we own
                    try { Files.deleteIfExists(parent); } catch (Throwable ignored) {}
                }
            } catch (Throwable ignored) {}
        }
    }

    @Override
    public String toString() {
        return "DistributedStore{type=" + selectedType
                + ", rank=" + rank + ", worldSize=" + worldSize
                + (fileStorePath != null ? ", path=" + fileStorePath : "") + '}';
    }

    /**
     * AUTO policy:
     * <ul>
     *   <li>worldSize==1 → HASH (never block on FileStore rendezvous)</li>
     *   <li>non-local masterAddr → TCP</li>
     *   <li>else → FILE with unique path</li>
     * </ul>
     */
    private static StoreType resolveType(Options opts, int worldSize) {
        if (opts.getType() != StoreType.AUTO) {
            return opts.getType();
        }
        if (worldSize <= 1) {
            return StoreType.HASH;
        }
        String addr = opts.getMasterAddr();
        boolean multi = addr != null
                && !"127.0.0.1".equals(addr)
                && !"localhost".equalsIgnoreCase(addr);
        return multi ? StoreType.TCP : StoreType.FILE;
    }

    private static Path resolveFileStorePath(Options opts, AtomicReference<Boolean> ownsOut) {
        // 1) Explicit option
        if (opts.getFileStorePath() != null && !opts.getFileStorePath().isBlank()) {
            ownsOut.set(false);
            return asFileStoreFile(Path.of(opts.getFileStorePath()));
        }
        // 2) MultiProcessLauncher env (shared across children) — may be a directory
        String env = System.getenv(ENV_FILE_STORE);
        if (env == null || env.isBlank()) {
            env = System.getProperty("pytorch.filestore", "");
        }
        if (env != null && !env.isBlank()) {
            ownsOut.set(false);
            return asFileStoreFile(Path.of(env));
        }
        // 3) Unique path per process-group creation — prevents cross-run deadlock
        try {
            Path dir = Files.createTempDirectory("pytorch_ddp_");
            Path file = dir.resolve("store");
            ownsOut.set(true);
            return file;
        } catch (Exception e) {
            ownsOut.set(false);
            return Path.of("/tmp/pytorch_ddp_store_"
                    + System.getProperty("user.name", "user")
                    + "_" + UUID.randomUUID());
        }
    }

    /**
     * c10d {@link FileStore} requires a <em>file</em> path. MultiProcessLauncher
     * historically exports a directory via {@code ACCELERATE_FILE_STORE}; append
     * {@code /store} so all ranks share one file under that directory.
     */
    private static Path asFileStoreFile(Path path) {
        try {
            if (Files.isDirectory(path) || (!Files.exists(path) && !path.getFileName().toString().contains("."))) {
                // Treat bare dir / path-without-extension as directory root
                if (!Files.exists(path)) {
                    Files.createDirectories(path);
                }
                if (Files.isDirectory(path)) {
                    return path.resolve("store");
                }
            }
        } catch (Exception ignored) {
            // fall through
        }
        return path;
    }

    private static Store createTcpStore(Options opts, int worldSize, int rank) {
        TCPStoreOptions tcpOpts = new TCPStoreOptions();
        tcpOpts.port((short) opts.getMasterPort());
        // Rank 0 is the TCPStore server for multi-process jobs.
        tcpOpts.isServer(rank == 0);
        SizeTPointer workers = tcpOpts.numWorkers();
        if (workers != null) {
            workers.put(Math.max(1, opts.getNumWorkers() > 0 ? opts.getNumWorkers() : worldSize));
        }
        tcpOpts.timeout(new Milliseconds(opts.getTimeoutMs()));
        return new TCPStore(opts.getMasterAddr(), tcpOpts);
    }

    /** Builder-style options for {@link DistributedStore}. */
    public static final class Options {
        private StoreType storeType = StoreType.AUTO;
        private int timeoutMs = 30_000; // tighter default — was 300s and masked hangs
        private String masterAddr = "127.0.0.1";
        private int masterPort = 29_500;
        private int numWorkers = 1;
        private String fileStorePath; // optional shared path for multi-proc

        public Options type(StoreType t) { this.storeType = t; return this; }
        public Options timeout(int ms) { this.timeoutMs = ms; return this; }
        public Options masterAddr(String addr) { this.masterAddr = addr; return this; }
        public Options masterPort(int port) { this.masterPort = port; return this; }
        public Options numWorkers(int n) { this.numWorkers = n; return this; }
        public Options fileStorePath(String path) { this.fileStorePath = path; return this; }

        public StoreType getType() { return storeType; }
        public int getTimeoutMs() { return timeoutMs; }
        public String getMasterAddr() { return masterAddr; }
        public int getMasterPort() { return masterPort; }
        public int getNumWorkers() { return numWorkers; }
        public String getFileStorePath() { return fileStorePath; }
    }
}
