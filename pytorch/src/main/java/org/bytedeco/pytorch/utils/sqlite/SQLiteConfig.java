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
package org.bytedeco.pytorch.utils.sqlite;

import org.sqlite.SQLiteConfig.JournalMode;
import org.sqlite.SQLiteConfig.LockingMode;
import org.sqlite.SQLiteConfig.SynchronousMode;
import org.sqlite.SQLiteConfig.TempStore;
import org.sqlite.SQLiteOpenMode;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**
 * Enterprise configuration for official {@code org.xerial:sqlite-jdbc}.
 *
 * <p>Wraps {@link org.sqlite.SQLiteConfig} with recsys / multimodal / edge presets:
 * <ul>
 *   <li>{@link #onlineFeatureCache()} — Meta/ByteDance nearline feature cache
 *       (WAL, NORMAL sync, mmap, busy timeout)</li>
 *   <li>{@link #embeddingStore()} — local embedding / ANN sidecar (WAL, large
 *       page/cache, mmap — Apple Core ML / on-device style)</li>
 *   <li>{@link #readOnlyServing()} — immutable snapshot readers</li>
 *   <li>{@link #bulkLoad()} — OFF journal + OFF sync for one-shot ETL (danger:
 *       not crash-safe; re-enable WAL after load)</li>
 *   <li>{@link #analytics()} — general local analytics</li>
 * </ul>
 *
 * <pre>{@code
 * SQLiteConfig cfg = SQLiteConfig.onlineFeatureCache()
 *     .cacheSize(-(64 * 1024))  // 64MB (negative = KiB)
 *     .mmapSize(256L * 1024 * 1024)
 *     .busyTimeoutMs(5_000);
 * try (SQLite db = SQLite.open(path, cfg)) { ... }
 * }</pre>
 */
public final class SQLiteConfig {

    private final org.sqlite.SQLiteConfig nativeConfig;
    private final Map<String, String> extraPragmas = new LinkedHashMap<>();
    private int busyTimeoutMs = 5_000;
    private long mmapSize = -1; // -1 = leave default
    private Integer cacheSize;  // null = leave default; negative => KiB
    private Integer pageSize;
    private boolean foreignKeys = true;
    private boolean recursiveTriggers;
    private String tempStoreDirectory;

    private SQLiteConfig() {
        this.nativeConfig = new org.sqlite.SQLiteConfig();
    }

    public static SQLiteConfig create() {
        return new SQLiteConfig();
    }

    /**
     * Online / nearline feature cache (serving sidecars, edge rankers).
     * WAL + NORMAL synchronous is the industry default for concurrent readers
     * with occasional writers (SQLite docs; Mobile/desktop feature stores).
     */
    public static SQLiteConfig onlineFeatureCache() {
        return create()
                .journalMode(JournalMode.WAL)
                .synchronous(SynchronousMode.NORMAL)
                .lockingMode(LockingMode.NORMAL)
                .foreignKeys(true)
                .busyTimeoutMs(5_000)
                .cacheSize(-(64 * 1024))       // 64 MiB
                .mmapSize(256L * 1024 * 1024)  // 256 MiB
                .tempStore(TempStore.MEMORY);
    }

    /** Local embedding / vector KV store for on-device or process-local ANN. */
    public static SQLiteConfig embeddingStore() {
        return create()
                .journalMode(JournalMode.WAL)
                .synchronous(SynchronousMode.NORMAL)
                .busyTimeoutMs(10_000)
                .cacheSize(-(128 * 1024))
                .mmapSize(512L * 1024 * 1024)
                .pageSize(8192)
                .tempStore(TempStore.MEMORY);
    }

    /** Read-only open of a published snapshot. */
    public static SQLiteConfig readOnlyServing() {
        SQLiteConfig c = create()
                .journalMode(JournalMode.WAL)
                .synchronous(SynchronousMode.NORMAL)
                .busyTimeoutMs(3_000)
                .cacheSize(-(32 * 1024))
                .mmapSize(128L * 1024 * 1024);
        c.nativeConfig.setReadOnly(true);
        c.nativeConfig.setOpenMode(SQLiteOpenMode.READONLY);
        return c;
    }

    /**
     * Bulk load: disable journal/sync for speed. <b>Not crash-safe.</b>
     * Call {@link SQLite#enableWalSafe()} after load completes.
     */
    public static SQLiteConfig bulkLoad() {
        return create()
                .journalMode(JournalMode.OFF)
                .synchronous(SynchronousMode.OFF)
                .lockingMode(LockingMode.EXCLUSIVE)
                .busyTimeoutMs(60_000)
                .cacheSize(-(256 * 1024))
                .tempStore(TempStore.MEMORY)
                .foreignKeys(false);
    }

    public static SQLiteConfig analytics() {
        return create()
                .journalMode(JournalMode.WAL)
                .synchronous(SynchronousMode.NORMAL)
                .busyTimeoutMs(5_000)
                .cacheSize(-(64 * 1024))
                .mmapSize(256L * 1024 * 1024)
                .tempStore(TempStore.MEMORY);
    }

    public static SQLiteConfig inMemory() {
        return create()
                .journalMode(JournalMode.MEMORY)
                .synchronous(SynchronousMode.OFF)
                .tempStore(TempStore.MEMORY)
                .foreignKeys(true)
                .busyTimeoutMs(1_000);
    }

    // ---- builders ----------------------------------------------------------

    public SQLiteConfig journalMode(JournalMode mode) {
        if (mode != null) nativeConfig.setJournalMode(mode);
        return this;
    }

    public SQLiteConfig synchronous(SynchronousMode mode) {
        if (mode != null) nativeConfig.setSynchronous(mode);
        return this;
    }

    public SQLiteConfig lockingMode(LockingMode mode) {
        if (mode != null) nativeConfig.setLockingMode(mode);
        return this;
    }

    public SQLiteConfig tempStore(TempStore store) {
        if (store != null) nativeConfig.setTempStore(store);
        return this;
    }

    public SQLiteConfig openMode(SQLiteOpenMode mode) {
        if (mode != null) nativeConfig.setOpenMode(mode);
        return this;
    }

    public SQLiteConfig readOnly(boolean v) {
        nativeConfig.setReadOnly(v);
        if (v) nativeConfig.setOpenMode(SQLiteOpenMode.READONLY);
        return this;
    }

    public SQLiteConfig sharedCache(boolean v) {
        nativeConfig.setSharedCache(v);
        return this;
    }

    public SQLiteConfig loadExtension(boolean v) {
        nativeConfig.enableLoadExtension(v);
        return this;
    }

    public SQLiteConfig foreignKeys(boolean v) {
        this.foreignKeys = v;
        nativeConfig.enforceForeignKeys(v);
        return this;
    }

    public SQLiteConfig recursiveTriggers(boolean v) {
        this.recursiveTriggers = v;
        nativeConfig.enableRecursiveTriggers(v);
        return this;
    }

    public SQLiteConfig busyTimeoutMs(int ms) {
        this.busyTimeoutMs = Math.max(0, ms);
        nativeConfig.setBusyTimeout(this.busyTimeoutMs);
        return this;
    }

    /**
     * Cache size pragma. Negative values are in KiB (SQLite convention),
     * positive values are in pages.
     */
    public SQLiteConfig cacheSize(int pagesOrNegKiB) {
        this.cacheSize = pagesOrNegKiB;
        nativeConfig.setCacheSize(pagesOrNegKiB);
        return this;
    }

    public SQLiteConfig pageSize(int bytes) {
        this.pageSize = bytes;
        if (bytes > 0) nativeConfig.setPageSize(bytes);
        return this;
    }

    /** mmap_size in bytes (0 disables). Applied as extra PRAGMA after connect. */
    public SQLiteConfig mmapSize(long bytes) {
        this.mmapSize = Math.max(0L, bytes);
        return this;
    }

    public SQLiteConfig tempStoreDirectory(String path) {
        this.tempStoreDirectory = path;
        if (path != null) nativeConfig.setTempStoreDirectory(path);
        return this;
    }

    public SQLiteConfig userVersion(int v) {
        nativeConfig.setUserVersion(v);
        return this;
    }

    public SQLiteConfig applicationId(int v) {
        nativeConfig.setApplicationId(v);
        return this;
    }

    /** Extra PRAGMA name/value applied after {@link org.sqlite.SQLiteConfig#apply}. */
    public SQLiteConfig pragma(String name, String value) {
        Objects.requireNonNull(name, "name");
        if (value == null) extraPragmas.remove(name);
        else extraPragmas.put(name, value);
        return this;
    }

    // ---- materialize -------------------------------------------------------

    public org.sqlite.SQLiteConfig nativeConfig() {
        return nativeConfig;
    }

    public Properties toProperties() {
        return nativeConfig.toProperties();
    }

    public int busyTimeoutMs() { return busyTimeoutMs; }
    public long mmapSize() { return mmapSize; }
    public Integer cacheSize() { return cacheSize; }
    public Integer pageSize() { return pageSize; }
    public boolean foreignKeys() { return foreignKeys; }
    public boolean recursiveTriggers() { return recursiveTriggers; }

    public Map<String, String> extraPragmasView() {
        return Map.copyOf(extraPragmas);
    }

    @Override
    public String toString() {
        return "SQLiteConfig{busyTimeoutMs=" + busyTimeoutMs
                + ", mmapSize=" + mmapSize
                + ", cacheSize=" + cacheSize
                + ", pageSize=" + pageSize
                + ", extra=" + extraPragmas + "}";
    }
}
