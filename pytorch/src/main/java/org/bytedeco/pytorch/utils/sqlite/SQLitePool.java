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

import java.nio.file.Path;
import java.sql.SQLException;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Small connection pool for SQLite WAL databases.
 *
 * <p>SQLite allows multiple concurrent readers + one writer under WAL. This pool
 * opens independent connections to the same file (not shared Connection objects)
 * — the correct model for multi-threaded feature-cache lookups on the serving
 * path (Meta/ByteDance process-local feature mirrors, Apple on-device stores).
 *
 * <pre>{@code
 * try (SQLitePool pool = SQLitePool.open(path, SQLiteConfig.onlineFeatureCache(), 8)) {
 *     float ctr = pool.withConnection(db -> {
 *         var df = db.query("SELECT ctr FROM user_feat WHERE user_id=?", userId);
 *         return df.rowCount() == 0 ? 0f : ((Number) df.get(0, "ctr")).floatValue();
 *     });
 * }
 * }</pre>
 */
public final class SQLitePool implements AutoCloseable {

    private final String jdbcUrl;
    private final SQLiteConfig config;
    private final int maxSize;
    private final long borrowTimeoutMs;
    private final ArrayDeque<SQLite> idle = new ArrayDeque<>();
    private final AtomicInteger created = new AtomicInteger(0);
    private final AtomicInteger borrowed = new AtomicInteger(0);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Object lock = new Object();

    private SQLitePool(String jdbcUrl, SQLiteConfig config, int maxSize, long borrowTimeoutMs)
            throws SQLException {
        this.jdbcUrl = Objects.requireNonNull(jdbcUrl, "jdbcUrl");
        this.config = config == null ? SQLiteConfig.onlineFeatureCache() : config;
        this.maxSize = Math.max(1, maxSize);
        this.borrowTimeoutMs = Math.max(0L, borrowTimeoutMs);
        // prime one connection
        idle.add(SQLite.open(jdbcUrl, this.config));
        created.set(1);
    }

    public static SQLitePool open(Path dbFile, SQLiteConfig config, int maxSize) throws Exception {
        if (dbFile.getParent() != null) {
            java.nio.file.Files.createDirectories(dbFile.getParent());
        }
        String url = SQLite.URL_PREFIX + dbFile.toAbsolutePath();
        return new SQLitePool(url, config, maxSize, TimeUnit.SECONDS.toMillis(30));
    }

    public static SQLitePool open(String jdbcUrl, SQLiteConfig config, int maxSize)
            throws SQLException {
        return new SQLitePool(jdbcUrl, config, maxSize, TimeUnit.SECONDS.toMillis(30));
    }

    public static SQLitePool sharedMemory(String name, int maxSize) throws SQLException {
        String n = (name == null || name.isEmpty()) ? "pool" : name;
        String url = "jdbc:sqlite:file:" + n + "?mode=memory&cache=shared";
        return new SQLitePool(url, SQLiteConfig.inMemory().sharedCache(true), maxSize,
                TimeUnit.SECONDS.toMillis(10));
    }

    public SQLite borrow() throws SQLException, InterruptedException {
        ensureOpen();
        long deadline = borrowTimeoutMs == 0
                ? Long.MAX_VALUE
                : System.currentTimeMillis() + borrowTimeoutMs;
        synchronized (lock) {
            while (true) {
                ensureOpen();
                if (!idle.isEmpty()) {
                    SQLite db = idle.pollFirst();
                    borrowed.incrementAndGet();
                    return db;
                }
                if (created.get() < maxSize) {
                    SQLite db = SQLite.open(jdbcUrl, config);
                    created.incrementAndGet();
                    borrowed.incrementAndGet();
                    return db;
                }
                long wait = deadline - System.currentTimeMillis();
                if (wait <= 0) {
                    throw new SQLException("SQLitePool borrow timeout after " + borrowTimeoutMs
                            + "ms (maxSize=" + maxSize + ", borrowed=" + borrowed.get() + ")");
                }
                lock.wait(Math.min(wait, 1000L));
            }
        }
    }

    public void release(SQLite db) {
        if (db == null) return;
        synchronized (lock) {
            if (closed.get()) {
                safeClose(db);
                return;
            }
            idle.addLast(db);
            borrowed.decrementAndGet();
            lock.notifyAll();
        }
    }

    public <T> T withConnection(Function<SQLite, T> fn) throws Exception {
        SQLite db = borrow();
        try {
            return fn.apply(db);
        } finally {
            release(db);
        }
    }

    public void withConnectionRun(ThrowingConsumer<SQLite> fn) throws Exception {
        SQLite db = borrow();
        try {
            fn.accept(db);
        } finally {
            release(db);
        }
    }

    public int maxSize() { return maxSize; }
    public int createdCount() { return created.get(); }
    public int borrowedCount() { return borrowed.get(); }
    public int idleCount() {
        synchronized (lock) { return idle.size(); }
    }

    public String jdbcUrl() { return jdbcUrl; }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        synchronized (lock) {
            while (!idle.isEmpty()) {
                safeClose(idle.pollFirst());
            }
            lock.notifyAll();
        }
    }

    private void ensureOpen() throws SQLException {
        if (closed.get()) throw new SQLException("SQLitePool is closed");
    }

    private static void safeClose(SQLite db) {
        try { db.close(); } catch (Exception ignored) {}
    }

    @FunctionalInterface
    public interface ThrowingConsumer<T> {
        void accept(T t) throws Exception;
    }
}
