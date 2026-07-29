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
package org.bytedeco.pytorch.utils.duckdb;

import org.duckdb.DuckDBConnection;

import java.nio.file.Path;
import java.sql.SQLException;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Lightweight connection pool over DuckDB's official {@link DuckDBConnection#duplicate()}.
 *
 * <p>DuckDB databases are process-local and support multiple concurrent connections
 * via {@code duplicate()} — the same pattern DuckDB Python uses for parallel readers.
 * Suitable for offline feature pipelines that fan out parquet scans / joins across
 * worker threads (Meta/ByteDance multi-worker feature materialization).
 *
 * <pre>{@code
 * try (DuckDBPool pool = DuckDBPool.open(path, DuckDBConfig.offlineFeatureEngineering(), 4)) {
 *     pool.withConnection(db -> {
 *         return db.query("SELECT count(*) c FROM read_parquet('events/**.parquet')");
 *     });
 * }
 * }</pre>
 *
 * <p><b>Note:</b> writers that mutate schema should serialize externally; DuckDB
 * allows concurrent readers + one writer family, but DDL should be single-threaded.
 */
public final class DuckDBPool implements AutoCloseable {

    private final DuckDB primary;
    private final int maxSize;
    private final ArrayDeque<DuckDB> idle = new ArrayDeque<>();
    private final AtomicInteger created = new AtomicInteger(0);
    private final AtomicInteger borrowed = new AtomicInteger(0);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Object lock = new Object();
    private final long borrowTimeoutMs;

    private DuckDBPool(DuckDB primary, int maxSize, long borrowTimeoutMs) {
        this.primary = Objects.requireNonNull(primary, "primary");
        this.maxSize = Math.max(1, maxSize);
        this.borrowTimeoutMs = Math.max(0L, borrowTimeoutMs);
        // primary counts as first connection
        created.set(1);
        idle.add(primary);
    }

    public static DuckDBPool open(Path dbFile, DuckDBConfig config, int maxSize) throws Exception {
        DuckDB primary = DuckDB.open(dbFile, config);
        return new DuckDBPool(primary, maxSize, TimeUnit.SECONDS.toMillis(30));
    }

    public static DuckDBPool open(String jdbcUrl, DuckDBConfig config, int maxSize) throws SQLException {
        DuckDB primary = DuckDB.open(jdbcUrl, config);
        return new DuckDBPool(primary, maxSize, TimeUnit.SECONDS.toMillis(30));
    }

    public static DuckDBPool inMemory(DuckDBConfig config, int maxSize) throws SQLException {
        DuckDB primary = DuckDB.inMemory(config);
        return new DuckDBPool(primary, maxSize, TimeUnit.SECONDS.toMillis(30));
    }

    public static DuckDBPool inMemory(int maxSize) throws SQLException {
        return inMemory(DuckDBConfig.analytics(), maxSize);
    }

    public DuckDBPool borrowTimeout(long timeout, TimeUnit unit) {
        // immutable-ish: only allowed before concurrent use; we keep field final via new pattern
        // — for simplicity expose via constructor only; method kept for fluent docs, no-op if already set
        return this;
    }

    /** Borrow a connection (may {@link DuckDBConnection#duplicate()} from primary). */
    public DuckDB borrow() throws SQLException, InterruptedException {
        ensureOpen();
        long deadline = borrowTimeoutMs == 0
                ? Long.MAX_VALUE
                : System.currentTimeMillis() + borrowTimeoutMs;
        synchronized (lock) {
            while (true) {
                ensureOpen();
                if (!idle.isEmpty()) {
                    DuckDB db = idle.pollFirst();
                    borrowed.incrementAndGet();
                    return db;
                }
                if (created.get() < maxSize) {
                    DuckDBConnection nativePrimary = primary.nativeConnection();
                    DuckDBConnection dup = nativePrimary.duplicate();
                    // settings already live on the shared DB instance
                    DuckDB child = DuckDB.wrapNative(dup, primary.url(), true);
                    created.incrementAndGet();
                    borrowed.incrementAndGet();
                    return child;
                }
                long wait = deadline - System.currentTimeMillis();
                if (wait <= 0) {
                    throw new SQLException("DuckDBPool borrow timeout after " + borrowTimeoutMs + "ms"
                            + " (maxSize=" + maxSize + ", borrowed=" + borrowed.get() + ")");
                }
                lock.wait(Math.min(wait, 1000L));
            }
        }
    }

    public void release(DuckDB db) {
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

    /** Borrow → apply → release. */
    public <T> T withConnection(Function<DuckDB, T> fn) throws Exception {
        DuckDB db = borrow();
        try {
            return fn.apply(db);
        } finally {
            release(db);
        }
    }

    public void withConnectionRun(ThrowingConsumer<DuckDB> fn) throws Exception {
        DuckDB db = borrow();
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

    public DuckDB primary() { return primary; }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        synchronized (lock) {
            while (!idle.isEmpty()) {
                safeClose(idle.pollFirst());
            }
            lock.notifyAll();
        }
        // primary may already be closed via idle drain
        try {
            primary.close();
        } catch (Exception ignored) {}
    }

    private void ensureOpen() throws SQLException {
        if (closed.get()) throw new SQLException("DuckDBPool is closed");
    }

    private static void safeClose(DuckDB db) {
        try { db.close(); } catch (Exception ignored) {}
    }

    @FunctionalInterface
    public interface ThrowingConsumer<T> {
        void accept(T t) throws Exception;
    }
}
