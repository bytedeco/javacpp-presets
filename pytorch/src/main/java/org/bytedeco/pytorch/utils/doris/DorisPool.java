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
package org.bytedeco.pytorch.utils.doris;

import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.ArrayDeque;
import java.util.Objects;
import java.util.Properties;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

/**
 * Lightweight JDBC connection pool for Doris FE MySQL protocol (high-QPS point query / scan).
 *
 * <p>Pattern mirrors {@code DuckDBPool}: borrow / release / withConnection, no external pool dep.</p>
 */
public final class DorisPool implements AutoCloseable {

    private final DorisOptions options;
    private final int maxSize;
    private final long borrowTimeoutMs;
    private final ArrayDeque<Connection> idle = new ArrayDeque<>();
    private final AtomicInteger created = new AtomicInteger(0);
    private final AtomicInteger borrowed = new AtomicInteger(0);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Object lock = new Object();

    private DorisPool(DorisOptions options, int maxSize, long borrowTimeoutMs) {
        this.options = Objects.requireNonNull(options, "options");
        this.maxSize = Math.max(1, maxSize);
        this.borrowTimeoutMs = Math.max(0L, borrowTimeoutMs);
    }

    public static DorisPool open(DorisOptions options) {
        Objects.requireNonNull(options, "options");
        return new DorisPool(options, options.poolSize(), options.poolBorrowTimeoutMs());
    }

    public static DorisPool open(DorisOptions options, int maxSize) {
        Objects.requireNonNull(options, "options");
        return new DorisPool(options, maxSize, options.poolBorrowTimeoutMs());
    }

    public DorisOptions options() {
        return options;
    }

    public int maxSize() {
        return maxSize;
    }

    public int created() {
        return created.get();
    }

    public int borrowed() {
        return borrowed.get();
    }

    public int idleCount() {
        synchronized (lock) {
            return idle.size();
        }
    }

    public Connection borrow() throws SQLException, InterruptedException {
        ensureOpen();
        long deadline = borrowTimeoutMs == 0
                ? Long.MAX_VALUE
                : System.currentTimeMillis() + borrowTimeoutMs;
        synchronized (lock) {
            while (true) {
                ensureOpen();
                while (!idle.isEmpty()) {
                    Connection c = idle.pollFirst();
                    if (isValid(c)) {
                        borrowed.incrementAndGet();
                        return c;
                    }
                    closeQuietly(c);
                    created.decrementAndGet();
                }
                if (created.get() < maxSize) {
                    Connection c = newConnection();
                    created.incrementAndGet();
                    borrowed.incrementAndGet();
                    return c;
                }
                long wait = deadline - System.currentTimeMillis();
                if (wait <= 0) {
                    throw new SQLException("DorisPool borrow timeout after " + borrowTimeoutMs + "ms");
                }
                lock.wait(Math.min(wait, 1000L));
            }
        }
    }

    public void release(Connection c) {
        if (c == null) return;
        synchronized (lock) {
            borrowed.decrementAndGet();
            if (closed.get() || !isValid(c)) {
                closeQuietly(c);
                created.decrementAndGet();
                lock.notifyAll();
                return;
            }
            idle.offerLast(c);
            lock.notifyAll();
        }
    }

    public <T> T withConnection(Function<Connection, T> fn) {
        Objects.requireNonNull(fn, "fn");
        Connection c;
        try {
            c = borrow();
        } catch (Exception e) {
            throw new LakeException(LakeFormat.DORIS, "pool.borrow", e.getMessage(), e);
        }
        try {
            return fn.apply(c);
        } finally {
            release(c);
        }
    }

    public void execute(ConnectionConsumer consumer) {
        Objects.requireNonNull(consumer, "consumer");
        withConnection(c -> {
            try {
                consumer.accept(c);
                return null;
            } catch (SQLException e) {
                throw new LakeException(LakeFormat.DORIS, "pool.execute", e.getMessage(), e);
            }
        });
    }

    @FunctionalInterface
    public interface ConnectionConsumer {
        void accept(Connection c) throws SQLException;
    }

    private Connection newConnection() throws SQLException {
        ensureDriver();
        Properties props = new Properties();
        props.setProperty("user", options.username());
        props.setProperty("password", options.password() == null ? "" : options.password());
        if (options.connectTimeoutMs() > 0) {
            props.setProperty("connectTimeout", Integer.toString(options.connectTimeoutMs()));
        }
        if (options.socketTimeoutMs() > 0) {
            props.setProperty("socketTimeout", Integer.toString(options.socketTimeoutMs()));
        }
        Connection c = DriverManager.getConnection(options.jdbcUrl(), props);
        c.setAutoCommit(true);
        return c;
    }

    private static void ensureDriver() {
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
        } catch (ClassNotFoundException e) {
            // DriverManager may still locate via SPI
        }
    }

    private static boolean isValid(Connection c) {
        try {
            return c != null && !c.isClosed() && c.isValid(2);
        } catch (SQLException e) {
            return false;
        }
    }

    private static void closeQuietly(Connection c) {
        if (c == null) return;
        try { c.close(); } catch (SQLException ignored) {}
    }

    private void ensureOpen() throws SQLException {
        if (closed.get()) throw new SQLException("DorisPool is closed");
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        synchronized (lock) {
            Connection c;
            while ((c = idle.pollFirst()) != null) {
                closeQuietly(c);
                created.decrementAndGet();
            }
            lock.notifyAll();
        }
    }
}
