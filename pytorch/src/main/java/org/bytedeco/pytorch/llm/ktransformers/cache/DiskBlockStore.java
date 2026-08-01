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
package org.bytedeco.pytorch.llm.ktransformers.cache;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Pure-Java disk block store for L3 prefix / KV payloads.
 *
 * <p>Each logical block is a file {@code block-<id>.bin} under {@code root},
 * or an in-memory map when capacity is small (tests). Layout:
 * <pre>
 *   int32 magic 'KTBLK' (0x4B54424C)
 *   int32 version
 *   int32 nbytes
 *   byte[nbytes] payload
 * </pre>
 */
public final class DiskBlockStore implements AutoCloseable {

    public static final int MAGIC = 0x4B54424C; // 'KTBL'
    public static final int VERSION = 1;

    private final Path root;
    private final int maxBlocks;
    private final ConcurrentHashMap<Long, byte[]> memoryFallback;
    private final ConcurrentHashMap<Long, Path> paths = new ConcurrentHashMap<>();
    private final LinkedHashMap<Long, Boolean> lru;
    private final ReentrantLock lruLock = new ReentrantLock();
    private final AtomicInteger live = new AtomicInteger();
    private final boolean memoryOnly;
    private boolean closed;

    public DiskBlockStore(Path root, int maxBlocks) throws IOException {
        this.maxBlocks = Math.max(1, maxBlocks);
        this.root = root;
        this.memoryOnly = root == null;
        if (!memoryOnly) {
            Files.createDirectories(root);
            this.memoryFallback = new ConcurrentHashMap<>();
        } else {
            this.memoryFallback = new ConcurrentHashMap<>();
        }
        this.lru = new LinkedHashMap<>(16, 0.75f, true);
    }

    /** In-memory only store (unit tests / no disk). */
    public static DiskBlockStore memory(int maxBlocks) {
        try {
            return new DiskBlockStore(null, maxBlocks);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static DiskBlockStore open(Path root, int maxBlocks) throws IOException {
        return new DiskBlockStore(Objects.requireNonNull(root, "root"), maxBlocks);
    }

    public Path root() { return root; }
    public int maxBlocks() { return maxBlocks; }
    public int liveBlocks() { return live.get(); }
    public boolean memoryOnly() { return memoryOnly || root == null; }

    public synchronized boolean contains(long blockId) {
        if (memoryFallback.containsKey(blockId)) return true;
        Path p = paths.get(blockId);
        return p != null && Files.isRegularFile(p);
    }

    public void put(long blockId, byte[] payload) throws IOException {
        Objects.requireNonNull(payload, "payload");
        ensureOpen();
        evictIfNeeded();
        if (memoryOnly()) {
            memoryFallback.put(blockId, payload.clone());
            touch(blockId);
            live.set(memoryFallback.size());
            return;
        }
        Path file = root.resolve("block-" + blockId + ".bin");
        ByteBuffer hdr = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        hdr.putInt(MAGIC).putInt(VERSION).putInt(payload.length);
        hdr.flip();
        try (FileChannel ch = FileChannel.open(file,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE)) {
            ch.write(hdr);
            ch.write(ByteBuffer.wrap(payload));
        }
        paths.put(blockId, file);
        memoryFallback.remove(blockId);
        touch(blockId);
        live.incrementAndGet();
    }

    public byte[] get(long blockId) throws IOException {
        ensureOpen();
        byte[] mem = memoryFallback.get(blockId);
        if (mem != null) {
            touch(blockId);
            return mem.clone();
        }
        Path file = paths.get(blockId);
        if (file == null) {
            file = memoryOnly() ? null : root.resolve("block-" + blockId + ".bin");
            if (file == null || !Files.isRegularFile(file)) {
                return null;
            }
            paths.put(blockId, file);
        }
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer hdr = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
            if (ch.read(hdr) < 12) return null;
            hdr.flip();
            int magic = hdr.getInt();
            int ver = hdr.getInt();
            int n = hdr.getInt();
            if (magic != MAGIC || ver != VERSION || n < 0 || n > (1 << 28)) {
                return null;
            }
            ByteBuffer body = ByteBuffer.allocate(n);
            while (body.hasRemaining()) {
                if (ch.read(body) < 0) break;
            }
            touch(blockId);
            return body.array();
        }
    }

    public boolean remove(long blockId) throws IOException {
        memoryFallback.remove(blockId);
        Path p = paths.remove(blockId);
        lruLock.lock();
        try {
            lru.remove(blockId);
        } finally {
            lruLock.unlock();
        }
        if (p != null && Files.isRegularFile(p)) {
            Files.deleteIfExists(p);
            live.decrementAndGet();
            return true;
        }
        live.set(Math.max(0, memoryFallback.size() + paths.size()));
        return false;
    }

    private void touch(long blockId) {
        lruLock.lock();
        try {
            lru.put(blockId, Boolean.TRUE);
        } finally {
            lruLock.unlock();
        }
    }

    private void evictIfNeeded() throws IOException {
        while (live.get() >= maxBlocks) {
            Long victim;
            lruLock.lock();
            try {
                if (lru.isEmpty()) break;
                Map.Entry<Long, Boolean> e = lru.entrySet().iterator().next();
                victim = e.getKey();
                lru.remove(victim);
            } finally {
                lruLock.unlock();
            }
            if (victim != null) {
                remove(victim);
            } else {
                break;
            }
        }
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("DiskBlockStore closed");
    }

    @Override
    public void close() {
        closed = true;
        memoryFallback.clear();
        paths.clear();
        lruLock.lock();
        try {
            lru.clear();
        } finally {
            lruLock.unlock();
        }
    }
}
