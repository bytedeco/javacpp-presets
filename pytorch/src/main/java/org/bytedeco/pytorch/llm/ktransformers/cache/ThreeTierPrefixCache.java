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

import org.bytedeco.pytorch.llm.ktransformers.config.KtCacheConfig;
import org.bytedeco.pytorch.llm.ktransformers.util.DeviceBudget;
import org.bytedeco.pytorch.llm.kvcache.BlockHashIndex;
import org.bytedeco.pytorch.llm.kvcache.PrefixRadixCache;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Three-tier (GPU–CPU–Disk) content-addressed prefix cache.
 *
 * <p>Aligns with upstream "3-layer prefix cache reuse". Control plane:
 * <ul>
 *   <li>L1 GPU — hot in-heap map (capacity {@code gpuBlocks})</li>
 *   <li>L2 CPU — cold in-heap map (capacity {@code cpuBlocks})</li>
 *   <li>L3 Disk — {@link DiskBlockStore}</li>
 * </ul>
 * Lookup promotes Disk→CPU→GPU; pressure demotes GPU→CPU→Disk.
 * Token-prefix matching uses the same rolling hash as {@link BlockHashIndex}.
 */
public final class ThreeTierPrefixCache implements AutoCloseable {

    private final int blockSize;
    private final int gpuCap;
    private final int cpuCap;
    private final LinkedHashMap<Long, byte[]> gpu; // access-order LRU
    private final LinkedHashMap<Long, byte[]> cpu;
    private final DiskBlockStore disk;
    private final PrefixHitStats stats = new PrefixHitStats();
    private final DeviceBudget budget;
    private final ReentrantLock lock = new ReentrantLock();
    private final double gpuWatermark;
    private final double cpuWatermark;
    private boolean closed;

    public ThreeTierPrefixCache(KtCacheConfig cfg, DeviceBudget budget) throws IOException {
        Objects.requireNonNull(cfg, "cfg");
        this.blockSize = cfg.blockSize();
        this.gpuCap = cfg.gpuBlocks();
        this.cpuCap = cfg.cpuBlocks();
        this.gpuWatermark = cfg.gpuWatermark();
        this.cpuWatermark = cfg.cpuWatermark();
        this.budget = budget;
        this.gpu = new LinkedHashMap<>(16, 0.75f, true);
        this.cpu = new LinkedHashMap<>(16, 0.75f, true);
        if (cfg.diskEnabled()) {
            this.disk = DiskBlockStore.open(cfg.diskPath(), Math.max(1, cfg.diskBlocks()));
        } else {
            this.disk = DiskBlockStore.memory(Math.max(1, cfg.diskBlocks() > 0 ? cfg.diskBlocks() : 64));
        }
    }

    public static ThreeTierPrefixCache mini() throws IOException {
        return new ThreeTierPrefixCache(KtCacheConfig.mini(), DeviceBudget.mini());
    }

    public int blockSize() { return blockSize; }
    public PrefixHitStats stats() { return stats; }
    public int gpuSize() { lock.lock(); try { return gpu.size(); } finally { lock.unlock(); } }
    public int cpuSize() { lock.lock(); try { return cpu.size(); } finally { lock.unlock(); } }
    public int diskLive() { return disk.liveBlocks(); }

    /**
     * Hash one token block (parent chain) — same algorithm as {@link BlockHashIndex}.
     */
    public static long hashBlock(long parentHash, int[] tokens, int offset, int len) {
        return BlockHashIndex.hashBlock(parentHash, tokens, offset, len);
    }

    /** Store payload for a content hash at L1 (may demote). */
    public void put(long hash, byte[] payload) {
        Objects.requireNonNull(payload, "payload");
        lock.lock();
        try {
            ensureOpen();
            gpu.put(hash, payload.clone());
            cpu.remove(hash);
            demoteGpuIfNeeded();
        } finally {
            lock.unlock();
        }
    }

    /**
     * Lookup by hash; promotes to L1 on L2/L3 hit.
     *
     * @return payload clone or null on miss
     */
    public byte[] get(long hash) {
        lock.lock();
        try {
            ensureOpen();
            byte[] v = gpu.get(hash);
            if (v != null) {
                stats.recordHit(Tier.GPU);
                return v.clone();
            }
            v = cpu.get(hash);
            if (v != null) {
                stats.recordHit(Tier.CPU);
                promoteToGpu(hash, v);
                return v.clone();
            }
            try {
                v = disk.get(hash);
            } catch (IOException e) {
                stats.recordMiss();
                return null;
            }
            if (v != null) {
                stats.recordHit(Tier.DISK);
                putCpu(hash, v);
                promoteToGpu(hash, v);
                return v.clone();
            }
            stats.recordMiss();
            return null;
        } finally {
            lock.unlock();
        }
    }

    public boolean contains(long hash) {
        lock.lock();
        try {
            if (gpu.containsKey(hash) || cpu.containsKey(hash)) return true;
            return disk.contains(hash);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Match a token prefix block-by-block; returns hashes that hit any tier
     * and how many tokens were covered.
     */
    public PrefixMatch matchPrefix(int[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return new PrefixMatch(0, List.of(), List.of());
        }
        List<Long> hashes = new ArrayList<>();
        List<Tier> tiers = new ArrayList<>();
        long parent = 0L;
        int matched = 0;
        for (int off = 0; off + blockSize <= tokens.length; off += blockSize) {
            long h = hashBlock(parent, tokens, off, blockSize);
            Tier t = locate(h);
            if (t == null) {
                break;
            }
            hashes.add(h);
            tiers.add(t);
            // promote path for hit
            get(h);
            parent = h;
            matched += blockSize;
        }
        return new PrefixMatch(matched, hashes, tiers);
    }

    /**
     * Insert full sequence as consecutive blocks; payload is a compact int token dump
     * for the block (demo / correctness). Production would store KV bytes.
     */
    public void insertTokens(int[] tokens) {
        if (tokens == null || tokens.length < blockSize) return;
        long parent = 0L;
        for (int off = 0; off + blockSize <= tokens.length; off += blockSize) {
            long h = hashBlock(parent, tokens, off, blockSize);
            byte[] payload = encodeTokenBlock(tokens, off, blockSize);
            put(h, payload);
            parent = h;
        }
    }

    private Tier locate(long hash) {
        lock.lock();
        try {
            if (gpu.containsKey(hash)) return Tier.GPU;
            if (cpu.containsKey(hash)) return Tier.CPU;
            if (disk.contains(hash)) return Tier.DISK;
            return null;
        } finally {
            lock.unlock();
        }
    }

    private void promoteToGpu(long hash, byte[] v) {
        gpu.put(hash, v.clone());
        stats.recordPromote(v.length);
        if (budget != null) {
            budget.tryReserveGpu(v.length);
        }
        demoteGpuIfNeeded();
    }

    private void putCpu(long hash, byte[] v) {
        cpu.put(hash, v.clone());
        demoteCpuIfNeeded();
    }

    private void demoteGpuIfNeeded() {
        while (gpu.size() > gpuCap || gpuPressure()) {
            Map.Entry<Long, byte[]> e = oldest(gpu);
            if (e == null) break;
            gpu.remove(e.getKey());
            cpu.put(e.getKey(), e.getValue());
            stats.recordDemote(e.getValue().length);
            if (budget != null) {
                budget.releaseGpu(e.getValue().length);
                budget.tryReserveCpu(e.getValue().length);
            }
            demoteCpuIfNeeded();
        }
    }

    private void demoteCpuIfNeeded() {
        while (cpu.size() > cpuCap || cpuPressure()) {
            Map.Entry<Long, byte[]> e = oldest(cpu);
            if (e == null) break;
            cpu.remove(e.getKey());
            try {
                disk.put(e.getKey(), e.getValue());
                stats.recordDemote(e.getValue().length);
                if (budget != null) {
                    budget.releaseCpu(e.getValue().length);
                    budget.tryReserveDisk(e.getValue().length);
                }
            } catch (IOException ex) {
                // drop if disk full
            }
        }
    }

    private boolean gpuPressure() {
        return budget != null && budget.gpuPressure(gpuWatermark);
    }

    private boolean cpuPressure() {
        return budget != null && budget.cpuPressure(cpuWatermark);
    }

    private static Map.Entry<Long, byte[]> oldest(LinkedHashMap<Long, byte[]> map) {
        if (map.isEmpty()) return null;
        return map.entrySet().iterator().next();
    }

    public static byte[] encodeTokenBlock(int[] tokens, int off, int len) {
        ByteBuffer buf = ByteBuffer.allocate(len * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < len; i++) {
            buf.putInt(tokens[off + i]);
        }
        return buf.array();
    }

    public static int[] decodeTokenBlock(byte[] payload) {
        if (payload == null) return new int[0];
        ByteBuffer buf = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN);
        int n = payload.length / 4;
        int[] t = new int[n];
        for (int i = 0; i < n; i++) t[i] = buf.getInt();
        return t;
    }

    /** Force demote of one GPU entry (budget / long-context tests). */
    public boolean forceDemoteOneFromGpu() {
        lock.lock();
        try {
            Map.Entry<Long, byte[]> e = oldest(gpu);
            if (e == null) return false;
            gpu.remove(e.getKey());
            cpu.put(e.getKey(), e.getValue());
            stats.recordDemote(e.getValue().length);
            demoteCpuIfNeeded();
            return true;
        } finally {
            lock.unlock();
        }
    }

    /** Force demote of one CPU entry to disk. */
    public boolean forceDemoteOneFromCpu() {
        lock.lock();
        try {
            Map.Entry<Long, byte[]> e = oldest(cpu);
            if (e == null) return false;
            cpu.remove(e.getKey());
            try {
                disk.put(e.getKey(), e.getValue());
                stats.recordDemote(e.getValue().length);
                return true;
            } catch (IOException ex) {
                return false;
            }
        } finally {
            lock.unlock();
        }
    }

    private void ensureOpen() {
        if (closed) throw new IllegalStateException("ThreeTierPrefixCache closed");
    }

    @Override
    public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            gpu.clear();
            cpu.clear();
            disk.close();
        } finally {
            lock.unlock();
        }
    }

    /** Result of a multi-block prefix match. */
    public static final class PrefixMatch {
        public final int matchedTokens;
        public final List<Long> hashes;
        public final List<Tier> tiers;

        public PrefixMatch(int matchedTokens, List<Long> hashes, List<Tier> tiers) {
            this.matchedTokens = matchedTokens;
            this.hashes = List.copyOf(hashes);
            this.tiers = List.copyOf(tiers);
        }

        public boolean hit() { return matchedTokens > 0; }

        @Override
        public String toString() {
            return "PrefixMatch{tokens=" + matchedTokens + ", blocks=" + hashes.size()
                    + ", tiers=" + tiers + "}";
        }
    }
}
