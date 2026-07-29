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
package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Explicit block/session LRU policy over {@link PagedBlockManager}.
 *
 * <p>Each sequence owns a block table. On allocation pressure (free list empty),
 * the least-recently-used <em>other</em> sequence is fully preempted and its
 * blocks released — same idea as {@link CoWBlockManager} session LRU, exposed
 * as a clean {@link KvCache} with metrics.
 *
 * <p>Token write uses {@link PagedBlockManager#writeToken}; gather uses
 * {@link PagedBlockManager#gather}.
 */
public class BlockLruKvCache implements KvCache {

    public final LongAdder appendCount = new LongAdder();
    public final LongAdder allocCount = new LongAdder();
    public final LongAdder preemptCount = new LongAdder();

    private final PagedBlockManager pool;
    private final int numLayers;
    private final int blockSize;
    private final AtomicLong nextId = new AtomicLong(1);
    /** Access-order map of live sequences. */
    private final LinkedHashMap<Long, Seq> lru = new LinkedHashMap<>(16, 0.75f, true);
    private final Object lock = new Object();

    public BlockLruKvCache(int maxBlocks, int numLayers, int blockSize, int numHeads, int headDim) {
        this.pool = new PagedBlockManager(maxBlocks, numLayers, blockSize, numHeads, headDim);
        this.numLayers = numLayers;
        this.blockSize = blockSize;
    }

    public BlockLruKvCache(PagedBlockManager pool) {
        this.pool = pool;
        this.numLayers = pool.numLayers();
        this.blockSize = pool.blockSize();
    }

    public PagedBlockManager pool() {
        return pool;
    }

    public int freeBlocks() {
        return pool.freeBlocks();
    }

    @Override
    public long createSequence() {
        long id = nextId.getAndIncrement();
        synchronized (lock) {
            lru.put(id, new Seq());
        }
        return id;
    }

    @Override
    public void releaseSequence(long seqId) {
        synchronized (lock) {
            Seq s = lru.remove(seqId);
            if (s != null) {
                freeSeq(s);
            }
        }
    }

    @Override
    public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
        if (kLayers == null || kLayers.length != numLayers) {
            throw new IllegalArgumentException("kLayers length");
        }
        synchronized (lock) {
            Seq s = lru.get(seqId);
            if (s == null) {
                throw new IllegalArgumentException("unknown seq " + seqId);
            }
            lru.get(seqId); // LRU touch
            int tokens = inferTokens(kLayers[0]);
            for (int t = 0; t < tokens; t++) {
                ensureCapacity(s, seqId);
                int blockIdx = s.length / blockSize;
                int pos = s.length % blockSize;
                if (pos == 0) {
                    // need new block
                    int bid = allocateWithPreempt(seqId);
                    s.blocks.add(bid);
                    allocCount.increment();
                }
                int bid = s.blocks.get(s.blocks.size() - 1);
                for (int L = 0; L < numLayers; L++) {
                    Tensor k = sliceToken(kLayers[L], t);
                    Tensor v = sliceToken(vLayers[L], t);
                    pool.writeToken(bid, L, pos, k, v);
                }
                s.length++;
                s.totalAppended++;
                appendCount.increment();
            }
        }
    }

    private void ensureCapacity(Seq s, long selfId) {
        // no-op placeholder — allocation happens in allocateWithPreempt
    }

    private int allocateWithPreempt(long selfId) {
        try {
            return pool.allocateBlock();
        } catch (RuntimeException first) {
            // preempt LRU victims until allocation succeeds or only self remains
            while (true) {
                Long victim = null;
                for (Long id : lru.keySet()) {
                    if (id != selfId) {
                        victim = id;
                        break; // eldest in access-order
                    }
                }
                if (victim == null) {
                    throw first;
                }
                Seq vs = lru.remove(victim);
                if (vs != null) {
                    freeSeq(vs);
                    preemptCount.increment();
                }
                try {
                    return pool.allocateBlock();
                } catch (RuntimeException e) {
                    first = e;
                }
            }
        }
    }

    private void freeSeq(Seq s) {
        for (int b : s.blocks) {
            pool.release(b);
        }
        s.blocks.clear();
        s.length = 0;
    }

    @Override
    public Tensor[] gather(long seqId, int layer) {
        synchronized (lock) {
            Seq s = lru.get(seqId);
            if (s == null) {
                throw new IllegalArgumentException("unknown seq " + seqId);
            }
            lru.get(seqId); // touch
            return pool.gather(s.blocks, layer, s.length);
        }
    }

    @Override
    public int sequenceLength(long seqId) {
        synchronized (lock) {
            return require(seqId).totalAppended;
        }
    }

    @Override
    public int retainedLength(long seqId) {
        synchronized (lock) {
            return require(seqId).length;
        }
    }

    @Override
    public int numLayers() {
        return numLayers;
    }

    public List<Integer> blockTable(long seqId) {
        synchronized (lock) {
            return new ArrayList<>(require(seqId).blocks);
        }
    }

    @Override
    public void close() {
        synchronized (lock) {
            for (Seq s : lru.values()) {
                freeSeq(s);
            }
            lru.clear();
        }
        pool.close();
    }

    private Seq require(long id) {
        Seq s = lru.get(id);
        if (s == null) {
            throw new IllegalArgumentException("unknown seq " + id);
        }
        return s;
    }

    private static int inferTokens(Tensor t) {
        if (t.dim() == 2) return 1;
        if (t.dim() == 3) return (int) t.size(0);
        if (t.dim() == 4) return (int) t.size(2);
        throw new IllegalArgumentException("rank " + t.dim());
    }

    private static Tensor sliceToken(Tensor t, int tIdx) {
        if (t.dim() == 2) return t;
        if (t.dim() == 3) return t.select(0, tIdx);
        if (t.dim() == 4) return t.select(0, 0).select(1, tIdx);
        throw new IllegalArgumentException("rank");
    }

    private static final class Seq {
        final List<Integer> blocks = new ArrayList<>();
        int length;
        int totalAppended;
    }
}
