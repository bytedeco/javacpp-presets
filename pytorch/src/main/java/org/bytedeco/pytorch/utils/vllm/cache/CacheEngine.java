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
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.vllm.cache;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.kvcache.PagedKvCache;
import org.bytedeco.pytorch.utils.vllm.EngineConfig;
import org.bytedeco.pytorch.utils.vllm.Sequence;

import java.util.Objects;

/**
 * Thin adapter: wires engine sequences ↔ {@link PagedKvCache}.
 *
 * <p>Per-layer K/V shape is {@code [numHeads, headDim]} per token.
 * The cache stores them as {@code [2, blockSize, numHeads, headDim]} blocks.
 */
public final class CacheEngine implements AutoCloseable {

    private final PagedKvCache cache;
    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int blockSize;
    private final int maxBlocks;
    private final EngineConfig config;

    public CacheEngine(EngineConfig config) {
        this.config = config;
        this.numLayers = config.numLayers;
        this.numHeads = config.numHeads;
        this.headDim = config.headDim;
        this.blockSize = config.blockSize;
        this.maxBlocks = config.maxBlocks;

        Device dev = "cuda".equalsIgnoreCase(config.device)
                ? new Device("cuda:0") : null;
        this.cache = new PagedKvCache(numLayers, numHeads, headDim, blockSize, maxBlocks, dev);
    }

    public PagedKvCache cache() { return cache; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public int headDim() { return headDim; }
    public int blockSize() { return blockSize; }
    public int freeBlocks() { return cache.freeBlocks(); }
    public int liveSequences() { return cache.liveSequences(); }

    /** Create a cache entry for a newly-scheduled sequence. */
    public long createSequence(Sequence seq) {
        long id = cache.createSequence();
        seq.setCacheSeqId(id);
        return id;
    }

    /**
     * Append K/V tensors for one new token across all layers.
     * kLayers[i] / vLayers[i] each shape {@code [numHeads, headDim]}.
     */
    public void append(long seqId, int tokenId, Tensor[] kLayers, Tensor[] vLayers) {
        cache.append(seqId, tokenId, kLayers, vLayers);
    }

    /** Gather full K/V for a sequence at a given layer. Returns [K, V] each [T, numHeads, headDim]. */
    public Tensor[] gather(long seqId, int layer) {
        return cache.gather(seqId, layer);
    }

    /** Release a sequence and return its blocks to the pool. */
    public void releaseSequence(Sequence seq) {
        if (seq.cacheSeqId() < 0) return;
        cache.releaseSequence(seq.cacheSeqId());
        seq.setCacheSeqId(-1);
    }

    /** Fork a cache entry (prefix sharing). */
    public long fork(long srcSeqId) {
        return cache.fork(srcSeqId);
    }

    @Override
    public void close() {
        cache.close();
    }

    /** Number of blocks needed for a given sequence length (rough upper bound). */
    public int blocksForTokens(int tokens) {
        return (tokens + blockSize - 1) / blockSize;
    }

    public EngineConfig config() { return config; }

    public String stats() {
        return String.format("Cache{kv=%d/%d blocks free, live=%d seqs}",
                freeBlocks(), maxBlocks, liveSequences());
    }
}
