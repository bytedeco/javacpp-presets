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
import org.bytedeco.pytorch.llm.kvcache.HierarchicalKvCache;
import org.bytedeco.pytorch.llm.kvcache.PrefixRadixCache;

import java.io.IOException;
import java.util.Objects;

/**
 * Bridge between KT three-tier prefix cache and existing {@code llm.kvcache} types.
 *
 * <p>Owns:
 * <ul>
 *   <li>{@link ThreeTierPrefixCache} — content-addressed GPU/CPU/Disk prefix reuse</li>
 *   <li>optional {@link HierarchicalKvCache} — per-sequence hot/cold paged KV</li>
 * </ul>
 * Does not reimplement paged block managers; composes them.
 */
public final class KtCacheManager implements AutoCloseable {

    private final KtCacheConfig config;
    private final ThreeTierPrefixCache prefix;
    private final HierarchicalKvCache hierarchical;
    private final DeviceBudget budget;
    private boolean closed;

    public KtCacheManager(KtCacheConfig config, DeviceBudget budget,
                          boolean createHierarchical) throws IOException {
        this.config = Objects.requireNonNull(config, "config");
        this.budget = budget != null ? budget : DeviceBudget.mini();
        this.prefix = new ThreeTierPrefixCache(config, this.budget);
        if (createHierarchical) {
            this.hierarchical = new HierarchicalKvCache(
                    config.gpuBlocks(),
                    config.cpuBlocks(),
                    config.numLayers(),
                    config.blockSize(),
                    config.numHeads(),
                    config.headDim());
        } else {
            this.hierarchical = null;
        }
    }

    public static KtCacheManager open(KtCacheConfig config) throws IOException {
        return new KtCacheManager(config, DeviceBudget.mini(), true);
    }

    public static KtCacheManager mini() throws IOException {
        return open(KtCacheConfig.mini());
    }

    /** Prefix-only manager (no HierarchicalKvCache). */
    public static KtCacheManager prefixOnly(KtCacheConfig config) throws IOException {
        return new KtCacheManager(config, DeviceBudget.mini(), false);
    }

    public KtCacheConfig config() { return config; }
    public ThreeTierPrefixCache prefix() { return prefix; }
    public HierarchicalKvCache hierarchical() { return hierarchical; }
    public DeviceBudget budget() { return budget; }
    public PrefixHitStats stats() { return prefix.stats(); }

    /**
     * Lookup shared prefix for token ids; returns matched token count.
     * When hierarchical is present, sequence creation is left to the caller.
     */
    public ThreeTierPrefixCache.PrefixMatch lookupPrefix(int[] tokens) {
        if (!config.prefixEnable() || tokens == null) {
            return new ThreeTierPrefixCache.PrefixMatch(0, java.util.List.of(), java.util.List.of());
        }
        return prefix.matchPrefix(tokens);
    }

    public void rememberPrefix(int[] tokens) {
        if (config.prefixEnable() && tokens != null) {
            prefix.insertTokens(tokens);
        }
    }

    public long createSequence() {
        if (hierarchical == null) {
            throw new IllegalStateException("hierarchical KV not enabled");
        }
        return hierarchical.createSequence();
    }

    public void releaseSequence(long seqId) {
        if (hierarchical != null) {
            hierarchical.releaseSequence(seqId);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            prefix.close();
        } finally {
            if (hierarchical != null) {
                try {
                    hierarchical.close();
                } catch (Throwable ignored) {
                }
            }
        }
    }
}
