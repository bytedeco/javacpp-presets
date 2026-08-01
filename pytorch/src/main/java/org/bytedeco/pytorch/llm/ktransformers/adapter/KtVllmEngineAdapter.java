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
package org.bytedeco.pytorch.llm.ktransformers.adapter;

import org.bytedeco.pytorch.llm.ktransformers.cache.KtCacheManager;
import org.bytedeco.pytorch.llm.ktransformers.cache.PrefixHitStats;
import org.bytedeco.pytorch.llm.ktransformers.cache.ThreeTierPrefixCache;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.inject.ModelInjector;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtMetrics;
import org.bytedeco.pytorch.llm.ktransformers.moe.RoutedMoE;
import org.bytedeco.pytorch.llm.vllm.EngineConfig;
import org.bytedeco.pytorch.llm.vllm.LLMEngine;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Composition hook: attach KT three-tier prefix cache + inject plan metadata
 * onto a host {@link LLMEngine} without replacing the whole engine.
 *
 * <p>Upstream KT integrates with SGLang / vLLM-like runtimes via scheduler
 * hooks; this adapter is the Java equivalent surface.
 */
public final class KtVllmEngineAdapter implements AutoCloseable {

    private final KtConfig ktConfig;
    private final LLMEngine engine;
    private final KtCacheManager cacheManager;
    private final ModelInjector injector;
    private final KtMetrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final boolean ownsCache;

    public KtVllmEngineAdapter(KtConfig ktConfig, LLMEngine engine, KtCacheManager cacheManager,
                               boolean ownsCache) {
        this.ktConfig = Objects.requireNonNull(ktConfig, "ktConfig");
        this.engine = Objects.requireNonNull(engine, "engine");
        this.cacheManager = cacheManager;
        this.ownsCache = ownsCache;
        this.injector = ModelInjector.forConfig(ktConfig);
        this.metrics = new KtMetrics();
    }

    public static KtVllmEngineAdapter wrap(KtConfig ktConfig, LLMEngine engine) {
        Objects.requireNonNull(ktConfig, "ktConfig");
        Objects.requireNonNull(engine, "engine");
        KtCacheManager cache = null;
        try {
            cache = KtCacheManager.prefixOnly(ktConfig.cache());
        } catch (Exception e) {
            cache = null;
        }
        return new KtVllmEngineAdapter(ktConfig, engine, cache, cache != null);
    }

    /** Suggest EngineConfig overrides derived from KT inference/cache budgets. */
    public static EngineConfig.Builder suggestEngineConfig(KtConfig kt, EngineConfig.Builder base) {
        Objects.requireNonNull(kt, "kt");
        EngineConfig.Builder b = base != null ? base : EngineConfig.builder();
        if (kt.inference() != null) {
            b.maxNumSeqs(Math.max(1, kt.inference().concurrency()));
            b.maxNumBatchedTokens(Math.max(64, kt.inference().maxBatch() * 32));
        }
        if (kt.cache() != null) {
            b.blockSize(Math.max(8, kt.cache().blockSize()));
            b.maxBlocks(Math.max(16, kt.cache().gpuBlocks() + kt.cache().cpuBlocks()));
            b.numLayers(Math.max(1, kt.cache().numLayers()));
            b.numHeads(Math.max(1, kt.cache().numHeads()));
            b.headDim(Math.max(8, kt.cache().headDim()));
        }
        if (kt.vocabSize() > 0) {
            b.vocabSize(kt.vocabSize());
        }
        return b;
    }

    public KtConfig ktConfig() { return ktConfig; }
    public LLMEngine engine() { return engine; }
    public KtCacheManager cacheManager() { return cacheManager; }
    public ModelInjector injector() { return injector; }
    public KtMetrics metrics() { return metrics; }

    /** Record a prefix lookup against the KT three-tier cache (if present). */
    public int lookupPrefix(int[] tokenIds) {
        ensureOpen();
        if (cacheManager == null || tokenIds == null || tokenIds.length == 0) return 0;
        ThreeTierPrefixCache.PrefixMatch match = cacheManager.lookupPrefix(tokenIds);
        int hit = match != null ? match.matchedTokens : 0;
        PrefixHitStats stats = cacheManager.stats();
        if (stats != null) {
            metrics.setAll(stats.toMetricMap());
        }
        metrics.set("kt/vllm/last_prefix_hit", hit);
        return hit;
    }

    public void rememberPrefix(int[] tokenIds) {
        ensureOpen();
        if (cacheManager != null && tokenIds != null) {
            cacheManager.rememberPrefix(tokenIds);
        }
    }

    /** Publish inject + engine scalar snapshot. */
    public Map<String, Double> snapshot() {
        Map<String, Double> m = new LinkedHashMap<>(metrics.snapshot());
        m.putAll(injector.stats());
        m.put("kt/vllm/wrapped", 1.0);
        try {
            if (engine.metrics() != null) {
                m.put("kt/vllm/engine_alive", 1.0);
            }
        } catch (Throwable ignored) {
        }
        return m;
    }

    /**
     * Optional: mark that a host runner layer uses {@link RoutedMoE}.
     * Records a gauge only — actual module swap is host-owned.
     */
    public void noteRoutedMoE(int layerIndex, RoutedMoE moe) {
        if (moe == null) return;
        metrics.set("kt/vllm/routed_moe_layer", layerIndex);
        try {
            metrics.setAll(moe.metrics().toMetricMap());
        } catch (Throwable ignored) {
        }
    }

    private void ensureOpen() {
        if (closed.get()) throw new IllegalStateException("KtVllmEngineAdapter closed");
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        if (ownsCache && cacheManager != null) {
            try {
                cacheManager.close();
            } catch (Throwable ignored) {
            }
        }
    }
}
