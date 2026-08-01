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
package org.bytedeco.pytorch.llm.ktransformers.inference;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.llm.ktransformers.cache.KtCacheManager;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.model.KtMiniMoECausalLM;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtMetrics;
import org.bytedeco.pytorch.nn.Module;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Primary inference entry (upstream kt-kernel Inference Quick Start).
 *
 * <pre>{@code
 * try (KtInferenceEngine eng = KtInferenceEngine.openMini()) {
 *     KtGenerateOutput out = eng.generate(KtGenerateRequest.of(new int[]{1,2,3}, 8));
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class KtInferenceEngine implements AutoCloseable {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final KtConfig config;
    private final Module model;
    private final boolean ownsModel;
    private final KtCacheManager cacheManager;
    private final PrefillDecodePipeline pipeline;
    private final MultiConcurrencyRuntime runtime;
    private final MultiGpuCoordinator multiGpu;
    private final KtMetrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final AtomicLong generateCount = new AtomicLong();
    private final Map<String, Double> lastMetrics = new ConcurrentHashMap<>();
    private final int vocabSize;

    public KtInferenceEngine(KtConfig config, Module model, boolean ownsModel,
                             KtCacheManager cacheManager) {
        this.config = Objects.requireNonNull(config, "config");
        this.model = Objects.requireNonNull(model, "model");
        this.ownsModel = ownsModel;
        this.cacheManager = cacheManager;
        this.vocabSize = resolveVocab(model, config);
        this.pipeline = new PrefillDecodePipeline(model, config.inference(), cacheManager, vocabSize);
        this.runtime = new MultiConcurrencyRuntime(pipeline, config.inference().concurrency());
        int numExperts = config.moe() != null ? config.moe().numExperts() : 1;
        this.multiGpu = new MultiGpuCoordinator(
                config.inference(), config.placement(), config.numLayers(), numExperts);
        this.metrics = new KtMetrics();
    }

    public static KtInferenceEngine open(KtConfig config) throws IOException {
        Objects.requireNonNull(config, "config");
        KtMiniMoECausalLM model = new KtMiniMoECausalLM(config);
        KtCacheManager cache = KtCacheManager.prefixOnly(config.cache());
        return new KtInferenceEngine(config, model, true, cache);
    }

    public static KtInferenceEngine openMini() throws IOException {
        return open(KtConfig.miniDemo());
    }

    public static KtInferenceEngine wrap(KtConfig config, Module model, KtCacheManager cache) {
        return new KtInferenceEngine(config, model, false, cache);
    }

    public KtConfig config() { return config; }
    public Module model() { return model; }
    public KtCacheManager cacheManager() { return cacheManager; }
    public MultiGpuCoordinator multiGpu() { return multiGpu; }
    public KtMetrics metrics() { return metrics; }
    public long generateCount() { return generateCount.get(); }
    public Map<String, Double> lastMetrics() { return Map.copyOf(lastMetrics); }

    public KtGenerateOutput generate(KtGenerateRequest request) {
        ensureOpen();
        Objects.requireNonNull(request, "request");
        try {
            KtGenerateOutput out = runtime.generate(request);
            generateCount.incrementAndGet();
            lastMetrics.clear();
            lastMetrics.putAll(out.metrics());
            metrics.recordGenerate(out);
            return out;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("generate interrupted", e);
        }
    }

    public KtGenerateOutput generate(int[] prompt, int maxNewTokens) {
        return generate(KtGenerateRequest.of(prompt, maxNewTokens));
    }

    public List<KtGenerateOutput> generateBatch(List<KtGenerateRequest> requests)
            throws InterruptedException, ExecutionException, java.util.concurrent.TimeoutException {
        ensureOpen();
        List<KtGenerateOutput> outs = runtime.generateAll(requests, 0L);
        generateCount.addAndGet(outs.size());
        if (!outs.isEmpty()) {
            lastMetrics.clear();
            lastMetrics.putAll(outs.get(outs.size() - 1).metrics());
            for (KtGenerateOutput o : outs) {
                metrics.recordGenerate(o);
            }
        }
        return outs;
    }

    private static int resolveVocab(Module model, KtConfig config) {
        if (model instanceof KtMiniMoECausalLM) {
            return ((KtMiniMoECausalLM) model).vocabSize();
        }
        return Math.max(1, config.vocabSize());
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KtInferenceEngine closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        try {
            runtime.close();
        } finally {
            try {
                if (cacheManager != null) {
                    cacheManager.close();
                }
            } finally {
                if (ownsModel && model != null) {
                    try {
                        model.close();
                    } catch (Throwable ignored) {
                    }
                }
            }
        }
    }
}
