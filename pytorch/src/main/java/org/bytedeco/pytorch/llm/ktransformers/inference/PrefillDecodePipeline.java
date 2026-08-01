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
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.ktransformers.cache.KtCacheManager;
import org.bytedeco.pytorch.llm.ktransformers.cache.ThreeTierPrefixCache;
import org.bytedeco.pytorch.llm.ktransformers.config.KtInferenceConfig;
import org.bytedeco.pytorch.llm.ktransformers.model.KtMiniMoECausalLM;
import org.bytedeco.pytorch.llm.ktransformers.util.Timing;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Prefill + autoregressive decode pipeline for a single request.
 *
 * <p>Optional three-tier prefix lookup short-circuits matched prompt tokens for
 * metrics (full recompute still runs on the mini model so logits stay correct
 * without storing real KV tensors in the prefix store).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PrefillDecodePipeline {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final Module model;
    private final KtInferenceConfig inferenceConfig;
    private final KtCacheManager cacheManager;
    private final int vocabSize;

    public PrefillDecodePipeline(Module model, KtInferenceConfig inferenceConfig,
                                 KtCacheManager cacheManager, int vocabSize) {
        this.model = Objects.requireNonNull(model, "model");
        this.inferenceConfig = Objects.requireNonNull(inferenceConfig, "inferenceConfig");
        this.cacheManager = cacheManager;
        this.vocabSize = Math.max(1, vocabSize);
    }

    public PrefillDecodePipeline(KtMiniMoECausalLM model, KtInferenceConfig cfg,
                                 KtCacheManager cache) {
        this(model, cfg, cache, model.vocabSize());
    }

    public KtGenerateOutput generate(KtGenerateRequest req) {
        Objects.requireNonNull(req, "req");
        int[] prompt = req.promptTokenIds();
        int maxNew = Math.min(req.maxNewTokens(), inferenceConfig.maxNewTokens());
        maxNew = Math.min(maxNew, Math.max(1, inferenceConfig.maxSeqLen() - prompt.length));

        int prefixHit = 0;
        if (req.usePrefixCache() && cacheManager != null && cacheManager.config().prefixEnable()) {
            ThreeTierPrefixCache.PrefixMatch m = cacheManager.lookupPrefix(prompt);
            prefixHit = m.matchedTokens;
        }

        Timing prefill = Timing.start();
        // Prefill: full forward on prompt to obtain last-token logits
        long[] ids = toLong(prompt);
        Tensor input = org.bytedeco.pytorch.global.torch.tensor(ids).unsqueeze(0);
        Tensor logits;
        try {
            logits = model.forward(input);
        } finally {
            input.close();
        }
        long prefillNs = prefill.elapsedNs();

        List<Integer> seq = new ArrayList<>(prompt.length + maxNew);
        for (int t : prompt) seq.add(Math.floorMod(t, vocabSize));

        Timing decode = Timing.start();
        int generated = 0;
        Random rng = req.seed() != 0L ? new Random(req.seed()) : null;
        for (int step = 0; step < maxNew; step++) {
            Tensor lastLogits = logits.slice(1, new LongOptional(logits.size(1) - 1),
                    new LongOptional(logits.size(1)), 1).squeeze(0).squeeze(0); // [V]
            int next = sample(lastLogits, req, rng);
            lastLogits.close();
            logits.close();
            seq.add(next);
            generated++;

            // Decode step: re-forward full sequence (mini model; no real KV cache required for correctness)
            long[] full = new long[seq.size()];
            for (int i = 0; i < seq.size(); i++) full[i] = seq.get(i);
            Tensor stepIn = org.bytedeco.pytorch.global.torch.tensor(full).unsqueeze(0);
            try {
                logits = model.forward(stepIn);
            } finally {
                stepIn.close();
            }
        }
        long decodeNs = decode.elapsedNs();
        if (logits != null) {
            logits.close();
        }

        if (req.usePrefixCache() && cacheManager != null) {
            int[] fullTokens = new int[seq.size()];
            for (int i = 0; i < seq.size(); i++) fullTokens[i] = seq.get(i);
            cacheManager.rememberPrefix(fullTokens);
        }

        int[] outIds = new int[seq.size()];
        for (int i = 0; i < seq.size(); i++) outIds[i] = seq.get(i);

        Map<String, Double> metrics = new LinkedHashMap<>();
        metrics.put("kt/infer/prompt_tokens", (double) prompt.length);
        metrics.put("kt/infer/new_tokens", (double) generated);
        metrics.put("kt/infer/prefix_hit_tokens", (double) prefixHit);
        metrics.put("kt/infer/prefill_ns", (double) prefillNs);
        metrics.put("kt/infer/decode_ns", (double) decodeNs);
        if (prefillNs > 0) {
            metrics.put("kt/infer/prefill_tok_s", prompt.length * 1e9 / prefillNs);
        }
        if (decodeNs > 0 && generated > 0) {
            metrics.put("kt/infer/decode_tok_s", generated * 1e9 / decodeNs);
        }
        if (cacheManager != null) {
            metrics.putAll(cacheManager.stats().toMetricMap());
        }

        return new KtGenerateOutput(req.requestId(), outIds, prompt.length, generated,
                prefillNs, decodeNs, prefixHit, metrics);
    }

    private int sample(Tensor lastLogits, KtGenerateRequest req, Random rng) {
        double temp = req.temperature() > 0 ? req.temperature() : inferenceConfig.temperature();
        if (temp <= 1e-6) {
            return (int) lastLogits.argmax().item_long();
        }
        // Temperature softmax sample (host path for determinism on mini models)
        float[] logits = org.bytedeco.pytorch.llm.ktransformers.kernel.DequantOps.toFloatArray(lastLogits);
        int V = logits.length;
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) if (v > max) max = v;
        double[] p = new double[V];
        double sum = 0;
        for (int i = 0; i < V; i++) {
            p[i] = Math.exp((logits[i] - max) / temp);
            sum += p[i];
        }
        for (int i = 0; i < V; i++) p[i] /= sum;
        // optional top-k
        int topK = req.topK() > 0 ? req.topK() : inferenceConfig.topK();
        if (topK > 0 && topK < V) {
            // zero out all but topK — simple selection
            int[] idx = new int[V];
            for (int i = 0; i < V; i++) idx[i] = i;
            for (int a = 0; a < topK; a++) {
                int best = a;
                for (int b = a + 1; b < V; b++) if (p[idx[b]] > p[idx[best]]) best = b;
                int tmp = idx[a]; idx[a] = idx[best]; idx[best] = tmp;
            }
            double[] p2 = new double[V];
            double s2 = 0;
            for (int a = 0; a < topK; a++) {
                p2[idx[a]] = p[idx[a]];
                s2 += p2[idx[a]];
            }
            if (s2 > 0) for (int i = 0; i < V; i++) p[i] = p2[i] / s2;
        }
        Random r = rng != null ? rng : new Random();
        double u = r.nextDouble();
        double c = 0;
        for (int i = 0; i < V; i++) {
            c += p[i];
            if (u <= c) return i;
        }
        return V - 1;
    }

    private static long[] toLong(int[] a) {
        long[] o = new long[a.length];
        for (int i = 0; i < a.length; i++) o[i] = a[i];
        return o;
    }
}
