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
package org.bytedeco.pytorch.llm.ktransformers.monitor;

import org.bytedeco.pytorch.llm.ktransformers.inference.KtGenerateOutput;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

/**
 * Process-local scalar metrics for inference + SFT (TensorBoard / Board keys).
 */
public final class KtMetrics {

    private final ConcurrentHashMap<String, Double> gauges = new ConcurrentHashMap<>();
    private final LongAdder generateCalls = new LongAdder();
    private final LongAdder trainSteps = new LongAdder();
    private final DoubleAdder totalNewTokens = new DoubleAdder();
    private final DoubleAdder totalPrefillNs = new DoubleAdder();
    private final DoubleAdder totalDecodeNs = new DoubleAdder();
    private final DoubleAdder lossSum = new DoubleAdder();

    public void set(String key, double value) {
        if (key != null) {
            gauges.put(key, value);
        }
    }

    public void setAll(Map<String, Double> m) {
        if (m == null) return;
        for (Map.Entry<String, Double> e : m.entrySet()) {
            if (e.getKey() != null && e.getValue() != null) {
                gauges.put(e.getKey(), e.getValue());
            }
        }
    }

    public void recordGenerate(KtGenerateOutput out) {
        if (out == null) return;
        generateCalls.increment();
        totalNewTokens.add(out.newTokens());
        totalPrefillNs.add(out.prefillNanos());
        totalDecodeNs.add(out.decodeNanos());
        set("kt/infer/last_ttft_ms", out.ttftMillis());
        set("kt/infer/last_decode_tok_s", out.decodeTokensPerSec());
        set("kt/infer/last_prefix_hit", out.prefixHitTokens());
        setAll(out.metrics());
    }

    public void recordTrainStep(int step, double loss, double lr, double gradNorm) {
        trainSteps.increment();
        lossSum.add(loss);
        set("kt/train/step", step);
        set("kt/train/loss", loss);
        set("kt/train/lr", lr);
        set("kt/train/grad_norm", gradNorm);
        long n = trainSteps.sum();
        if (n > 0) {
            set("kt/train/loss_avg", lossSum.sum() / n);
        }
    }

    public long generateCalls() { return generateCalls.sum(); }
    public long trainSteps() { return trainSteps.sum(); }

    public Map<String, Double> snapshot() {
        Map<String, Double> m = new LinkedHashMap<>(gauges);
        m.put("kt/infer/generate_calls", (double) generateCalls.sum());
        m.put("kt/infer/total_new_tokens", totalNewTokens.sum());
        m.put("kt/train/steps", (double) trainSteps.sum());
        return Collections.unmodifiableMap(m);
    }

    public void reset() {
        gauges.clear();
        generateCalls.reset();
        trainSteps.reset();
        totalNewTokens.reset();
        totalPrefillNs.reset();
        totalDecodeNs.reset();
        lossSum.reset();
    }
}
