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
package org.bytedeco.pytorch.utils.vllm.metrics;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/** Lightweight metrics emitted by {@code LLMEngine}. */
public final class EngineMetrics {

    public final LongAdder numRequests = new LongAdder();
    public final LongAdder numTokens = new LongAdder();
    public final LongAdder numPrefillTokens = new LongAdder();
    public final LongAdder numDecodeTokens = new LongAdder();
    public final LongAdder numFinished = new LongAdder();
    public final LongAdder numStepCalls = new LongAdder();
    public final AtomicLong totalStepTimeMs = new AtomicLong(0);
    public final AtomicLong cacheHits = new AtomicLong(0);
    public final AtomicLong cacheMisses = new AtomicLong(0);

    public void recordStep(long stepMs) {
        numStepCalls.increment();
        totalStepTimeMs.addAndGet(stepMs);
    }

    public double avgStepMs() {
        long n = numStepCalls.sum();
        return n == 0 ? 0 : (double) totalStepTimeMs.get() / n;
    }

    public double tokensPerSecond(long elapsedMs) {
        return elapsedMs <= 0 ? 0 : numTokens.sum() * 1000.0 / elapsedMs;
    }

    @Override
    public String toString() {
        return "EngineMetrics{req=" + numRequests.sum()
                + ", tokens=" + numTokens.sum()
                + ", finished=" + numFinished.sum()
                + ", steps=" + numStepCalls.sum()
                + ", avgStepMs=" + String.format("%.2f", avgStepMs())
                + ", cacheHits=" + cacheHits.get()
                + ", cacheMisses=" + cacheMisses.get() + "}";
    }
}
