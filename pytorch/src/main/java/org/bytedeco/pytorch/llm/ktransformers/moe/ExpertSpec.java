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
package org.bytedeco.pytorch.llm.ktransformers.moe;

import org.bytedeco.pytorch.llm.modules.Mlp;
import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Metadata + module handle for a single MoE expert in the heterogeneous pool.
 *
 * <p>The compute module is typically {@link Mlp.SwiGLU} (or a quantized FFN).
 * Residency is mutable under the scheduler lock in {@link ExpertPool}.
 */
public final class ExpertSpec {

    private final int expertId;
    private final Module module;
    private volatile ExpertDevice device;
    private final int numaNode;
    private final LongAdder hitCount = new LongAdder();
    private final AtomicLong lastUsedStep = new AtomicLong(0);
    private final long paramBytesEstimate;

    public ExpertSpec(int expertId, Module module, ExpertDevice device, int numaNode,
                      long paramBytesEstimate) {
        if (expertId < 0) {
            throw new IllegalArgumentException("expertId must be >= 0");
        }
        this.expertId = expertId;
        this.module = Objects.requireNonNull(module, "module");
        this.device = device != null ? device : ExpertDevice.CPU;
        this.numaNode = Math.max(0, numaNode);
        this.paramBytesEstimate = Math.max(0L, paramBytesEstimate);
    }

    public ExpertSpec(int expertId, Mlp.SwiGLU module, ExpertDevice device) {
        this(expertId, module, device, 0, estimateSwiGluBytes(module));
    }

    public int expertId() { return expertId; }
    public Module module() { return module; }
    public ExpertDevice device() { return device; }
    public int numaNode() { return numaNode; }
    public long hits() { return hitCount.sum(); }
    public long lastUsedStep() { return lastUsedStep.get(); }
    public long paramBytesEstimate() { return paramBytesEstimate; }

    void setDevice(ExpertDevice d) {
        this.device = d != null ? d : ExpertDevice.CPU;
    }

    void recordHit(long step) {
        hitCount.increment();
        lastUsedStep.set(step);
    }

    public void resetStats() {
        hitCount.reset();
        lastUsedStep.set(0);
    }

    private static long estimateSwiGluBytes(Mlp.SwiGLU m) {
        if (m == null) return 0L;
        // Rough: 3 linear layers (gate/up/down) * H * I * 4 bytes
        try {
            long h = m.hiddenSize();
            long i = m.intermediateSize();
            return 3L * h * i * 4L;
        } catch (Throwable t) {
            return 0L;
        }
    }

    @Override
    public String toString() {
        return "ExpertSpec{id=" + expertId + ", device=" + device
                + ", numa=" + numaNode + ", hits=" + hits() + "}";
    }
}
