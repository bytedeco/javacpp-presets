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

import org.bytedeco.pytorch.llm.ktransformers.config.KtMoEConfig;
import org.bytedeco.pytorch.llm.modules.Mlp;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Weight residency pool for MoE experts across GPU / CPU (and optional Disk).
 *
 * <p>Holds {@link ExpertSpec} entries, enforces {@code gpuExpertSlots} capacity,
 * and exposes promote/demote used by {@link ExpertScheduler}. Does not move
 * actual tensor storage by itself in the reference path — residency is a
 * control-plane flag that {@link TokenDispatcher} honors (CPU path copies
 * activations; GPU path runs in place). Host meshes may plug real
 * {@code module.to(device)} in a subclass / hook.
 */
public final class ExpertPool implements AutoCloseable {

    private final ExpertSpec[] experts;
    private final int gpuSlots;
    private final ExpertLoadBalanceMetrics metrics;
    private final ReentrantLock lock = new ReentrantLock();
    private final List<ModuleMigrateHook> migrateHooks = new ArrayList<>();
    private long step;
    private boolean closed;

    @FunctionalInterface
    public interface ModuleMigrateHook {
        /**
         * Optional real device move. Default no-op keeps pure control-plane residency.
         *
         * @param expert expert being moved
         * @param from   previous tier
         * @param to     target tier
         */
        void onMigrate(ExpertSpec expert, ExpertDevice from, ExpertDevice to);
    }

    public ExpertPool(List<ExpertSpec> specs, int gpuExpertSlots) {
        Objects.requireNonNull(specs, "specs");
        if (specs.isEmpty()) {
            throw new IllegalArgumentException("experts must be non-empty");
        }
        this.experts = specs.toArray(new ExpertSpec[0]);
        for (int i = 0; i < experts.length; i++) {
            if (experts[i] == null || experts[i].expertId() != i) {
                throw new IllegalArgumentException("experts must be contiguous ids 0..N-1");
            }
        }
        this.gpuSlots = Math.max(0, gpuExpertSlots);
        this.metrics = new ExpertLoadBalanceMetrics(experts.length);
        this.step = 0L;
        // Apply initial capacity: demote overflow GPU experts
        enforceGpuCapacity();
    }

    /**
     * Build a pool of SwiGLU experts with initial placement from schedule policy.
     */
    public static ExpertPool createSwiGLU(long hidden, long intermediate, KtMoEConfig moe,
                                          NumaAwarePlacement numa) {
        Objects.requireNonNull(moe, "moe");
        int n = moe.numExperts();
        List<ExpertSpec> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Mlp.SwiGLU ffn = new Mlp.SwiGLU(hidden, intermediate);
            ExpertDevice dev = initialDevice(i, n, moe);
            int node = numa != null ? numa.nodeForExpert(i) : 0;
            list.add(new ExpertSpec(i, ffn, dev, node, 0L));
        }
        return new ExpertPool(list, moe.gpuExpertSlots());
    }

    private static ExpertDevice initialDevice(int id, int n, KtMoEConfig moe) {
        int slots = Math.min(moe.gpuExpertSlots(), n);
        switch (moe.schedule()) {
            case GPU_FIRST:
                return id < slots ? ExpertDevice.GPU : ExpertDevice.CPU;
            case CPU_FIRST:
                return ExpertDevice.CPU;
            case BALANCED:
                return (id % 2 == 0 && id / 2 < slots) || id < slots
                        ? (id < slots ? ExpertDevice.GPU : ExpertDevice.CPU)
                        : ExpertDevice.CPU;
            case AUTO:
            default:
                // start with first `slots` on GPU
                return id < slots ? ExpertDevice.GPU : ExpertDevice.CPU;
        }
    }

    public int numExperts() { return experts.length; }
    public int gpuSlots() { return gpuSlots; }
    public ExpertLoadBalanceMetrics metrics() { return metrics; }
    public long step() { return step; }

    public ExpertSpec get(int id) {
        if (id < 0 || id >= experts.length) {
            throw new IndexOutOfBoundsException("expert id " + id);
        }
        return experts[id];
    }

    public ExpertSpec[] all() {
        return experts.clone();
    }

    public List<ExpertSpec> snapshot() {
        return Collections.unmodifiableList(ArraysAsList());
    }

    private List<ExpertSpec> ArraysAsList() {
        List<ExpertSpec> l = new ArrayList<>(experts.length);
        Collections.addAll(l, experts);
        return l;
    }

    public void addMigrateHook(ModuleMigrateHook hook) {
        if (hook != null) {
            lock.lock();
            try {
                migrateHooks.add(hook);
            } finally {
                lock.unlock();
            }
        }
    }

    public int gpuResidentCount() {
        int c = 0;
        for (ExpertSpec e : experts) {
            if (e.device() == ExpertDevice.GPU) c++;
        }
        return c;
    }

    public long advanceStep() {
        lock.lock();
        try {
            return ++step;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Promote expert to GPU, demoting coldest GPU expert if over capacity.
     *
     * @return true if residency changed to GPU
     */
    public boolean promoteToGpu(int expertId) {
        lock.lock();
        try {
            ensureOpen();
            ExpertSpec e = get(expertId);
            if (e.device() == ExpertDevice.GPU) {
                return false;
            }
            if (gpuResidentCount() >= gpuSlots && gpuSlots > 0) {
                int cold = metrics.coldestOn(ExpertDevice.GPU, experts);
                if (cold >= 0 && cold != expertId) {
                    demoteToCpuLocked(cold);
                } else if (gpuResidentCount() >= gpuSlots) {
                    return false;
                }
            }
            ExpertDevice from = e.device();
            e.setDevice(ExpertDevice.GPU);
            fireHooks(e, from, ExpertDevice.GPU);
            metrics.recordPromote();
            return true;
        } finally {
            lock.unlock();
        }
    }

    public boolean demoteToCpu(int expertId) {
        lock.lock();
        try {
            return demoteToCpuLocked(expertId);
        } finally {
            lock.unlock();
        }
    }

    private boolean demoteToCpuLocked(int expertId) {
        ensureOpen();
        ExpertSpec e = get(expertId);
        if (e.device() == ExpertDevice.CPU) {
            return false;
        }
        ExpertDevice from = e.device();
        e.setDevice(ExpertDevice.CPU);
        fireHooks(e, from, ExpertDevice.CPU);
        metrics.recordDemote();
        return true;
    }

    private void fireHooks(ExpertSpec e, ExpertDevice from, ExpertDevice to) {
        for (ModuleMigrateHook h : migrateHooks) {
            try {
                h.onMigrate(e, from, to);
            } catch (RuntimeException ex) {
                // hooks must not break scheduling
            }
        }
    }

    private void enforceGpuCapacity() {
        int gpu = 0;
        for (ExpertSpec e : experts) {
            if (e.device() == ExpertDevice.GPU) {
                gpu++;
                if (gpu > gpuSlots) {
                    e.setDevice(ExpertDevice.CPU);
                }
            }
        }
    }

    public void recordHit(int expertId) {
        ExpertSpec e = get(expertId);
        e.recordHit(step);
        metrics.recordSelection(expertId, e.device());
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("ExpertPool closed");
        }
    }

    @Override
    public void close() {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            for (ExpertSpec e : experts) {
                Module m = e.module();
                if (m != null) {
                    try {
                        m.close();
                    } catch (Throwable ignored) {
                    }
                }
            }
        } finally {
            lock.unlock();
        }
    }
}
