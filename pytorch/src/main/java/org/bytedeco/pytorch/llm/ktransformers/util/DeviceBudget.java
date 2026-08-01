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
package org.bytedeco.pytorch.llm.ktransformers.util;

import org.bytedeco.pytorch.llm.ktransformers.config.KtDevicePlacement;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Logical VRAM / DRAM / Disk budgets for heterogeneous placement decisions.
 *
 * <p>Tracks reserved bytes; does not query the OS. Engines and schedulers use this
 * to decide demote/promote and expert migration — matching the control-plane role
 * of upstream device planners without hard-coding a single SKU (e.g. 24GB cards).
 */
public final class DeviceBudget {

    private final long gpuBudgetBytes;
    private final long cpuBudgetBytes;
    private final long diskBudgetBytes;
    private final AtomicLong gpuUsed = new AtomicLong();
    private final AtomicLong cpuUsed = new AtomicLong();
    private final AtomicLong diskUsed = new AtomicLong();

    public DeviceBudget(long gpuBudgetBytes, long cpuBudgetBytes, long diskBudgetBytes) {
        KtPreconditions.checkArgument(gpuBudgetBytes > 0, "gpuBudgetBytes must be > 0");
        KtPreconditions.checkArgument(cpuBudgetBytes > 0, "cpuBudgetBytes must be > 0");
        KtPreconditions.checkArgument(diskBudgetBytes >= 0, "diskBudgetBytes must be >= 0");
        this.gpuBudgetBytes = gpuBudgetBytes;
        this.cpuBudgetBytes = cpuBudgetBytes;
        this.diskBudgetBytes = diskBudgetBytes;
    }

    public static DeviceBudget fromPlacement(KtDevicePlacement p, long assumedGpuBytes) {
        long gpu = Math.max(1L, (long) (assumedGpuBytes * p.gpuMemFraction()));
        return new DeviceBudget(gpu, p.cpuExpertBytesBudget(), p.diskBytesBudget());
    }

    /** Conservative default: 24 GiB GPU class, 128 GiB host, 512 GiB disk. */
    public static DeviceBudget consumer24g() {
        return new DeviceBudget(24L << 30, 128L << 30, 512L << 30);
    }

    public static DeviceBudget mini() {
        return new DeviceBudget(256L << 20, 1L << 30, 2L << 30);
    }

    public long gpuBudgetBytes() { return gpuBudgetBytes; }
    public long cpuBudgetBytes() { return cpuBudgetBytes; }
    public long diskBudgetBytes() { return diskBudgetBytes; }
    public long gpuUsed() { return gpuUsed.get(); }
    public long cpuUsed() { return cpuUsed.get(); }
    public long diskUsed() { return diskUsed.get(); }

    public double gpuUtilization() {
        return gpuUsed.get() / (double) gpuBudgetBytes;
    }

    public double cpuUtilization() {
        return cpuUsed.get() / (double) cpuBudgetBytes;
    }

    public boolean tryReserveGpu(long bytes) {
        while (true) {
            long cur = gpuUsed.get();
            if (cur + bytes > gpuBudgetBytes) return false;
            if (gpuUsed.compareAndSet(cur, cur + bytes)) return true;
        }
    }

    public boolean tryReserveCpu(long bytes) {
        while (true) {
            long cur = cpuUsed.get();
            if (cur + bytes > cpuBudgetBytes) return false;
            if (cpuUsed.compareAndSet(cur, cur + bytes)) return true;
        }
    }

    public boolean tryReserveDisk(long bytes) {
        if (diskBudgetBytes <= 0) return false;
        while (true) {
            long cur = diskUsed.get();
            if (cur + bytes > diskBudgetBytes) return false;
            if (diskUsed.compareAndSet(cur, cur + bytes)) return true;
        }
    }

    public void releaseGpu(long bytes) {
        gpuUsed.addAndGet(-Math.min(bytes, gpuUsed.get()));
    }

    public void releaseCpu(long bytes) {
        cpuUsed.addAndGet(-Math.min(bytes, cpuUsed.get()));
    }

    public void releaseDisk(long bytes) {
        diskUsed.addAndGet(-Math.min(bytes, diskUsed.get()));
    }

    public boolean gpuPressure(double watermark) {
        return gpuUtilization() >= watermark;
    }

    public boolean cpuPressure(double watermark) {
        return cpuUtilization() >= watermark;
    }

    /**
     * Whether the planner may promote another expert onto GPU.
     * Uses a soft watermark (90%) so decode / KV still have headroom.
     */
    public boolean allowsGpuExpertPromote() {
        return !gpuPressure(0.90);
    }

    public boolean allowsGpuExpertPromote(long extraBytes) {
        return gpuUsed.get() + Math.max(0L, extraBytes) <= (long) (gpuBudgetBytes * 0.90);
    }

    public long gpuFreeBytes() {
        return Math.max(0L, gpuBudgetBytes - gpuUsed.get());
    }

    public long cpuFreeBytes() {
        return Math.max(0L, cpuBudgetBytes - cpuUsed.get());
    }

    @Override
    public String toString() {
        return "DeviceBudget{gpu=" + gpuUsed + "/" + gpuBudgetBytes
                + ", cpu=" + cpuUsed + "/" + cpuBudgetBytes
                + ", disk=" + diskUsed + "/" + diskBudgetBytes + "}";
    }
}
