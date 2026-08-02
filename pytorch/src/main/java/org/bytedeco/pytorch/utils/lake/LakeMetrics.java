/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.utils.lake;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Lightweight process-local metrics for lake adapters (QPS / latency / rows / failures).
 *
 * <p>Not a full metrics backend — callers can scrape {@link #snapshot()} into Micrometer / Prometheus.
 * Hot path uses {@link LongAdder} / {@link AtomicLong} only.</p>
 */
public final class LakeMetrics {

    private final String name;
    private final LongAdder rowsRead = new LongAdder();
    private final LongAdder rowsWritten = new LongAdder();
    private final LongAdder bytesWritten = new LongAdder();
    private final LongAdder queries = new LongAdder();
    private final LongAdder loads = new LongAdder();
    private final LongAdder failures = new LongAdder();
    private final LongAdder batches = new LongAdder();
    private final AtomicLong lastLatencyNs = new AtomicLong();
    private final AtomicLong maxLatencyNs = new AtomicLong();
    private final LongAdder totalLatencyNs = new LongAdder();

    public LakeMetrics(String name) {
        this.name = name == null || name.isBlank() ? "lake" : name;
    }

    public static LakeMetrics of(String name) {
        return new LakeMetrics(name);
    }

    public String name() {
        return name;
    }

    public void recordRead(long rows, long latencyNs) {
        if (rows > 0) rowsRead.add(rows);
        queries.increment();
        recordLatency(latencyNs);
    }

    public void recordWrite(long rows, long bytes, long latencyNs) {
        if (rows > 0) rowsWritten.add(rows);
        if (bytes > 0) bytesWritten.add(bytes);
        loads.increment();
        recordLatency(latencyNs);
    }

    public void recordBatch(long rows) {
        batches.increment();
        if (rows > 0) rowsRead.add(rows);
    }

    public void recordFailure() {
        failures.increment();
    }

    private void recordLatency(long latencyNs) {
        if (latencyNs < 0) return;
        lastLatencyNs.set(latencyNs);
        totalLatencyNs.add(latencyNs);
        long cur;
        do {
            cur = maxLatencyNs.get();
            if (latencyNs <= cur) break;
        } while (!maxLatencyNs.compareAndSet(cur, latencyNs));
    }

    public long rowsRead() { return rowsRead.sum(); }
    public long rowsWritten() { return rowsWritten.sum(); }
    public long bytesWritten() { return bytesWritten.sum(); }
    public long queries() { return queries.sum(); }
    public long loads() { return loads.sum(); }
    public long failures() { return failures.sum(); }
    public long batches() { return batches.sum(); }
    public long lastLatencyNs() { return lastLatencyNs.get(); }
    public long maxLatencyNs() { return maxLatencyNs.get(); }

    /** Average latency in nanoseconds across recorded ops (queries + loads). */
    public long avgLatencyNs() {
        long ops = queries.sum() + loads.sum();
        if (ops == 0) return 0L;
        return totalLatencyNs.sum() / ops;
    }

    public Map<String, Long> snapshot() {
        Map<String, Long> m = new LinkedHashMap<>();
        m.put("rows_read", rowsRead());
        m.put("rows_written", rowsWritten());
        m.put("bytes_written", bytesWritten());
        m.put("queries", queries());
        m.put("loads", loads());
        m.put("batches", batches());
        m.put("failures", failures());
        m.put("last_latency_ns", lastLatencyNs());
        m.put("max_latency_ns", maxLatencyNs());
        m.put("avg_latency_ns", avgLatencyNs());
        return m;
    }

    public void reset() {
        rowsRead.reset();
        rowsWritten.reset();
        bytesWritten.reset();
        queries.reset();
        loads.reset();
        failures.reset();
        batches.reset();
        lastLatencyNs.set(0);
        maxLatencyNs.set(0);
        totalLatencyNs.reset();
    }

    @Override
    public String toString() {
        return "LakeMetrics{" + name + " " + snapshot() + "}";
    }
}
