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
package org.bytedeco.pytorch.llm.llamafactory.webui;

import org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Live training board state for LLaMA-Factory / host mesh UIs.
 *
 * <p>Holds scalar metrics, status text, loss history and a cooperative stop flag.
 * Used by {@link FinetuneAdapter#board()}.
 */
public final class BoardState {

    public enum Status {
        IDLE,
        RUNNING,
        STOPPING,
        STOPPED,
        COMPLETED,
        FAILED
    }

    private final AtomicReference<Status> status = new AtomicReference<>(Status.IDLE);
    private final AtomicInteger globalStep = new AtomicInteger(0);
    private final AtomicInteger epoch = new AtomicInteger(0);
    private final AtomicReference<String> message = new AtomicReference<>("");
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private final ConcurrentHashMap<String, Double> metrics = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<Double> lossHistory = new CopyOnWriteArrayList<>();
    private final CopyOnWriteArrayList<String> logs = new CopyOnWriteArrayList<>();
    private final long startedAtMs = System.currentTimeMillis();
    private volatile long updatedAtMs = startedAtMs;

    public Status status() { return status.get(); }
    public int globalStep() { return globalStep.get(); }
    public int epoch() { return epoch.get(); }
    public String message() { return message.get(); }
    public boolean stopRequested() { return stopRequested.get(); }
    public long startedAtMs() { return startedAtMs; }
    public long updatedAtMs() { return updatedAtMs; }

    public void setStatus(Status s) {
        if (s != null) {
            status.set(s);
            touch();
        }
    }

    public void setMessage(String msg) {
        message.set(msg != null ? msg : "");
        touch();
    }

    public void setGlobalStep(int step) {
        globalStep.set(step);
        touch();
    }

    public void setEpoch(int e) {
        epoch.set(e);
        touch();
    }

    public void requestStop() {
        stopRequested.set(true);
        status.compareAndSet(Status.RUNNING, Status.STOPPING);
        touch();
    }

    public void clearStop() {
        stopRequested.set(false);
    }

    public void putMetric(String key, double value) {
        if (key != null) {
            metrics.put(key, value);
            touch();
        }
    }

    public void putMetrics(Map<String, Double> m) {
        if (m == null) return;
        for (Map.Entry<String, Double> e : m.entrySet()) {
            if (e.getKey() != null && e.getValue() != null) {
                metrics.put(e.getKey(), e.getValue());
            }
        }
        touch();
    }

    public void recordLoss(double loss) {
        lossHistory.add(loss);
        metrics.put("loss", loss);
        touch();
    }

    public void log(String line) {
        if (line != null) {
            logs.add(line);
            if (logs.size() > 2000) {
                logs.remove(0);
            }
            touch();
        }
    }

    public Map<String, Double> metrics() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(metrics));
    }

    public List<Double> lossHistory() {
        return Collections.unmodifiableList(new ArrayList<>(lossHistory));
    }

    public List<String> logs() {
        return Collections.unmodifiableList(new ArrayList<>(logs));
    }

    /** Flat snapshot for JSON / polling demos. */
    public Map<String, Object> snapshot() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("status", status.get().name());
        m.put("global_step", globalStep.get());
        m.put("epoch", epoch.get());
        m.put("message", message.get());
        m.put("stop_requested", stopRequested.get());
        m.put("started_at_ms", startedAtMs);
        m.put("updated_at_ms", updatedAtMs);
        m.put("metrics", new LinkedHashMap<>(metrics));
        m.put("loss_history", new ArrayList<>(lossHistory));
        int from = Math.max(0, logs.size() - 50);
        m.put("logs_tail", new ArrayList<>(logs.subList(from, logs.size())));
        return m;
    }

    private void touch() {
        updatedAtMs = System.currentTimeMillis();
    }
}
