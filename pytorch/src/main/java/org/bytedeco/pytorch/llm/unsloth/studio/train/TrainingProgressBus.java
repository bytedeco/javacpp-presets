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

package org.bytedeco.pytorch.llm.unsloth.studio.train;

import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingProgressEvent;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/** Thread-safe progress fan-out for SSE / Board / MCP listeners. */
public final class TrainingProgressBus {

    public interface Listener extends Consumer<TrainingProgressEvent> {}

    private final Map<String, CopyOnWriteArrayList<Listener>> byRun = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<Listener> global = new CopyOnWriteArrayList<>();
    private final Map<String, List<TrainingProgressEvent>> history = new ConcurrentHashMap<>();
    private static final int HISTORY_CAP = 4096;

    public void subscribe(String runId, Listener listener) {
        byRun.computeIfAbsent(runId, k -> new CopyOnWriteArrayList<>()).add(listener);
    }

    public void subscribeAll(Listener listener) {
        global.add(listener);
    }

    public void unsubscribe(String runId, Listener listener) {
        CopyOnWriteArrayList<Listener> list = byRun.get(runId);
        if (list != null) list.remove(listener);
    }

    public void publish(TrainingProgressEvent event) {
        if (event == null) return;
        history.computeIfAbsent(event.runId(), k -> new CopyOnWriteArrayList<>());
        List<TrainingProgressEvent> h = history.get(event.runId());
        h.add(event);
        while (h.size() > HISTORY_CAP) h.remove(0);
        CopyOnWriteArrayList<Listener> list = byRun.get(event.runId());
        if (list != null) {
            for (Listener l : list) {
                try { l.accept(event); } catch (Throwable ignored) {}
            }
        }
        for (Listener l : global) {
            try { l.accept(event); } catch (Throwable ignored) {}
        }
    }

    public List<TrainingProgressEvent> history(String runId) {
        List<TrainingProgressEvent> h = history.get(runId);
        return h == null ? List.of() : List.copyOf(h);
    }

    public void clear(String runId) {
        history.remove(runId);
        byRun.remove(runId);
    }
}
