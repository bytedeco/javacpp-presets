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

package org.bytedeco.pytorch.llm.unsloth.studio.webui;

import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingProgressEvent;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingRunRecord;
import org.bytedeco.pytorch.llm.unsloth.studio.observe.LiveGraphBuffer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class BoardState {
    private final Map<String, TrainingRunRecord> runs = new ConcurrentHashMap<>();
    private final Map<String, TrainingProgressEvent> lastEvent = new ConcurrentHashMap<>();
    private final LiveGraphBuffer graphs;

    public BoardState(LiveGraphBuffer graphs) {
        this.graphs = graphs != null ? graphs : new LiveGraphBuffer();
    }

    public LiveGraphBuffer graphs() { return graphs; }

    public void upsertRun(TrainingRunRecord rec) {
        if (rec != null) runs.put(rec.runId(), rec);
    }

    public void onEvent(TrainingProgressEvent ev) {
        if (ev == null) return;
        lastEvent.put(ev.runId(), ev);
        graphs.record(org.bytedeco.pytorch.llm.unsloth.studio.observe.TrainingMetrics.from(ev));
    }

    public List<TrainingRunRecord> runs() { return new ArrayList<>(runs.values()); }

    public Map<String, Object> snapshot() {
        Map<String, Object> m = new LinkedHashMap<>();
        List<Map<String, Object>> rs = new ArrayList<>();
        for (TrainingRunRecord r : runs.values()) rs.add(r.toMap());
        m.put("runs", rs);
        Map<String, Object> lasts = new LinkedHashMap<>();
        for (Map.Entry<String, TrainingProgressEvent> e : lastEvent.entrySet()) {
            lasts.put(e.getKey(), e.getValue().toMap());
        }
        m.put("last_events", lasts);
        return m;
    }
}
