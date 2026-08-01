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

import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingRunRecord;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingStartRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/** In-memory + disk metadata store for training runs (resume support). */
public final class TrainingRunStore {

    private final Path runsDir;
    private final Map<String, TrainingRunRecord> runs = new ConcurrentHashMap<>();

    public TrainingRunStore(Path runsDir) {
        this.runsDir = runsDir;
        try {
            StudioPaths.mkdirs(runsDir);
            loadAll();
        } catch (Exception ignored) {}
    }

    public TrainingRunRecord put(TrainingRunRecord record) {
        runs.put(record.runId(), record);
        try { persist(record); } catch (Exception ignored) {}
        return record;
    }

    public Optional<TrainingRunRecord> get(String runId) {
        return Optional.ofNullable(runs.get(runId));
    }

    public List<TrainingRunRecord> list() {
        return new ArrayList<>(runs.values());
    }

    public void update(TrainingRunRecord record) {
        put(record.toBuilder().updatedAtMs(System.currentTimeMillis()).build());
    }

    private void persist(TrainingRunRecord record) throws IOException {
        Path dir = runsDir.resolve(record.runId());
        StudioPaths.mkdirs(dir);
        Path meta = dir.resolve("run.json");
        Files.writeString(meta, JsonMaps.stringify(record.toMap()), StandardCharsets.UTF_8);
    }

    private void loadAll() throws IOException {
        if (!Files.isDirectory(runsDir)) return;
        try (var stream = Files.list(runsDir)) {
            stream.filter(Files::isDirectory).forEach(dir -> {
                Path meta = dir.resolve("run.json");
                if (!Files.exists(meta)) return;
                try {
                    Map<String, Object> m = JsonMaps.parseObject(Files.readString(meta));
                    String runId = String.valueOf(m.getOrDefault("run_id", dir.getFileName().toString()));
                    TrainingRunRecord.Status status = TrainingRunRecord.Status.QUEUED;
                    try {
                        status = TrainingRunRecord.Status.valueOf(String.valueOf(m.getOrDefault("status", "QUEUED")));
                    } catch (Exception ignored) {}
                    TrainingStartRequest req = null;
                    if (m.get("request") instanceof Map<?, ?> rm) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> rmm = (Map<String, Object>) rm;
                        try { req = TrainingStartRequest.fromMap(rmm); } catch (Exception ignored) {}
                    }
                    TrainingRunRecord rec = TrainingRunRecord.builder()
                            .runId(runId)
                            .projectName(m.get("project_name") != null ? String.valueOf(m.get("project_name")) : null)
                            .request(req)
                            .status(status)
                            .outputDir(m.get("output_dir") != null ? Path.of(String.valueOf(m.get("output_dir"))) : dir)
                            .globalStep(m.get("global_step") instanceof Number n ? n.intValue() : 0)
                            .lastLoss(m.get("last_loss") instanceof Number n ? n.doubleValue() : Double.NaN)
                            .error(m.get("error") != null ? String.valueOf(m.get("error")) : null)
                            .createdAtMs(m.get("created_at_ms") instanceof Number n ? n.longValue() : System.currentTimeMillis())
                            .updatedAtMs(m.get("updated_at_ms") instanceof Number n ? n.longValue() : System.currentTimeMillis())
                            .finishedAtMs(m.get("finished_at_ms") instanceof Number n ? n.longValue() : 0)
                            .build();
                    runs.put(runId, rec);
                } catch (Exception ignored) {}
            });
        }
    }
}
