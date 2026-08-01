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

package org.bytedeco.pytorch.llm.unsloth.studio.model;

import org.bytedeco.pytorch.llm.unsloth.studio.util.Validate;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public final class ExportRequest {
    private final String checkpointPath;
    private final ExportFormat format;
    private final String saveDirectory;
    private final boolean loadIn4bit;
    private final int maxSeqLength;
    private final boolean pushToHub;
    private final String hubModelId;
    private final String hfToken;
    private final String ggufQuant;
    private final Map<String, Object> extra;

    private ExportRequest(Builder b) {
        this.checkpointPath = Objects.requireNonNull(b.checkpointPath, "checkpoint_path");
        Validate.requireNonBlank("checkpoint_path", b.checkpointPath);
        this.format = Objects.requireNonNull(b.format, "format");
        this.saveDirectory = Validate.saveDirectory(b.saveDirectory);
        this.loadIn4bit = b.loadIn4bit;
        this.maxSeqLength = Validate.maxSeqLength(b.maxSeqLength);
        this.pushToHub = b.pushToHub;
        this.hubModelId = b.hubModelId;
        this.hfToken = b.hfToken;
        this.ggufQuant = b.ggufQuant;
        this.extra = Map.copyOf(b.extra);
    }

    public static Builder builder() { return new Builder(); }

    public String checkpointPath() { return checkpointPath; }
    public ExportFormat format() { return format; }
    public String saveDirectory() { return saveDirectory; }
    public boolean loadIn4bit() { return loadIn4bit; }
    public int maxSeqLength() { return maxSeqLength; }
    public boolean pushToHub() { return pushToHub; }
    public Optional<String> hubModelId() { return Optional.ofNullable(hubModelId); }
    public Optional<String> hfToken() { return Optional.ofNullable(hfToken); }
    public Optional<String> ggufQuant() { return Optional.ofNullable(ggufQuant); }
    public Map<String, Object> extra() { return extra; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("checkpoint_path", checkpointPath);
        m.put("format", format.name());
        m.put("save_directory", saveDirectory);
        m.put("load_in_4bit", loadIn4bit);
        m.put("max_seq_length", maxSeqLength);
        m.put("push_to_hub", pushToHub);
        if (hubModelId != null) m.put("hub_model_id", hubModelId);
        if (ggufQuant != null) m.put("gguf_quant", ggufQuant);
        return m;
    }

    public static final class Builder {
        private String checkpointPath;
        private ExportFormat format = ExportFormat.SAFETENSORS_16BIT;
        private String saveDirectory = "exports/default";
        private boolean loadIn4bit = true;
        private int maxSeqLength = 2048;
        private boolean pushToHub = false;
        private String hubModelId;
        private String hfToken;
        private String ggufQuant;
        private Map<String, Object> extra = Map.of();

        public Builder checkpointPath(String v) { this.checkpointPath = v; return this; }
        public Builder format(ExportFormat v) { this.format = v; return this; }
        public Builder format(String v) { this.format = ExportFormat.fromLabel(v); return this; }
        public Builder saveDirectory(String v) { this.saveDirectory = v; return this; }
        public Builder loadIn4bit(boolean v) { this.loadIn4bit = v; return this; }
        public Builder maxSeqLength(int v) { this.maxSeqLength = v; return this; }
        public Builder pushToHub(boolean v) { this.pushToHub = v; return this; }
        public Builder hubModelId(String v) { this.hubModelId = v; return this; }
        public Builder hfToken(String v) { this.hfToken = v; return this; }
        public Builder ggufQuant(String v) { this.ggufQuant = v; return this; }
        public Builder extra(Map<String, Object> v) { this.extra = v != null ? v : Map.of(); return this; }
        public ExportRequest build() { return new ExportRequest(this); }
    }
}
