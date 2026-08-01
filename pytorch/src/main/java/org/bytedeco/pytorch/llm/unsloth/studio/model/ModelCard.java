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

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Inventory entry for a searchable / downloadable / local model. */
public final class ModelCard {
    private final String id;
    private final String displayName;
    private final String family;
    private final Path localPath;
    private final boolean local;
    private final boolean quant4bit;
    private final boolean vision;
    private final boolean audio;
    private final boolean embedding;
    private final boolean moe;
    private final long parameterCount;
    private final List<String> ggufVariants;
    private final String chatTemplate;
    private final Map<String, Object> meta;

    private ModelCard(Builder b) {
        this.id = b.id;
        this.displayName = b.displayName != null ? b.displayName : b.id;
        this.family = b.family;
        this.localPath = b.localPath;
        this.local = b.local || b.localPath != null;
        this.quant4bit = b.quant4bit;
        this.vision = b.vision;
        this.audio = b.audio;
        this.embedding = b.embedding;
        this.moe = b.moe;
        this.parameterCount = b.parameterCount;
        this.ggufVariants = List.copyOf(b.ggufVariants);
        this.chatTemplate = b.chatTemplate;
        this.meta = Map.copyOf(b.meta);
    }

    public static Builder builder() { return new Builder(); }

    public String id() { return id; }
    public String displayName() { return displayName; }
    public Optional<String> family() { return Optional.ofNullable(family); }
    public Optional<Path> localPath() { return Optional.ofNullable(localPath); }
    public boolean local() { return local; }
    public boolean quant4bit() { return quant4bit; }
    public boolean vision() { return vision; }
    public boolean audio() { return audio; }
    public boolean embedding() { return embedding; }
    public boolean moe() { return moe; }
    public long parameterCount() { return parameterCount; }
    public List<String> ggufVariants() { return ggufVariants; }
    public Optional<String> chatTemplate() { return Optional.ofNullable(chatTemplate); }
    public Map<String, Object> meta() { return meta; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("display_name", displayName);
        if (family != null) m.put("family", family);
        if (localPath != null) m.put("local_path", localPath.toString());
        m.put("local", local);
        m.put("quant_4bit", quant4bit);
        m.put("vision", vision);
        m.put("audio", audio);
        m.put("embedding", embedding);
        m.put("moe", moe);
        m.put("parameter_count", parameterCount);
        if (!ggufVariants.isEmpty()) m.put("gguf_variants", ggufVariants);
        if (chatTemplate != null) m.put("chat_template", chatTemplate);
        return m;
    }

    public static final class Builder {
        private String id;
        private String displayName;
        private String family;
        private Path localPath;
        private boolean local;
        private boolean quant4bit;
        private boolean vision;
        private boolean audio;
        private boolean embedding;
        private boolean moe;
        private long parameterCount;
        private List<String> ggufVariants = List.of();
        private String chatTemplate;
        private Map<String, Object> meta = Map.of();

        public Builder id(String v) { this.id = v; return this; }
        public Builder displayName(String v) { this.displayName = v; return this; }
        public Builder family(String v) { this.family = v; return this; }
        public Builder localPath(Path v) { this.localPath = v; return this; }
        public Builder local(boolean v) { this.local = v; return this; }
        public Builder quant4bit(boolean v) { this.quant4bit = v; return this; }
        public Builder vision(boolean v) { this.vision = v; return this; }
        public Builder audio(boolean v) { this.audio = v; return this; }
        public Builder embedding(boolean v) { this.embedding = v; return this; }
        public Builder moe(boolean v) { this.moe = v; return this; }
        public Builder parameterCount(long v) { this.parameterCount = v; return this; }
        public Builder ggufVariants(List<String> v) { this.ggufVariants = v != null ? v : List.of(); return this; }
        public Builder chatTemplate(String v) { this.chatTemplate = v; return this; }
        public Builder meta(Map<String, Object> v) { this.meta = v != null ? v : Map.of(); return this; }
        public ModelCard build() { return new ModelCard(this); }
    }
}
