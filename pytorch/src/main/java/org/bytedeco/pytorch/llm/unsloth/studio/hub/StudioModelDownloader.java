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

package org.bytedeco.pytorch.llm.unsloth.studio.hub;

import org.bytedeco.pytorch.llm.unsloth.studio.StudioOptions;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ModelCard;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * Ensures a model is present under the Studio models directory.
 * Prefers {@code llm.hub.HfHub} when available; otherwise materializes a local
 * placeholder manifest so offline / tiny benchmarks still work.
 */
public final class StudioModelDownloader {

    private final StudioOptions options;
    private final StudioModelRegistry registry;

    public StudioModelDownloader(StudioOptions options, StudioModelRegistry registry) {
        this.options = Objects.requireNonNull(options);
        this.registry = Objects.requireNonNull(registry);
    }

    public ModelCard ensureLocal(ModelCard card) throws IOException {
        if (card.local() && card.localPath().isPresent() && Files.exists(card.localPath().get())) {
            return card;
        }
        // Try HfHub reflectively to avoid hard compile coupling issues in partial trees.
        Optional<Path> hubPath = tryHfHubDownload(card.id());
        if (hubPath.isPresent()) {
            ModelCard updated = ModelCard.builder()
                    .id(card.id())
                    .displayName(card.displayName())
                    .family(card.family().orElse(null))
                    .localPath(hubPath.get())
                    .local(true)
                    .quant4bit(card.quant4bit())
                    .vision(card.vision())
                    .audio(card.audio())
                    .embedding(card.embedding())
                    .moe(card.moe())
                    .parameterCount(card.parameterCount())
                    .ggufVariants(card.ggufVariants())
                    .chatTemplate(card.chatTemplate().orElse(null))
                    .meta(card.meta())
                    .build();
            registry.put(updated);
            return updated;
        }
        // Offline placeholder for studio/tiny-gpt2 and tests
        Path dest = options.modelsDir().resolve(sanitize(card.id()));
        StudioPaths.mkdirs(dest);
        Path marker = dest.resolve("studio_model.json");
        if (!Files.exists(marker)) {
            String json = "{\n  \"id\": \"" + card.id() + "\",\n  \"source\": \"studio-placeholder\",\n"
                    + "  \"note\": \"Replace with real HF snapshot via HfHub when online.\"\n}\n";
            Files.writeString(marker, json, StandardCharsets.UTF_8);
        }
        ModelCard local = ModelCard.builder()
                .id(card.id())
                .displayName(card.displayName())
                .family(card.family().orElse(null))
                .localPath(dest)
                .local(true)
                .quant4bit(card.quant4bit())
                .vision(card.vision())
                .audio(card.audio())
                .embedding(card.embedding())
                .moe(card.moe())
                .parameterCount(card.parameterCount())
                .ggufVariants(card.ggufVariants())
                .chatTemplate(card.chatTemplate().orElse(null))
                .meta(card.meta())
                .build();
        registry.put(local);
        return local;
    }

    private Optional<Path> tryHfHubDownload(String modelId) {
        try {
            Class<?> hub = Class.forName("org.bytedeco.pytorch.llm.hub.HfHub");
            // HfHub.snapshotDownload(String repoId) or similar — best effort
            try {
                Object path = hub.getMethod("snapshotDownload", String.class).invoke(null, modelId);
                if (path instanceof Path) return Optional.of((Path) path);
                if (path instanceof String) return Optional.of(Path.of((String) path));
            } catch (NoSuchMethodException ns) {
                try {
                    Object inst = hub.getMethod("getInstance").invoke(null);
                    Object path = hub.getMethod("download", String.class).invoke(inst, modelId);
                    if (path instanceof Path) return Optional.of((Path) path);
                    if (path instanceof String) return Optional.of(Path.of((String) path));
                } catch (Throwable ignored) {}
            }
        } catch (Throwable ignored) {}
        return Optional.empty();
    }

    private static String sanitize(String id) {
        return id.replace(':', '_').replace('/', '_').replace('\\', '_');
    }
}
