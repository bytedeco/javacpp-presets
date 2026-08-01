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

import org.bytedeco.pytorch.llm.unsloth.studio.model.ModelCard;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Stream;

/** Resolves whether a path is GGUF / safetensors / peft adapter. */
public final class LocalModelResolver {

    public enum Kind { SAFETENSORS, GGUF, PEFT_ADAPTER, PYTORCH_BIN, DIRECTORY, UNKNOWN }

    private LocalModelResolver() {}

    public static Kind kindOf(Path path) {
        if (path == null || !Files.exists(path)) return Kind.UNKNOWN;
        if (Files.isRegularFile(path)) {
            String n = path.getFileName().toString().toLowerCase(Locale.ROOT);
            if (n.endsWith(".gguf")) return Kind.GGUF;
            if (n.endsWith(".safetensors")) return Kind.SAFETENSORS;
            if (n.endsWith(".bin") || n.endsWith(".pt") || n.endsWith(".pth")) return Kind.PYTORCH_BIN;
            return Kind.UNKNOWN;
        }
        // directory
        try (Stream<Path> s = Files.list(path)) {
            boolean hasAdapter = Files.exists(path.resolve("adapter_config.json"))
                    || Files.exists(path.resolve("adapter_model.safetensors"));
            if (hasAdapter) return Kind.PEFT_ADAPTER;
            boolean hasGguf = s.anyMatch(p -> p.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".gguf"));
            if (hasGguf) return Kind.GGUF;
        } catch (Exception ignored) {}
        if (Files.exists(path.resolve("model.safetensors"))
                || Files.exists(path.resolve("model.safetensors.index.json"))
                || Files.exists(path.resolve("config.json"))) {
            return Kind.SAFETENSORS;
        }
        return Kind.DIRECTORY;
    }

    public static Kind kindOf(ModelCard card) {
        Optional<Path> lp = card.localPath();
        if (lp.isPresent()) return kindOf(lp.get());
        String id = card.id().toLowerCase(Locale.ROOT);
        if (id.contains("gguf")) return Kind.GGUF;
        return Kind.SAFETENSORS;
    }
}
