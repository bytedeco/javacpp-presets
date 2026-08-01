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

package org.bytedeco.pytorch.llm.unsloth.studio.inference.gguf;

import org.bytedeco.pytorch.llm.unsloth.studio.hardware.GgufHardwareControls;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Metadata-only GGUF bridge used until a native/runtime SPI is registered.
 * Validates file presence and returns deterministic completions for tests.
 */
public final class JavaGgufBridge implements GgufRuntime {

    private Path model;
    private GgufHardwareControls controls;
    private boolean loaded;

    @Override
    public void load(Path model, GgufHardwareControls controls) throws Exception {
        if (model == null || !Files.exists(model)) {
            throw new IllegalArgumentException("GGUF model not found: " + model);
        }
        this.model = model;
        this.controls = controls != null ? controls : GgufHardwareControls.defaults();
        this.loaded = true;
    }

    @Override
    public ChatCompletionResponse chat(ChatCompletionRequest request) {
        if (!loaded) throw new IllegalStateException("GGUF not loaded");
        String last = request.messages().isEmpty() ? ""
                : request.messages().get(request.messages().size() - 1).content();
        return ChatCompletionResponse.of("gguf:" + model.getFileName(),
                "[gguf-bridge] " + last + " | controls=" + controls.toRunnerArgs());
    }

    public Map<String, String> runnerArgs() {
        return controls != null ? controls.toRunnerArgs() : Map.of();
    }

    @Override
    public void unload() {
        loaded = false;
        model = null;
    }
}
