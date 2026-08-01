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

import org.bytedeco.pytorch.llm.llamacpp.LlamaBackend;
import org.bytedeco.pytorch.llm.llamacpp.studio.StudioGgufRuntimeAdapter;
import org.bytedeco.pytorch.llm.unsloth.studio.hardware.GgufHardwareControls;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;

import java.nio.file.Path;

/**
 * Studio {@link GgufRuntime} backed by enterprise llama.cpp process/in-process engine.
 * Prefers PROCESS_SERVER when {@code llama-server} is on PATH; falls back to IN_PROCESS.
 */
public final class ProcessLlamaCppRuntime implements GgufRuntime {

    private final StudioGgufRuntimeAdapter delegate;

    public ProcessLlamaCppRuntime() {
        this(LlamaBackend.AUTO);
    }

    public ProcessLlamaCppRuntime(LlamaBackend backend) {
        this.delegate = new StudioGgufRuntimeAdapter(backend != null ? backend : LlamaBackend.AUTO);
    }

    @Override
    public void load(Path model, GgufHardwareControls controls) throws Exception {
        delegate.load(model, controls);
    }

    @Override
    public ChatCompletionResponse chat(ChatCompletionRequest request) throws Exception {
        return delegate.chat(request);
    }

    @Override
    public void unload() {
        delegate.unload();
    }

    public StudioGgufRuntimeAdapter delegate() {
        return delegate;
    }
}
