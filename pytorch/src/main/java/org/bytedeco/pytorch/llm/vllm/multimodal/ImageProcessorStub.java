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
package org.bytedeco.pytorch.llm.vllm.multimodal;

/**
 * Image processor stub — validates input and reserves token budget.
 * Future: integrate a real vision encoder (CLIP-ViT, Qwen2-VL image pixel patch, etc.)
 */
public final class ImageProcessorStub implements MultimodalProcessor {

    private final int placeholderTokenCount;

    public ImageProcessorStub() {
        this(256); // default Vision Transformer token budget
    }

    public ImageProcessorStub(int placeholderTokenCount) {
        this.placeholderTokenCount = placeholderTokenCount;
    }

    @Override
    public int[] process(MultimodalPrompt prompt, java.util.List<java.util.Map<String, String>> messages) {
        // Stub: return placeholder token IDs so the text path can still run.
        // Real implementation would:
        // 1. Decode image (JPEG/PNG) from path or bytes
        // 2. Resize / normalize to pixel values
        // 3. Patch-ify into grid tokens
        // 4. Encode with vision encoder → hidden states
        // 5. Concatenate with text token embeddings via multimodal projector
        int[] placeholders = new int[placeholderTokenCount];
        for (int i = 0; i < placeholderTokenCount; i++) placeholders[i] = -1; // special sentinel
        return placeholders;
    }

    @Override
    public int estimateTokenBudget(MediaInput input) {
        return placeholderTokenCount;
    }
}
