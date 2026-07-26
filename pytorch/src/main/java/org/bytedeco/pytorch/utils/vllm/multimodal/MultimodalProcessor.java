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
package org.bytedeco.pytorch.utils.vllm.multimodal;

import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.transformers.tokenization.ChatTemplate;

import java.util.List;

/**
 * Processes a {@link MultimodalPrompt} into token IDs for the engine.
 *
 * <p>Implementations: {@link TextOnlyProcessor} (real), image/audio/video stubs.
 */
public interface MultimodalProcessor {

    /**
     * Convert a multimodal prompt to token IDs for the LLM.
     *
     * @param prompt   the multimodal input
     * @param messages optional chat messages (used by text-only processor)
     * @return token IDs ready for generation
     */
    int[] process(MultimodalPrompt prompt, List<java.util.Map<String, String>> messages);

    /** Estimate how many tokens a media input will consume (for stub processors). */
    default int estimateTokenBudget(MediaInput input) {
        return switch (input.type) {
            case TEXT -> 0; // real tokenizer handles this
            case IMAGE -> 256;   // typical Vision Transformer budget
            case AUDIO -> 512;   // typical audio encoder budget
            case VIDEO -> 1024;  // typical video encoder budget
            case EMBEDDING -> 0; // fused directly
        };
    }
}
