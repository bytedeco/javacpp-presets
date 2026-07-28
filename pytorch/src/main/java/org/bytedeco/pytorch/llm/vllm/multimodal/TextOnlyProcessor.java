/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may not use this copy of the License at
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

import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Real text-only processor using {@link ChatTemplate} + {@link FastTokenizer}.
 */
public final class TextOnlyProcessor implements MultimodalProcessor {

    private final FastTokenizer tokenizer;
    private final ChatTemplate chatTemplate;

    public TextOnlyProcessor(FastTokenizer tokenizer, ChatTemplate chatTemplate) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.chatTemplate = chatTemplate;
    }

    @Override
    public int[] process(MultimodalPrompt prompt, List<Map<String, String>> messages) {
        // Extract text from prompt parts
        StringBuilder sb = new StringBuilder();
        for (MediaInput part : prompt.parts()) {
            if (part.type == MediaType.TEXT && part.text != null) {
                sb.append(part.text);
            }
        }
        String text = sb.toString();

        // Apply chat template if messages provided
        if (messages != null && !messages.isEmpty()) {
            String prompt2 = chatTemplate.apply(messages, true);
            Encoding enc = tokenizer.encode(prompt2, true);
            return enc.ids();
        }

        // Plain text tokenization
        Encoding enc = tokenizer.encode(text, true);
        return enc.ids();
    }

    @Override
    public int estimateTokenBudget(MediaInput input) {
        return 0; // real tokenizer
    }
}
