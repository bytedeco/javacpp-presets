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

package org.bytedeco.pytorch.llm.unsloth.studio.inference;

import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage;
import org.bytedeco.pytorch.llm.unsloth.studio.rag.RagPipeline;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Chat with images/audio/PDFs/DOCX by stuffing extracted text (and optional media
 * placeholders) into the prompt. Full vision encoders reuse {@code vllm.multimodal}
 * when the host attaches them.
 */
public final class MultimodalChatSession {

    private final InferenceOrchestrator inference;
    private final RagPipeline rag;
    private final List<Path> attachments = new ArrayList<>();

    public MultimodalChatSession(InferenceOrchestrator inference, RagPipeline rag) {
        this.inference = inference;
        this.rag = rag;
    }

    public MultimodalChatSession attach(Path path) {
        if (path != null) attachments.add(path);
        return this;
    }

    public ChatCompletionResponse chat(String userText) throws Exception {
        String prompt = userText;
        if (!attachments.isEmpty() && rag != null) {
            prompt = rag.augmentUserPrompt(userText, attachments, 4);
        } else if (!attachments.isEmpty()) {
            prompt = userText + "\n\n[attachments=" + attachments.size() + "]";
        }
        return inference.chatCompletions(ChatCompletionRequest.of(
                "You are a multimodal assistant. Use attached document context when present.",
                prompt));
    }

    public ChatCompletionResponse chatMessages(List<ChatMessage> messages) throws Exception {
        return inference.chatCompletions(ChatCompletionRequest.builder().messages(messages).build());
    }
}
