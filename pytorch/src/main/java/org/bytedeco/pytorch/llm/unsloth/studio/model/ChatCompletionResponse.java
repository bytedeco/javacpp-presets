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

import org.bytedeco.pytorch.llm.unsloth.studio.util.IdGen;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class ChatCompletionResponse {
    private final String id;
    private final String model;
    private final long created;
    private final List<Choice> choices;
    private final Usage usage;

    public ChatCompletionResponse(String id, String model, long created, List<Choice> choices, Usage usage) {
        this.id = id != null ? id : "chatcmpl-" + IdGen.uuid().substring(0, 12);
        this.model = model != null ? model : "studio-local";
        this.created = created > 0 ? created : System.currentTimeMillis() / 1000L;
        this.choices = List.copyOf(choices);
        this.usage = usage != null ? usage : Usage.ZERO;
    }

    public static ChatCompletionResponse of(String model, String content) {
        Choice c = new Choice(0, ChatMessage.assistant(content), "stop");
        return new ChatCompletionResponse(null, model, 0, List.of(c), new Usage(0, content.length() / 4, content.length() / 4));
    }

    public String id() { return id; }
    public String model() { return model; }
    public long created() { return created; }
    public List<Choice> choices() { return choices; }
    public Usage usage() { return usage; }

    public String firstContent() {
        if (choices.isEmpty()) return "";
        return choices.get(0).message().content();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("object", "chat.completion");
        m.put("created", created);
        m.put("model", model);
        List<Map<String, Object>> ch = new ArrayList<>();
        for (Choice c : choices) ch.add(c.toMap());
        m.put("choices", ch);
        m.put("usage", usage.toMap());
        return m;
    }

    public static final class Choice {
        private final int index;
        private final ChatMessage message;
        private final String finishReason;

        public Choice(int index, ChatMessage message, String finishReason) {
            this.index = index;
            this.message = message;
            this.finishReason = finishReason;
        }

        public int index() { return index; }
        public ChatMessage message() { return message; }
        public String finishReason() { return finishReason; }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("index", index);
            m.put("message", message.toMap());
            m.put("finish_reason", finishReason);
            return m;
        }
    }

    public static final class Usage {
        public static final Usage ZERO = new Usage(0, 0, 0);
        private final int promptTokens;
        private final int completionTokens;
        private final int totalTokens;

        public Usage(int promptTokens, int completionTokens, int totalTokens) {
            this.promptTokens = promptTokens;
            this.completionTokens = completionTokens;
            this.totalTokens = totalTokens > 0 ? totalTokens : promptTokens + completionTokens;
        }

        public int promptTokens() { return promptTokens; }
        public int completionTokens() { return completionTokens; }
        public int totalTokens() { return totalTokens; }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("prompt_tokens", promptTokens);
            m.put("completion_tokens", completionTokens);
            m.put("total_tokens", totalTokens);
            return m;
        }
    }
}
