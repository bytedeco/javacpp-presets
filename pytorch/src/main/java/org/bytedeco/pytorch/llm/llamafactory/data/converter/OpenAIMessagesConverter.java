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
package org.bytedeco.pytorch.llm.llamafactory.data.converter;

import org.bytedeco.pytorch.llm.llamafactory.data.template.Formatter;
import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * OpenAI Chat Completions style {@code messages:[{role,content},…]} converter.
 */
public final class OpenAIMessagesConverter {

    private final Template template;

    public OpenAIMessagesConverter(Template template) {
        this.template = Objects.requireNonNull(template, "template");
    }

    public OpenAIMessagesConverter(String templateName) {
        this(TemplateRegistry.getOrDefault(templateName));
    }

    public static OpenAIMessagesConverter defaults() {
        return new OpenAIMessagesConverter("chatml");
    }

    public Map<String, Object> convert(Map<String, Object> raw) {
        Objects.requireNonNull(raw, "raw");
        List<Template.Message> msgs = Formatter.messagesFromRow(raw);
        String system = null;
        for (Template.Message m : msgs) {
            if ("system".equalsIgnoreCase(m.role())) {
                system = m.content();
                break;
            }
        }
        String[] pr = template.encodeSupervised(msgs, system);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("messages", msgs);
        out.put("prompt", pr[0]);
        out.put("response", pr[1]);
        out.put("text", pr[0] + pr[1]);
        if (system != null) out.put("system", system);
        copyIfPresent(raw, out, "tools");
        copyIfPresent(raw, out, "tool_choice");
        return out;
    }

    public List<Map<String, Object>> convertAll(List<Map<String, Object>> rows) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) out.add(convert(r));
        return out;
    }

    private static void copyIfPresent(Map<String, Object> src, Map<String, Object> dst, String k) {
        if (src.containsKey(k)) dst.put(k, src.get(k));
    }
}
