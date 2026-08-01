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

import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * KTO converter: desirable / undesirable examples with binary tags.
 *
 * <p>Accepts {@code kto_tag}/{@code desirable}/{@code label} plus standard
 * instruction or messages fields.
 */
public final class KtoConverter {

    private final Template template;
    private final AlpacaConverter alpaca;
    private final OpenAIMessagesConverter openai;

    public KtoConverter(Template template) {
        this.template = Objects.requireNonNull(template, "template");
        this.alpaca = new AlpacaConverter(template);
        this.openai = new OpenAIMessagesConverter(template);
    }

    public KtoConverter(String templateName) {
        this(TemplateRegistry.getOrDefault(templateName));
    }

    public static KtoConverter defaults() {
        return new KtoConverter("default");
    }

    public Map<String, Object> convert(Map<String, Object> raw) {
        Objects.requireNonNull(raw, "raw");
        Map<String, Object> base;
        if (raw.containsKey("messages") || raw.containsKey("conversations")) {
            base = openai.convert(raw);
        } else {
            base = alpaca.convert(raw);
        }
        boolean desirable = parseDesirable(raw);
        base.put("desirable", desirable);
        base.put("kto_tags", desirable ? 1 : 0);
        return base;
    }

    public List<Map<String, Object>> convertAll(List<Map<String, Object>> rows) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) out.add(convert(r));
        return out;
    }

    static boolean parseDesirable(Map<String, Object> raw) {
        Object t = raw.get("kto_tags");
        if (t == null) t = raw.get("desirable");
        if (t == null) t = raw.get("label");
        if (t == null) t = raw.get("kto_tag");
        if (t instanceof Boolean b) return b;
        if (t instanceof Number n) return n.intValue() != 0;
        if (t instanceof String s) {
            String lower = s.toLowerCase(Locale.ROOT).trim();
            return switch (lower) {
                case "false", "0", "no", "undesirable", "rejected", "bad" -> false;
                default -> true;
            };
        }
        // default desirable when unspecified (matches common demo datasets)
        return true;
    }
}
