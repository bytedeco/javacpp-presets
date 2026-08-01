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

package org.bytedeco.pytorch.llm.unsloth.studio.inference.sandbox;

import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Safe default sandbox: only evaluates simple arithmetic expressions.
 * Does <strong>not</strong> spawn processes or eval arbitrary scripts.
 */
public final class InProcessSandbox implements CodeSandbox {

    private static final Pattern ARITH = Pattern.compile(
            "^\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*([+\\-*/])\\s*([+-]?\\d+(?:\\.\\d+)?)\\s*$");

    @Override
    public String execute(String code) {
        Map<String, Object> out = new LinkedHashMap<>();
        if (code == null || code.isBlank()) {
            out.put("ok", false);
            out.put("error", "empty code");
            return JsonMaps.stringify(out);
        }
        String trimmed = code.trim();
        Matcher m = ARITH.matcher(trimmed);
        if (m.matches()) {
            double a = Double.parseDouble(m.group(1));
            String op = m.group(2);
            double b = Double.parseDouble(m.group(3));
            double r = switch (op) {
                case "+" -> a + b;
                case "-" -> a - b;
                case "*" -> a * b;
                case "/" -> b == 0 ? Double.NaN : a / b;
                default -> Double.NaN;
            };
            out.put("ok", true);
            out.put("result", r);
            out.put("expression", trimmed);
            return JsonMaps.stringify(out);
        }
        out.put("ok", false);
        out.put("error", "Only simple arithmetic 'a + b' is allowed in default sandbox");
        out.put("code_preview", trimmed.length() > 120 ? trimmed.substring(0, 120) : trimmed);
        return JsonMaps.stringify(out);
    }
}
