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

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/** Auto inference settings per model family (temperature / top_p / max_tokens). */
public final class SamplingDefaults {

    private SamplingDefaults() {}

    public static ChatCompletionRequest apply(String modelId, ChatCompletionRequest req) {
        Map<String, Double> d = forModel(modelId);
        // Only fill if request uses "unset-ish" defaults — we always return a copy with clamps.
        double temp = clamp(req.temperature(), 0.0, 2.0);
        double topP = clamp(req.topP(), 0.0, 1.0);
        int maxTok = Math.max(1, Math.min(req.maxTokens(), 128_000));
        if (req.temperature() == 0.7 && d.containsKey("temperature")) {
            temp = d.get("temperature");
        }
        if (req.topP() == 0.95 && d.containsKey("top_p")) {
            topP = d.get("top_p");
        }
        return ChatCompletionRequest.builder()
                .model(req.model().orElse(modelId))
                .messages(req.messages())
                .temperature(temp)
                .topP(topP)
                .maxTokens(maxTok)
                .stream(req.stream())
                .presencePenalty(req.presencePenalty())
                .frequencyPenalty(req.frequencyPenalty())
                .tools(req.tools())
                .toolChoice(req.toolChoice().orElse(null))
                .extra(req.extra())
                .build();
    }

    public static Map<String, Double> forModel(String modelId) {
        Map<String, Double> m = new LinkedHashMap<>();
        String s = modelId == null ? "" : modelId.toLowerCase(Locale.ROOT);
        if (s.contains("code") || s.contains("coder")) {
            m.put("temperature", 0.2);
            m.put("top_p", 0.95);
        } else if (s.contains("think") || s.contains("r1") || s.contains("reasoning")) {
            m.put("temperature", 0.6);
            m.put("top_p", 0.95);
        } else if (s.contains("instruct") || s.contains("-it")) {
            m.put("temperature", 0.7);
            m.put("top_p", 0.9);
        } else {
            m.put("temperature", 0.7);
            m.put("top_p", 0.95);
        }
        return m;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
