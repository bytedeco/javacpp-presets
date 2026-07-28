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
package org.bytedeco.pytorch.llm.ragas.llms;

import org.bytedeco.pytorch.llm.transformers.CausalLM;

import java.util.Optional;

/** Optional LLM judge wrapping {@link CausalLM}. */
public final class CausalLmJudge implements LlmJudge {
    private final CausalLM model;
    private final int maxNewTokens;

    public CausalLmJudge(CausalLM model) { this(model, 128); }
    public CausalLmJudge(CausalLM model, int maxNewTokens) {
        this.model = model;
        this.maxNewTokens = Math.max(1, maxNewTokens);
    }

    @Override
    public String generate(String prompt) {
        int[] ids = new int[Math.min(prompt.length(), 512)];
        for (int i = 0; i < ids.length; i++) ids[i] = (prompt.charAt(i) % 32000);
        int[] out = model.generate(ids, maxNewTokens);
        StringBuilder sb = new StringBuilder();
        for (int id : out) sb.append((char) (id % 128));
        return sb.toString();
    }

    @Override public Optional<Boolean> extractYesNo(String text) { return Optional.empty(); }
    @Override public float[] embed(String text) { return new float[64]; }
    @Override public boolean available() { return model != null; }
}
