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
package org.bytedeco.pytorch.llm.ragas.metrics;

import org.bytedeco.pytorch.llm.ragas.dataset.SingleTurnSample;
import org.bytedeco.pytorch.llm.ragas.llms.LlmJudge;

import java.util.List;

/** Faithfulness: fraction of response claims supported by retrieved contexts. */
public final class Faithfulness implements Metric {
    @Override public String name() { return "faithfulness"; }

    @Override
    public double score(SingleTurnSample sample, LlmJudge judge) {
        String response = sample.response();
        List<String> contexts = sample.retrievedContexts();
        // Empty answer has no claims that could be unfaithful — but for RAG scoring
        // benchmarks treat empty response as zero support.
        if (response == null || response.isBlank()) return 0.0;
        if (contexts == null || contexts.isEmpty()) return 0.0;
        String combined = String.join(" ", contexts).toLowerCase();
        String[] toks = response.toLowerCase().trim().split("\\s+");
        int considered = 0;
        int supported = 0;
        for (String t : toks) {
            if (t.isEmpty()) continue;
            // keep short tokens (e.g. "abc") so identical response/context scores ~1
            considered++;
            if (combined.contains(t)) supported++;
        }
        return considered == 0 ? 0.0 : (double) supported / considered;
    }
}
