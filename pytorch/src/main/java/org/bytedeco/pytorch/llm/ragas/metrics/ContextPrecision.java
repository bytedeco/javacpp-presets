/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.ragas.metrics;

import org.bytedeco.pytorch.llm.ragas.dataset.SingleTurnSample;
import org.bytedeco.pytorch.llm.ragas.llms.LlmJudge;
import org.bytedeco.pytorch.llm.ragas.llms.HeuristicJudge;
import java.util.List;

public final class ContextPrecision implements Metric {
    @Override public String name() { return "context_precision"; }

    @Override
    public double score(SingleTurnSample s, LlmJudge judge) {
        String ref = s.reference();
        List<String> ctxs = s.retrievedContexts();
        if (ctxs == null || ctxs.isEmpty()) return 0.0;
        if (ref == null || ref.isEmpty()) return 0.5;
        int hit = 0;
        for (String c : ctxs) {
            if (HeuristicJudge.jaccard(c, ref) > 0.1) hit++;
        }
        return (double) hit / ctxs.size();
    }
}
