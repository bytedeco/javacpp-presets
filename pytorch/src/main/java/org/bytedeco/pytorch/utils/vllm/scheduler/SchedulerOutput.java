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
package org.bytedeco.pytorch.utils.vllm.scheduler;

import org.bytedeco.pytorch.utils.vllm.Sequence;

import java.util.Collections;
import java.util.List;

/** Seqs scheduled for prefill (first pass) or decode (incremental) in one engine step. */
public final class SchedulerOutput {

    /** Seqs whose prompt tokens haven't all been processed — run prefill. */
    public final List<Sequence> prefillSeqs;

    /** Seqs already in decode phase — run T=1 decode step. */
    public final List<Sequence> decodeSeqs;

    /** Seqs that finished this step (already marked FINISHED by engine, freed here). */
    public final List<Sequence> finishedSeqs;

    public SchedulerOutput(List<Sequence> prefillSeqs, List<Sequence> decodeSeqs,
                           List<Sequence> finishedSeqs) {
        this.prefillSeqs = prefillSeqs == null ? List.of() : Collections.unmodifiableList(prefillSeqs);
        this.decodeSeqs = decodeSeqs == null ? List.of() : Collections.unmodifiableList(decodeSeqs);
        this.finishedSeqs = finishedSeqs == null ? List.of() : Collections.unmodifiableList(finishedSeqs);
    }

    public static SchedulerOutput empty() {
        return new SchedulerOutput(List.of(), List.of(), List.of());
    }

    public boolean hasWork() {
        return !prefillSeqs.isEmpty() || !decodeSeqs.isEmpty();
    }

    /** Total sequences in this step. */
    public int numSeqs() { return prefillSeqs.size() + decodeSeqs.size(); }

    /** Total tokens to process (prompt tokens for prefill, 1 per seq for decode). */
    public int numTokens() {
        int n = 0;
        for (Sequence s : prefillSeqs) n += s.numUncomputedTokens();
        n += decodeSeqs.size(); // 1 token per decode seq
        return n;
    }

    @Override
    public String toString() {
        return "SchedulerOutput{prefill=" + prefillSeqs.size()
                + ", decode=" + decodeSeqs.size()
                + ", finished=" + finishedSeqs.size()
                + ", tokens=" + numTokens() + "}";
    }
}
