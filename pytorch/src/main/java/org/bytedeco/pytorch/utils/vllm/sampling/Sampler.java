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
package org.bytedeco.pytorch.utils.vllm.sampling;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.vllm.SamplingParams;
import org.bytedeco.pytorch.utils.vllm.Sequence;

import java.util.HashSet;
import java.util.Set;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.argmax;
import static org.bytedeco.pytorch.global.torch.full;
import static org.bytedeco.pytorch.global.torch.multinomial;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.topk;

/**
 * Sampling logic extracted from {@link org.bytedeco.pytorch.utils.transformers.generation.Generator}
 * for use inside the engine. Stateless; all mutable state (output tokens) lives in {@link Sequence}.
 */
public final class Sampler {

    private Sampler() {}

    /** Sample one token for a sequence given logits [V] and update the sequence. */
    public static int sample(Tensor logits, Sequence seq) {
        SamplingParams p = seq.samplingParams();
        int vocabSize = (int) logits.size(0);

        // 1. Repetition penalty
        Tensor filtered = applyRepetitionPenalty(logits, seq.allTokenIds(), p.repetitionPenalty);

        // 2. Temperature
        if (p.temperature > 0 && Math.abs(p.temperature - 1.0) > 1e-6) {
            filtered = filtered.div(new Scalar(p.temperature));
        }

        // 3. Top-K
        if (p.topK > 0 && p.topK < vocabSize) {
            filtered = topKFilter(filtered, p.topK);
        }

        // 4. Top-P (nucleus)
        if (p.topP > 0 && p.topP < 1.0) {
            filtered = topPFilter(filtered, p.topP);
        }

        // 5. Sample
        int nextToken;
        if (p.doSample && p.temperature > 0) {
            Tensor probs = softmax(filtered, 0L);
            nextToken = (int) multinomial(probs, 1L).item_long();
        } else {
            nextToken = (int) argmax(filtered).item_long();
        }

        // 6. Append
        seq.appendToken(nextToken);

        // 7. Check stop
        boolean isStop = !p.ignoreEos && p.stopTokenIds.contains(nextToken);
        boolean maxReached = seq.numOutputTokens() >= p.maxTokens;
        if (isStop || maxReached) {
            seq.markFinished(isStop ? "stop" : "length");
        }

        return nextToken;
    }

    private static Tensor applyRepetitionPenalty(Tensor logits, java.util.List<Integer> seq,
                                                  double penalty) {
        if (penalty <= 0 || Math.abs(penalty - 1.0) < 1e-6 || seq.isEmpty()) return logits;
        Tensor out = logits.clone();
        Set<Integer> seen = new HashSet<>(seq);
        for (int id : seen) {
            if (id < 0 || id >= (int) out.size(0)) continue;
            try {
                Tensor t = out.select(0, id);
                float v = t.item_float();
                float nv = v > 0 ? (float) (v / penalty) : (float) (v * penalty);
                t.fill_(new Scalar(nv));
            } catch (Throwable ignored) {}
        }
        return out;
    }

    private static Tensor topKFilter(Tensor logits, int k) {
        long V = logits.size(0);
        if (k <= 0 || k >= V) return logits;
        var top = topk(logits, k);
        Tensor values = top.get0();
        float threshold = values
                .slice(0, new LongOptional(values.size(0) - 1), new LongOptional(values.size(0)), 1)
                .squeeze().item_float();
        Tensor negInf = full(new long[]{V}, new Scalar(-1e9f), new org.bytedeco.pytorch.TensorOptions(ScalarType.Float));
        Tensor mask = logits.gt(new Scalar(threshold - 1e-6f)).to(ScalarType.Float);
        return logits.mul(mask).add(negInf.mul(mask.eq(new Scalar(0f)).to(ScalarType.Float)));
    }

    private static Tensor topPFilter(Tensor logits, double topP) {
        long V = logits.size(0);
        if (topP <= 0 || topP >= 1.0) return logits;
        Tensor probs = softmax(logits, 0L);
        var sorted = topk(probs, (int) V);
        Tensor sortedProbs = sorted.get0();
        Tensor sortedIdx = sorted.get1();
        float cum = 0f;
        int cutoff = (int) V;
        for (int i = 0; i < V; i++) {
            cum += sortedProbs.select(0, i).item_float();
            if (cum >= topP) { cutoff = i + 1; break; }
        }
        Tensor negInf = full(new long[]{V}, new Scalar(-1e9f), new org.bytedeco.pytorch.TensorOptions(ScalarType.Float));
        Tensor out = negInf.clone();
        for (int i = 0; i < cutoff; i++) {
            int idx = (int) sortedIdx.select(0, i).item_long();
            try {
                out.select(0, idx).copy_(logits.select(0, idx));
            } catch (Throwable ignored) {}
        }
        return out;
    }
}
