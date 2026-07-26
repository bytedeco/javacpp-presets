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
package org.bytedeco.pytorch.utils.transformers.generation;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.IntConsumer;


import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.argmax;
import static org.bytedeco.pytorch.global.torch.full;
import static org.bytedeco.pytorch.global.torch.multinomial;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.tensor;
import static org.bytedeco.pytorch.global.torch.topk;

/**
 * Autoregressive token generator (prefill-every-step MVP; KV-cache later).
 *
 * <p>Works with any {@link Module} whose {@code forward(input_ids)} returns logits {@code [B,T,V]}.
 */
public final class Generator {

    private Generator() {}

    public static int[] generate(Module model, int[] promptIds, GenerationConfig gen, int maxContext) {
        return generate(model, promptIds, gen, maxContext, null);
    }

    public static int[] generate(Module model, int[] promptIds, GenerationConfig gen,
                                 int maxContext, IntConsumer onToken) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(promptIds, "promptIds");
        if (gen == null) gen = GenerationConfig.greedy();
        if (maxContext <= 0) maxContext = 2048;

        List<Integer> seq = new ArrayList<>(promptIds.length + gen.maxNewTokens);
        for (int id : promptIds) seq.add(id);

        Set<Integer> eos = new HashSet<>(gen.eosTokenIds);
        boolean wasTraining = model.is_training();
        model.eval();
        try {
            for (int step = 0; step < gen.maxNewTokens; step++) {
                int start = Math.max(0, seq.size() - maxContext);
                long[] cur = new long[seq.size() - start];
                for (int i = 0; i < cur.length; i++) cur[i] = seq.get(start + i);
                Tensor ids = tensor(cur).unsqueeze(0); // [1, T]
                Tensor logits = model.forward(ids);    // [1, T, V]
                Tensor last = logits
                        .slice(1, new LongOptional(logits.size(1) - 1), new LongOptional(logits.size(1)), 1)
                        .squeeze(0).squeeze(0); // [V]

                last = applyRepetitionPenalty(last, seq, gen.repetitionPenalty);
                if (gen.temperature > 0 && Math.abs(gen.temperature - 1.0) > 1e-6) {
                    last = last.div(new Scalar(gen.temperature));
                }

                int next;
                if (gen.doSample && gen.temperature > 0) {
                    if (gen.topK > 0) last = topKFilter(last, gen.topK);
                    if (gen.topP > 0 && gen.topP < 1.0) last = topPFilter(last, gen.topP);
                    Tensor probs = softmax(last, 0L);
                    next = (int) multinomial(probs, 1L).item_long();
                } else {
                    next = (int) argmax(last).item_long();
                }

                seq.add(next);
                if (onToken != null) onToken.accept(next);
                if (gen.eosStop && eos.contains(next)) break;
            }
        } finally {
            if (wasTraining) model.train(true);
        }

        int[] out = new int[seq.size()];
        for (int i = 0; i < seq.size(); i++) out[i] = seq.get(i);
        return out;
    }

    private static Tensor applyRepetitionPenalty(Tensor logits, List<Integer> seq, double penalty) {
        if (penalty <= 0 || Math.abs(penalty - 1.0) < 1e-6) return logits;
        Tensor out = logits.clone();
        Set<Integer> seen = new HashSet<>(seq);
        if (seen.isEmpty()) return out;
        for (int id : seen) {
            if (id < 0 || id >= (int) out.size(0)) continue;
            try {
                Tensor t = out.select(0, id);
                float v = t.item_float();
                float nv = v > 0 ? (float) (v / penalty) : (float) (v * penalty);
                t.fill_(new Scalar(nv));
            } catch (Throwable ignored) {
                // ignore if select/fill unavailable on this build
            }
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
                .squeeze()
                .item_float();
        Tensor negInf = full(new long[]{V}, new Scalar(-1e9f));
        Tensor mask = logits.gt(new Scalar(threshold - 1e-6f)).to(ScalarType.Float);
        Tensor ones = full(new long[]{V}, new Scalar(1.0f));
        return logits.mul(mask).add(negInf.mul(ones.sub(mask)));
    }

    /** Nucleus (top-p) filtering on 1-D logits. */
    private static Tensor topPFilter(Tensor logits, double topP) {
        long V = logits.size(0);
        if (topP <= 0 || topP >= 1.0) return logits;
        Tensor probs = softmax(logits, 0L);
        var sorted = topk(probs, (int) V); // descending
        Tensor sortedProbs = sorted.get0();
        Tensor sortedIdx = sorted.get1();
        // cumulative
        float cum = 0f;
        int cutoff = (int) V;
        for (int i = 0; i < V; i++) {
            cum += sortedProbs.select(0, i).item_float();
            if (cum >= topP) {
                cutoff = i + 1;
                break;
            }
        }
        Tensor negInf = full(new long[]{V}, new Scalar(-1e9f));
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
