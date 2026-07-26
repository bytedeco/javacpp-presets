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
package org.bytedeco.pytorch.utils.vllm.runner;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.transformers.modeling.CachedForwardResult;
import org.bytedeco.pytorch.utils.transformers.modeling.LlamaForCausalLM;
import org.bytedeco.pytorch.utils.transformers.modeling.Qwen2ForCausalLM;
import org.bytedeco.pytorch.utils.vllm.Sequence;
import org.bytedeco.pytorch.utils.vllm.cache.CacheEngine;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Executes {@code Qwen2ForCausalLM} / {@code LlamaForCausalLM} with incremental KV cache.
 *
 * <p>Prefill: runs full prompt once, stores K/V per token into {@link CacheEngine}.
 * Decode: gathers past K/V, runs T=1 forward on the last generated token, appends new K/V.
 *
 * <p>MVP: one sequence at a time inside the batch (sequential-per-seq correctness).
 * The {@link CacheEngine} holds the per-sequence KV blocks independently.
 */
public final class CausalLmRunner implements ModelRunner {

    private final Module model;
    private final PretrainedConfig config;
    private final CacheEngine cache;

    public CausalLmRunner(Module model, PretrainedConfig config, CacheEngine cache) {
        this.model = model;
        this.config = config;
        this.cache = cache;
    }

    @Override
    public Tensor forwardOne(Sequence seq, long cacheSeqId) {
        if (model instanceof Qwen2ForCausalLM qwen) {
            return forwardOneQwen(seq, cacheSeqId, qwen);
        } else if (model instanceof LlamaForCausalLM llama) {
            return forwardOneLlama(seq, cacheSeqId, llama);
        } else {
            throw new UnsupportedOperationException(
                    "Model not Qwen2/Llama: " + model.getClass().getName());
        }
    }

    private Tensor forwardOneQwen(Sequence seq, long cacheSeqId, Qwen2ForCausalLM qwen) {
        int[] inputIds = nextInputIds(seq);
        int computed = seq.numComputedTokens();
        int T = inputIds.length;
        Tensor input = tensor(inputIds).unsqueeze(0); // [1, T]

        Tensor[] pastKs = new Tensor[config.numHiddenLayers()];
        Tensor[] pastVs = new Tensor[config.numHiddenLayers()];
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            Tensor[] kv = cache.gather(cacheSeqId, l);
            pastKs[l] = kv[0];
            pastVs[l] = kv[1];
        }

        CachedForwardResult result = qwen.forwardCached(input, computed, pastKs, pastVs);
        appendNewKv(cacheSeqId, inputIds, result);
        // Mark these tokens as computed so next schedule sees prefill done / decode phase.
        seq.setNumComputedTokens(computed + T);

        Tensor logits = result.logits(); // [1, T, V]
        return logits.select(1, logits.size(1) - 1).squeeze(0); // [V]
    }

    private Tensor forwardOneLlama(Sequence seq, long cacheSeqId, LlamaForCausalLM llama) {
        int[] inputIds = nextInputIds(seq);
        int computed = seq.numComputedTokens();
        int T = inputIds.length;
        Tensor input = tensor(inputIds).unsqueeze(0);

        Tensor[] pastKs = new Tensor[config.numHiddenLayers()];
        Tensor[] pastVs = new Tensor[config.numHiddenLayers()];
        for (int l = 0; l < config.numHiddenLayers(); l++) {
            Tensor[] kv = cache.gather(cacheSeqId, l);
            pastKs[l] = kv[0];
            pastVs[l] = kv[1];
        }

        CachedForwardResult result = llama.forwardCached(input, computed, pastKs, pastVs);
        appendNewKv(cacheSeqId, inputIds, result);
        seq.setNumComputedTokens(computed + T);

        Tensor logits = result.logits();
        return logits.select(1, logits.size(1) - 1).squeeze(0);
    }

    /**
     * Tokens to feed this step:
     * <ul>
     *   <li>Prefill: remaining uncomputed prompt tokens</li>
     *   <li>Decode: last generated token (T=1)</li>
     * </ul>
     */
    private static int[] nextInputIds(Sequence seq) {
        int[] prompt = seq.promptTokenIds();
        int computed = seq.numComputedTokens();
        if (computed < prompt.length) {
            int T = prompt.length - computed;
            int[] ids = new int[T];
            System.arraycopy(prompt, computed, ids, 0, T);
            return ids;
        }
        // Decode: feed the most recently generated token.
        List<Integer> outs = seq.outputTokenIds();
        if (outs.isEmpty()) {
            throw new IllegalStateException(
                    "Decode with empty outputs and prefill done: seq=" + seq);
        }
        return new int[]{outs.get(outs.size() - 1)};
    }

    /**
     * Append per-token K/V across all layers.
     * newKs[l] / newVs[l] are [B, nHeads, T, headDim]; store as [nHeads, headDim] per token.
     */
    private void appendNewKv(long cacheSeqId, int[] inputIds, CachedForwardResult result) {
        int layers = result.numLayers();
        int T = inputIds.length;
        for (int t = 0; t < T; t++) {
            Tensor[] kLayers = new Tensor[layers];
            Tensor[] vLayers = new Tensor[layers];
            for (int l = 0; l < layers; l++) {
                // result.newKs[l]: [1, H, T, D] → select token t → [H, D]
                kLayers[l] = result.newKs[l].select(0, 0).select(1, t);
                vLayers[l] = result.newVs[l].select(0, 0).select(1, t);
            }
            cache.append(cacheSeqId, inputIds[t], kLayers, vLayers);
        }
    }

    @Override
    public List<Tensor> forwardBatch(List<Sequence> prefillSeqs, List<Sequence> decodeSeqs,
                                      long[] cacheSeqIds) {
        // MVP: process sequentially. Each seq has independent cache blocks.
        List<Tensor> results = new ArrayList<>();
        int idx = 0;
        for (Sequence seq : prefillSeqs) {
            results.add(forwardOne(seq, cacheSeqIds[idx++]));
        }
        for (Sequence seq : decodeSeqs) {
            results.add(forwardOne(seq, cacheSeqIds[idx++]));
        }
        return results;
    }

    @Override
    public void close() { /* model/cache lifecycle managed by LLMEngine */ }

    public Module model() { return model; }
    public PretrainedConfig config() { return config; }
}
