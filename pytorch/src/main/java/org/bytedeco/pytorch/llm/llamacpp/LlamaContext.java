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

package org.bytedeco.pytorch.llm.llamacpp;

import org.bytedeco.pytorch.llm.llamacpp.model.LlamaTransformer;

import java.util.Objects;

/**
 * In-process decode context: holds KV + transformer + last logits.
 * Mirrors llama_context responsibilities for the pure-Java backend.
 */
public final class LlamaContext implements AutoCloseable {

    private final LlamaModel model;
    private final LlamaContextParams params;
    private final LlamaKvCache kv;
    private final LlamaTransformer transformer;
    private final LlamaTokenizer tokenizer;
    private float[] lastLogits;
    private int nPast;

    public LlamaContext(LlamaModel model, LlamaContextParams params) {
        this.model = Objects.requireNonNull(model);
        this.params = params != null ? params : LlamaContextParams.builder().nCtx(model.hparams().nCtxTrain()).build();
        this.kv = new LlamaKvCache(model.hparams(), this.params.nCtx());
        this.transformer = new LlamaTransformer(model, this.params.nCtx());
        this.tokenizer = new LlamaTokenizer(model.hparams(), model.metadata());
        this.nPast = 0;
    }

    public static LlamaContext create(LlamaModel model, LlamaRuntimeConfig runtime) {
        return new LlamaContext(model, LlamaContextParams.fromRuntime(runtime, model.hparams()));
    }

    public LlamaModel model() { return model; }
    public LlamaContextParams params() { return params; }
    public LlamaKvCache kv() { return kv; }
    public LlamaTokenizer tokenizer() { return tokenizer; }
    public int nPast() { return nPast; }
    public float[] lastLogits() { return lastLogits; }

    public void reset() {
        kv.reset();
        nPast = 0;
        lastLogits = null;
    }

    /**
     * Decode a batch; returns logits for the last position that requested them
     * (or the final token). Updates KV / nPast.
     */
    public float[] decode(LlamaBatch batch) throws Exception {
        Objects.requireNonNull(batch, "batch");
        if (batch.nTokens() == 0) throw new IllegalArgumentException("empty batch");
        float[] logits = null;
        for (int i = 0; i < batch.nTokens(); i++) {
            int pos = batch.pos(i);
            logits = transformer.logits(batch.token(i), pos, kv);
            if (batch.logits(i)) {
                lastLogits = logits;
            }
            nPast = kv.nPast();
        }
        if (lastLogits == null) lastLogits = logits;
        return lastLogits;
    }

    /** Prefill prompt tokens; returns logits after last token. */
    public float[] prefill(int[] tokens) throws Exception {
        return decode(LlamaBatch.ofTokens(tokens, nPast, true));
    }

    /** One decode step for a single new token at current nPast. */
    public float[] step(int token) throws Exception {
        LlamaBatch b = new LlamaBatch(1);
        b.add(token, nPast, true);
        return decode(b);
    }

    @Override
    public void close() {
        reset();
    }
}
