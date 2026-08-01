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

import java.util.Objects;

/** llama_context_params equivalent (subset used by enterprise runtime). */
public final class LlamaContextParams {
    private final int nCtx;
    private final int nBatch;
    private final int nUbBatch;
    private final int nThreads;
    private final int nThreadsBatch;
    private final boolean embeddings;
    private final boolean offloadKqv;
    private final float ropeFreqBase;
    private final float ropeFreqScale;

    private LlamaContextParams(Builder b) {
        this.nCtx = Math.max(8, b.nCtx);
        this.nBatch = Math.max(1, b.nBatch);
        this.nUbBatch = Math.max(1, b.nUbBatch);
        this.nThreads = b.nThreads > 0 ? b.nThreads : Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
        this.nThreadsBatch = b.nThreadsBatch > 0 ? b.nThreadsBatch : this.nThreads;
        this.embeddings = b.embeddings;
        this.offloadKqv = b.offloadKqv;
        this.ropeFreqBase = b.ropeFreqBase;
        this.ropeFreqScale = b.ropeFreqScale > 0 ? b.ropeFreqScale : 1f;
    }

    public static Builder builder() { return new Builder(); }

    public static LlamaContextParams fromRuntime(LlamaRuntimeConfig cfg, LlamaHParams hp) {
        Objects.requireNonNull(cfg);
        Builder b = builder()
                .nCtx(cfg.nCtx())
                .nBatch(cfg.nBatch())
                .nUbBatch(cfg.nUbBatch())
                .nThreads(cfg.nThreads())
                .offloadKqv(cfg.offloadKqv());
        if (hp != null) {
            b.ropeFreqBase(hp.ropeFreqBase()).ropeFreqScale(hp.ropeFreqScale());
            if (cfg.nCtx() <= 0) b.nCtx(hp.nCtxTrain());
        }
        return b.build();
    }

    public int nCtx() { return nCtx; }
    public int nBatch() { return nBatch; }
    public int nUbBatch() { return nUbBatch; }
    public int nThreads() { return nThreads; }
    public int nThreadsBatch() { return nThreadsBatch; }
    public boolean embeddings() { return embeddings; }
    public boolean offloadKqv() { return offloadKqv; }
    public float ropeFreqBase() { return ropeFreqBase; }
    public float ropeFreqScale() { return ropeFreqScale; }

    public static final class Builder {
        private int nCtx = 2048;
        private int nBatch = 512;
        private int nUbBatch = 512;
        private int nThreads;
        private int nThreadsBatch;
        private boolean embeddings;
        private boolean offloadKqv = true;
        private float ropeFreqBase = 10000f;
        private float ropeFreqScale = 1f;

        public Builder nCtx(int v) { this.nCtx = v; return this; }
        public Builder nBatch(int v) { this.nBatch = v; return this; }
        public Builder nUbBatch(int v) { this.nUbBatch = v; return this; }
        public Builder nThreads(int v) { this.nThreads = v; return this; }
        public Builder nThreadsBatch(int v) { this.nThreadsBatch = v; return this; }
        public Builder embeddings(boolean v) { this.embeddings = v; return this; }
        public Builder offloadKqv(boolean v) { this.offloadKqv = v; return this; }
        public Builder ropeFreqBase(float v) { this.ropeFreqBase = v; return this; }
        public Builder ropeFreqScale(float v) { this.ropeFreqScale = v; return this; }
        public LlamaContextParams build() { return new LlamaContextParams(this); }
    }
}
