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
package org.bytedeco.pytorch.llm.ktransformers.config;

/**
 * Inference-serving knobs for {@code KtInferenceEngine}.
 *
 * <p>Covers multi-concurrency, multi-GPU, prefill chunking and decode limits
 * corresponding to upstream kt-kernel Inference Quick Start options.
 */
public final class KtInferenceConfig {

    private final int maxBatch;
    private final int maxSeqLen;
    private final int maxNewTokens;
    private final int concurrency;
    private final int prefillChunk;
    private final boolean multiGpu;
    private final int tensorParallel;
    private final double temperature;
    private final double topP;
    private final int topK;
    private final long seed;
    private final boolean useMla;
    private final boolean usePagedAttention;

    private KtInferenceConfig(Builder b) {
        if (b.maxBatch < 1 || b.maxSeqLen < 1 || b.maxNewTokens < 1) {
            throw new IllegalArgumentException("maxBatch/maxSeqLen/maxNewTokens must be >= 1");
        }
        if (b.concurrency < 1) {
            throw new IllegalArgumentException("concurrency must be >= 1");
        }
        if (b.prefillChunk < 1) {
            throw new IllegalArgumentException("prefillChunk must be >= 1");
        }
        if (b.tensorParallel < 1) {
            throw new IllegalArgumentException("tensorParallel must be >= 1");
        }
        this.maxBatch = b.maxBatch;
        this.maxSeqLen = b.maxSeqLen;
        this.maxNewTokens = b.maxNewTokens;
        this.concurrency = b.concurrency;
        this.prefillChunk = b.prefillChunk;
        this.multiGpu = b.multiGpu;
        this.tensorParallel = b.tensorParallel;
        this.temperature = b.temperature;
        this.topP = b.topP;
        this.topK = b.topK;
        this.seed = b.seed;
        this.useMla = b.useMla;
        this.usePagedAttention = b.usePagedAttention;
    }

    public int maxBatch() { return maxBatch; }
    public int maxSeqLen() { return maxSeqLen; }
    public int maxNewTokens() { return maxNewTokens; }
    public int concurrency() { return concurrency; }
    public int prefillChunk() { return prefillChunk; }
    public boolean multiGpu() { return multiGpu; }
    public int tensorParallel() { return tensorParallel; }
    public double temperature() { return temperature; }
    public double topP() { return topP; }
    public int topK() { return topK; }
    public long seed() { return seed; }
    public boolean useMla() { return useMla; }
    public boolean usePagedAttention() { return usePagedAttention; }

    public static Builder builder() { return new Builder(); }

    public static KtInferenceConfig defaults() {
        return builder().build();
    }

    public static final class Builder {
        private int maxBatch = 8;
        private int maxSeqLen = 8192;
        private int maxNewTokens = 512;
        private int concurrency = 4;
        private int prefillChunk = 512;
        private boolean multiGpu = false;
        private int tensorParallel = 1;
        private double temperature = 0.7;
        private double topP = 0.95;
        private int topK = 0;
        private long seed = 42L;
        private boolean useMla = false;
        private boolean usePagedAttention = true;

        public Builder maxBatch(int v) { this.maxBatch = v; return this; }
        public Builder maxSeqLen(int v) { this.maxSeqLen = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder concurrency(int v) { this.concurrency = v; return this; }
        public Builder prefillChunk(int v) { this.prefillChunk = v; return this; }
        public Builder multiGpu(boolean v) { this.multiGpu = v; return this; }
        public Builder tensorParallel(int v) { this.tensorParallel = v; return this; }
        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topP(double v) { this.topP = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }
        public Builder useMla(boolean v) { this.useMla = v; return this; }
        public Builder usePagedAttention(boolean v) { this.usePagedAttention = v; return this; }

        public KtInferenceConfig build() { return new KtInferenceConfig(this); }
    }
}
