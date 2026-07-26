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
package org.bytedeco.pytorch.utils.vllm;

/** Runtime budget configuration for the engine. */
public final class EngineConfig {

    public final int maxNumSeqs;
    public final int maxNumBatchedTokens;
    public final int blockSize;
    public final int maxBlocks;
    public final String device;
    public final int numLayers;
    public final int numHeads;
    public final int headDim;
    public final int vocabSize;

    private EngineConfig(Builder b) {
        this.maxNumSeqs = b.maxNumSeqs;
        this.maxNumBatchedTokens = b.maxNumBatchedTokens;
        this.blockSize = b.blockSize;
        this.maxBlocks = b.maxBlocks;
        this.device = b.device;
        this.numLayers = b.numLayers;
        this.numHeads = b.numHeads;
        this.headDim = b.headDim;
        this.vocabSize = b.vocabSize;
    }

    public static Builder builder() { return new Builder(); }

    public static Builder fromPretrainedConfig(
            org.bytedeco.pytorch.utils.transformers.PretrainedConfig cfg) {
        return builder()
                .numLayers(cfg.numHiddenLayers())
                .numHeads(cfg.numAttentionHeads())
                .headDim(cfg.headDim())
                .vocabSize(cfg.vocabSize());
    }

    /** CPU-friendly defaults. */
    public static EngineConfig cpuDefault() {
        return builder()
                .maxNumSeqs(8)
                .maxNumBatchedTokens(512)
                .blockSize(32)
                .maxBlocks(256)
                .device("cpu")
                .build();
    }

    public Builder toBuilder() {
        return builder()
                .maxNumSeqs(maxNumSeqs)
                .maxNumBatchedTokens(maxNumBatchedTokens)
                .blockSize(blockSize)
                .maxBlocks(maxBlocks)
                .device(device)
                .numLayers(numLayers)
                .numHeads(numHeads)
                .headDim(headDim)
                .vocabSize(vocabSize);
    }

    @Override
    public String toString() {
        return "EngineConfig{seqs=" + maxNumSeqs + ", batchTokens=" + maxNumBatchedTokens
                + ", block=" + blockSize + ", blocks=" + maxBlocks + ", device=" + device + "}";
    }

    public static final class Builder {
        private int maxNumSeqs = 8;
        private int maxNumBatchedTokens = 512;
        private int blockSize = 32;
        private int maxBlocks = 256;
        private String device = "cpu";
        private int numLayers = 0;
        private int numHeads = 0;
        private int headDim = 0;
        private int vocabSize = 0;

        public Builder maxNumSeqs(int v) { this.maxNumSeqs = v; return this; }
        public Builder maxNumBatchedTokens(int v) { this.maxNumBatchedTokens = v; return this; }
        public Builder blockSize(int v) { this.blockSize = v; return this; }
        public Builder maxBlocks(int v) { this.maxBlocks = v; return this; }
        public Builder device(String v) { this.device = v == null ? "cpu" : v; return this; }
        public Builder numLayers(int v) { this.numLayers = v; return this; }
        public Builder numHeads(int v) { this.numHeads = v; return this; }
        public Builder headDim(int v) { this.headDim = v; return this; }
        public Builder vocabSize(int v) { this.vocabSize = v; return this; }

        public EngineConfig build() {
            if (maxNumSeqs <= 0) throw new IllegalArgumentException("maxNumSeqs must be > 0");
            if (maxNumBatchedTokens <= 0) throw new IllegalArgumentException("maxNumBatchedTokens must be > 0");
            return new EngineConfig(this);
        }
    }
}
