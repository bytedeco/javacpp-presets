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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** Model hyperparameters parsed from GGUF metadata (llama.cpp style). */
public final class LlamaHParams {
    private final LlamaArchitecture architecture;
    private final String name;
    private final int nVocab;
    private final int nEmbd;
    private final int nLayer;
    private final int nFF;
    private final int nHead;
    private final int nHeadKv;
    private final int nCtxTrain;
    private final int nRot;
    private final float ropeFreqBase;
    private final float ropeFreqScale;
    private final float rmsNormEps;
    private final int expertCount;
    private final int expertUsedCount;
    private final Map<String, Object> raw;

    private LlamaHParams(Builder b) {
        this.architecture = Objects.requireNonNull(b.architecture);
        this.name = b.name != null ? b.name : "unknown";
        this.nVocab = Math.max(1, b.nVocab);
        this.nEmbd = Math.max(1, b.nEmbd);
        this.nLayer = Math.max(1, b.nLayer);
        this.nFF = b.nFF > 0 ? b.nFF : b.nEmbd * 4;
        this.nHead = Math.max(1, b.nHead);
        this.nHeadKv = b.nHeadKv > 0 ? b.nHeadKv : b.nHead;
        this.nCtxTrain = b.nCtxTrain > 0 ? b.nCtxTrain : 2048;
        this.nRot = b.nRot > 0 ? b.nRot : Math.max(1, b.nEmbd / Math.max(1, b.nHead));
        this.ropeFreqBase = b.ropeFreqBase > 0 ? b.ropeFreqBase : 10000f;
        this.ropeFreqScale = b.ropeFreqScale > 0 ? b.ropeFreqScale : 1f;
        this.rmsNormEps = b.rmsNormEps > 0 ? b.rmsNormEps : 1e-5f;
        this.expertCount = Math.max(0, b.expertCount);
        this.expertUsedCount = Math.max(0, b.expertUsedCount);
        this.raw = Map.copyOf(b.raw);
    }

    public static Builder builder() { return new Builder(); }

    /** Tiny GPT-2-ish defaults for synthetic offline GGUF / unit tests. */
    public static LlamaHParams tiny() {
        return builder()
                .architecture(LlamaArchitecture.GPT2)
                .name("studio-tiny")
                .nVocab(256)
                .nEmbd(64)
                .nLayer(2)
                .nFF(128)
                .nHead(4)
                .nHeadKv(4)
                .nCtxTrain(128)
                .nRot(16)
                .build();
    }

    public LlamaArchitecture architecture() { return architecture; }
    public String name() { return name; }
    public int nVocab() { return nVocab; }
    public int nEmbd() { return nEmbd; }
    public int nLayer() { return nLayer; }
    public int nFF() { return nFF; }
    public int nHead() { return nHead; }
    public int nHeadKv() { return nHeadKv; }
    public int nCtxTrain() { return nCtxTrain; }
    public int nRot() { return nRot; }
    public float ropeFreqBase() { return ropeFreqBase; }
    public float ropeFreqScale() { return ropeFreqScale; }
    public float rmsNormEps() { return rmsNormEps; }
    public int expertCount() { return expertCount; }
    public int expertUsedCount() { return expertUsedCount; }
    public int headDim() { return Math.max(1, nEmbd / nHead); }
    public boolean isMoe() { return expertCount > 1; }
    public Map<String, Object> raw() { return raw; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("architecture", architecture.name());
        m.put("name", name);
        m.put("n_vocab", nVocab);
        m.put("n_embd", nEmbd);
        m.put("n_layer", nLayer);
        m.put("n_ff", nFF);
        m.put("n_head", nHead);
        m.put("n_head_kv", nHeadKv);
        m.put("n_ctx_train", nCtxTrain);
        m.put("n_rot", nRot);
        m.put("rope_freq_base", ropeFreqBase);
        m.put("rope_freq_scale", ropeFreqScale);
        m.put("rms_norm_eps", rmsNormEps);
        m.put("expert_count", expertCount);
        m.put("expert_used_count", expertUsedCount);
        return m;
    }

    public static final class Builder {
        private LlamaArchitecture architecture = LlamaArchitecture.LLAMA;
        private String name;
        private int nVocab = 32000;
        private int nEmbd = 4096;
        private int nLayer = 32;
        private int nFF = 11008;
        private int nHead = 32;
        private int nHeadKv = 32;
        private int nCtxTrain = 2048;
        private int nRot = 128;
        private float ropeFreqBase = 10000f;
        private float ropeFreqScale = 1f;
        private float rmsNormEps = 1e-5f;
        private int expertCount;
        private int expertUsedCount;
        private Map<String, Object> raw = Map.of();

        public Builder architecture(LlamaArchitecture v) { this.architecture = v; return this; }
        public Builder name(String v) { this.name = v; return this; }
        public Builder nVocab(int v) { this.nVocab = v; return this; }
        public Builder nEmbd(int v) { this.nEmbd = v; return this; }
        public Builder nLayer(int v) { this.nLayer = v; return this; }
        public Builder nFF(int v) { this.nFF = v; return this; }
        public Builder nHead(int v) { this.nHead = v; return this; }
        public Builder nHeadKv(int v) { this.nHeadKv = v; return this; }
        public Builder nCtxTrain(int v) { this.nCtxTrain = v; return this; }
        public Builder nRot(int v) { this.nRot = v; return this; }
        public Builder ropeFreqBase(float v) { this.ropeFreqBase = v; return this; }
        public Builder ropeFreqScale(float v) { this.ropeFreqScale = v; return this; }
        public Builder rmsNormEps(float v) { this.rmsNormEps = v; return this; }
        public Builder expertCount(int v) { this.expertCount = v; return this; }
        public Builder expertUsedCount(int v) { this.expertUsedCount = v; return this; }
        public Builder raw(Map<String, Object> v) { this.raw = v != null ? v : Map.of(); return this; }
        public LlamaHParams build() { return new LlamaHParams(this); }
    }
}
