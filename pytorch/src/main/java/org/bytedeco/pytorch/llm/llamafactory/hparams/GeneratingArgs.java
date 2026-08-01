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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.LinkedHashMap;
import java.util.Map;

/** Decoding / sampling args for chat & predict. */
public final class GeneratingArgs {
    private final boolean doSample;
    private final double temperature;
    private final double topP;
    private final int topK;
    private final int numBeams;
    private final int maxLength;
    private final int maxNewTokens;
    private final double repetitionPenalty;
    private final double lengthPenalty;
    private final String defaultSystem;
    private final boolean skipSpecialTokens;

    private GeneratingArgs(Builder b) {
        this.doSample = b.doSample;
        this.temperature = b.temperature;
        this.topP = b.topP;
        this.topK = b.topK;
        this.numBeams = b.numBeams;
        this.maxLength = b.maxLength;
        this.maxNewTokens = b.maxNewTokens;
        this.repetitionPenalty = b.repetitionPenalty;
        this.lengthPenalty = b.lengthPenalty;
        this.defaultSystem = b.defaultSystem;
        this.skipSpecialTokens = b.skipSpecialTokens;
    }

    public boolean doSample() { return doSample; }
    public double temperature() { return temperature; }
    public double topP() { return topP; }
    public int topK() { return topK; }
    public int numBeams() { return numBeams; }
    public int maxLength() { return maxLength; }
    public int maxNewTokens() { return maxNewTokens; }
    public double repetitionPenalty() { return repetitionPenalty; }
    public double lengthPenalty() { return lengthPenalty; }
    public String defaultSystem() { return defaultSystem; }
    public boolean skipSpecialTokens() { return skipSpecialTokens; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "do_sample", doSample);
        HparamsMaps.put(m, "temperature", temperature);
        HparamsMaps.put(m, "top_p", topP);
        HparamsMaps.put(m, "top_k", topK);
        HparamsMaps.put(m, "num_beams", numBeams);
        HparamsMaps.put(m, "max_length", maxLength);
        HparamsMaps.put(m, "max_new_tokens", maxNewTokens);
        HparamsMaps.put(m, "repetition_penalty", repetitionPenalty);
        HparamsMaps.put(m, "length_penalty", lengthPenalty);
        HparamsMaps.put(m, "default_system", defaultSystem);
        HparamsMaps.put(m, "skip_special_tokens", skipSpecialTokens);
        return m;
    }

    public static GeneratingArgs defaults() { return builder().build(); }

    public static GeneratingArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        b.doSample(HparamsMaps.bool(m, b.doSample, "do_sample"));
        b.temperature(HparamsMaps.dbl(m, b.temperature, "temperature"));
        b.topP(HparamsMaps.dbl(m, b.topP, "top_p"));
        b.topK(HparamsMaps.integer(m, b.topK, "top_k"));
        b.numBeams(HparamsMaps.integer(m, b.numBeams, "num_beams"));
        b.maxLength(HparamsMaps.integer(m, b.maxLength, "max_length"));
        b.maxNewTokens(HparamsMaps.integer(m, b.maxNewTokens, "max_new_tokens"));
        b.repetitionPenalty(HparamsMaps.dbl(m, b.repetitionPenalty, "repetition_penalty"));
        b.lengthPenalty(HparamsMaps.dbl(m, b.lengthPenalty, "length_penalty"));
        b.defaultSystem(HparamsMaps.strOrNull(m, "default_system"));
        b.skipSpecialTokens(HparamsMaps.bool(m, b.skipSpecialTokens, "skip_special_tokens"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private boolean doSample = true;
        private double temperature = 0.95;
        private double topP = 0.7;
        private int topK = 50;
        private int numBeams = 1;
        private int maxLength;
        private int maxNewTokens = 1024;
        private double repetitionPenalty = 1.0;
        private double lengthPenalty = 1.0;
        private String defaultSystem;
        private boolean skipSpecialTokens = true;

        public Builder doSample(boolean v) { this.doSample = v; return this; }
        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topP(double v) { this.topP = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder numBeams(int v) { this.numBeams = v; return this; }
        public Builder maxLength(int v) { this.maxLength = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder repetitionPenalty(double v) { this.repetitionPenalty = v; return this; }
        public Builder lengthPenalty(double v) { this.lengthPenalty = v; return this; }
        public Builder defaultSystem(String v) { this.defaultSystem = v; return this; }
        public Builder skipSpecialTokens(boolean v) { this.skipSpecialTokens = v; return this; }
        public GeneratingArgs build() { return new GeneratingArgs(this); }
    }
}
