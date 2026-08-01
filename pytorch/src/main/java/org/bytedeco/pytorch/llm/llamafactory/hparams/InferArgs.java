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

/** Inference / OpenAI API serve args. */
public final class InferArgs {
    private final String modelNameOrPath;
    private final String adapterNameOrPath;
    private final String template;
    private final String inferBackend;
    private final String host;
    private final int port;
    private final String apiKey;
    private final int maxConcurrent;
    private final GeneratingArgs generating;
    private final QuantizationMethod quantizationMethod;
    private final boolean flashAttn;
    private final boolean useUnsloth;

    private InferArgs(Builder b) {
        this.modelNameOrPath = b.modelNameOrPath == null ? "gpt2" : b.modelNameOrPath;
        this.adapterNameOrPath = b.adapterNameOrPath;
        this.template = b.template == null ? "default" : b.template;
        this.inferBackend = b.inferBackend == null ? "huggingface" : b.inferBackend;
        this.host = b.host == null ? "0.0.0.0" : b.host;
        this.port = b.port;
        this.apiKey = b.apiKey;
        this.maxConcurrent = b.maxConcurrent;
        this.generating = b.generating == null ? GeneratingArgs.defaults() : b.generating;
        this.quantizationMethod = b.quantizationMethod == null ? QuantizationMethod.NONE : b.quantizationMethod;
        this.flashAttn = b.flashAttn;
        this.useUnsloth = b.useUnsloth;
    }

    public String modelNameOrPath() { return modelNameOrPath; }
    public String adapterNameOrPath() { return adapterNameOrPath; }
    public String template() { return template; }
    public String inferBackend() { return inferBackend; }
    public String host() { return host; }
    public int port() { return port; }
    public String apiKey() { return apiKey; }
    public int maxConcurrent() { return maxConcurrent; }
    public GeneratingArgs generating() { return generating; }
    public QuantizationMethod quantizationMethod() { return quantizationMethod; }
    public boolean flashAttn() { return flashAttn; }
    public boolean useUnsloth() { return useUnsloth; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "model_name_or_path", modelNameOrPath);
        HparamsMaps.put(m, "adapter_name_or_path", adapterNameOrPath);
        HparamsMaps.put(m, "template", template);
        HparamsMaps.put(m, "infer_backend", inferBackend);
        HparamsMaps.put(m, "host", host);
        HparamsMaps.put(m, "port", port);
        HparamsMaps.put(m, "api_key", apiKey);
        HparamsMaps.put(m, "max_concurrent", maxConcurrent);
        HparamsMaps.put(m, "generating", generating.toMap());
        HparamsMaps.put(m, "quantization_method", quantizationMethod.wireName());
        HparamsMaps.put(m, "flash_attn", flashAttn);
        HparamsMaps.put(m, "use_unsloth", useUnsloth);
        return m;
    }

    public static InferArgs defaults() { return builder().build(); }

    public static InferArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        b.modelNameOrPath(HparamsMaps.str(m, b.modelNameOrPath, "model_name_or_path", "model"));
        b.adapterNameOrPath(HparamsMaps.strOrNull(m, "adapter_name_or_path", "adapter_path"));
        b.template(HparamsMaps.str(m, b.template, "template"));
        b.inferBackend(HparamsMaps.str(m, b.inferBackend, "infer_backend"));
        b.host(HparamsMaps.str(m, b.host, "host"));
        b.port(HparamsMaps.integer(m, b.port, "port"));
        b.apiKey(HparamsMaps.strOrNull(m, "api_key"));
        b.maxConcurrent(HparamsMaps.integer(m, b.maxConcurrent, "max_concurrent"));
        Map<String, Object> gen = HparamsMaps.asMap(HparamsMaps.get(m, "generating"));
        if (gen != null) b.generating(GeneratingArgs.fromMap(gen));
        else b.generating(GeneratingArgs.fromMap(m));
        String qm = HparamsMaps.strOrNull(m, "quantization_method");
        if (qm != null) b.quantizationMethod(QuantizationMethod.parse(qm));
        b.flashAttn(HparamsMaps.bool(m, b.flashAttn, "flash_attn"));
        b.useUnsloth(HparamsMaps.bool(m, b.useUnsloth, "use_unsloth"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String modelNameOrPath = "gpt2";
        private String adapterNameOrPath;
        private String template = "default";
        private String inferBackend = "huggingface";
        private String host = "0.0.0.0";
        private int port = 8000;
        private String apiKey;
        private int maxConcurrent = 64;
        private GeneratingArgs generating = GeneratingArgs.defaults();
        private QuantizationMethod quantizationMethod = QuantizationMethod.NONE;
        private boolean flashAttn;
        private boolean useUnsloth;

        public Builder modelNameOrPath(String v) { this.modelNameOrPath = v; return this; }
        public Builder adapterNameOrPath(String v) { this.adapterNameOrPath = v; return this; }
        public Builder template(String v) { this.template = v; return this; }
        public Builder inferBackend(String v) { this.inferBackend = v; return this; }
        public Builder host(String v) { this.host = v; return this; }
        public Builder port(int v) { this.port = v; return this; }
        public Builder apiKey(String v) { this.apiKey = v; return this; }
        public Builder maxConcurrent(int v) { this.maxConcurrent = v; return this; }
        public Builder generating(GeneratingArgs v) { this.generating = v; return this; }
        public Builder quantizationMethod(QuantizationMethod v) { this.quantizationMethod = v; return this; }
        public Builder flashAttn(boolean v) { this.flashAttn = v; return this; }
        public Builder useUnsloth(boolean v) { this.useUnsloth = v; return this; }
        public InferArgs build() { return new InferArgs(this); }
    }
}
