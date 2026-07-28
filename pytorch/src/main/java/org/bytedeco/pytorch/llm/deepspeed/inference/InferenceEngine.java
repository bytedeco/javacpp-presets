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
package org.bytedeco.pytorch.llm.deepspeed.inference;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * DeepSpeed-style inference engine (Java MVP).
 *
 * <p>{@code replace_with_kernel_inject} and related CUDA kernel flags are recorded
 * as configuration toggles only — numeric path uses standard libtorch Module.forward.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class InferenceEngine implements AutoCloseable {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final Module module;
    private final InferenceConfig config;
    private long numForwards;

    public InferenceEngine(Module module, InferenceConfig config) {
        this.module = Objects.requireNonNull(module, "module");
        this.config = config == null ? InferenceConfig.defaults() : config;
        this.module.eval();
    }

    public static InferenceEngine create(Module module) {
        return new InferenceEngine(module, InferenceConfig.defaults());
    }

    public static InferenceEngine create(Module module, InferenceConfig config) {
        return new InferenceEngine(module, config);
    }

    public Module module() { return module; }
    public InferenceConfig config() { return config; }
    public long numForwards() { return numForwards; }

    public Tensor forward(Tensor input) {
        Objects.requireNonNull(input, "input");
        module.eval();
        Tensor out = module.forward(input);
        numForwards++;
        return out;
    }

    public Map<String, Object> stats() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_forwards", numForwards);
        m.put("replace_with_kernel_inject", config.replaceWithKernelInject());
        m.put("enable_cuda_graph", config.enableCudaGraph());
        m.put("dtype", config.dtype());
        m.put("max_out_tokens", config.maxOutTokens());
        return m;
    }

    @Override
    public void close() {
        // nothing native to free
    }

    public static final class InferenceConfig {
        private final boolean replaceWithKernelInject;
        private final boolean enableCudaGraph;
        private final String dtype;
        private final int maxOutTokens;
        private final boolean triangularMasking;

        private InferenceConfig(Builder b) {
            this.replaceWithKernelInject = b.replaceWithKernelInject;
            this.enableCudaGraph = b.enableCudaGraph;
            this.dtype = b.dtype;
            this.maxOutTokens = b.maxOutTokens;
            this.triangularMasking = b.triangularMasking;
        }

        public static InferenceConfig defaults() { return builder().build(); }
        public static Builder builder() { return new Builder(); }

        public boolean replaceWithKernelInject() { return replaceWithKernelInject; }
        public boolean enableCudaGraph() { return enableCudaGraph; }
        public String dtype() { return dtype; }
        public int maxOutTokens() { return maxOutTokens; }
        public boolean triangularMasking() { return triangularMasking; }

        public static final class Builder {
            private boolean replaceWithKernelInject;
            private boolean enableCudaGraph;
            private String dtype = "fp32";
            private int maxOutTokens = 1024;
            private boolean triangularMasking = true;

            public Builder replaceWithKernelInject(boolean v) { this.replaceWithKernelInject = v; return this; }
            public Builder enableCudaGraph(boolean v) { this.enableCudaGraph = v; return this; }
            public Builder dtype(String v) { this.dtype = v; return this; }
            public Builder maxOutTokens(int v) { this.maxOutTokens = v; return this; }
            public Builder triangularMasking(boolean v) { this.triangularMasking = v; return this; }
            public InferenceConfig build() { return new InferenceConfig(this); }
        }
    }
}
