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
package org.bytedeco.pytorch.llm.modules;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.relu;
import static org.bytedeco.pytorch.global.torch.silu;

/**
 * Feed-forward / MLP blocks used across modern LLMs.
 *
 * <ul>
 *   <li>{@link SwiGLU} — Llama / Qwen2 / Qwen3 / Mistral / DeepSeek dense FFN</li>
 *   <li>{@link FusedSwiGLU} — GLM / ChatGLM fused gate_up</li>
 *   <li>{@link GeluMlp} — GPT-2 / GPT-NeoX classic GELU MLP</li>
 *   <li>{@link ReluMlp} — simple ReLU two-layer (baseline / distillation)</li>
 *   <li>{@link GeGLU} — Google / Gemma-style GELU-gated FFN</li>
 * </ul>
 */
public final class Mlp {

    private Mlp() {}

    /** SwiGLU: {@code down(silu(gate(x)) * up(x))}. HF names gate/up/down_proj. */
    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class SwiGLU extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl gate_proj;
        public final LinearImpl up_proj;
        public final LinearImpl down_proj;
        private final long hiddenSize;
        private final long intermediateSize;

        public SwiGLU(long hiddenSize, long intermediateSize, boolean bias) {
            super("SwiGLU");
            this.hiddenSize = hiddenSize;
            this.intermediateSize = intermediateSize;
            this.gate_proj = register_module("gate_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, intermediateSize).bias(bias)));
            this.up_proj = register_module("up_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, intermediateSize).bias(bias)));
            this.down_proj = register_module("down_proj",
                    new LinearImpl(new LinearOptions(intermediateSize, hiddenSize).bias(bias)));
        }

        public SwiGLU(long hiddenSize, long intermediateSize) {
            this(hiddenSize, intermediateSize, false);
        }

        public long hiddenSize() { return hiddenSize; }
        public long intermediateSize() { return intermediateSize; }

        @Override
        public Tensor forward(Tensor x) {
            return down_proj.forward(silu(gate_proj.forward(x)).mul(up_proj.forward(x)));
        }
    }

    /**
     * Fused SwiGLU (GLM-Edge / ChatGLM): single {@code gate_up_proj} [2I, H]
     * then {@code down_proj}.
     */
    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class FusedSwiGLU extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl gate_up_proj;
        public final LinearImpl down_proj;
        private final long intermediateSize;

        public FusedSwiGLU(long hiddenSize, long intermediateSize) {
            super("FusedSwiGLU");
            this.intermediateSize = intermediateSize;
            this.gate_up_proj = register_module("gate_up_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, 2L * intermediateSize).bias(false)));
            this.down_proj = register_module("down_proj",
                    new LinearImpl(new LinearOptions(intermediateSize, hiddenSize).bias(false)));
        }

        public long intermediateSize() { return intermediateSize; }

        @Override
        public Tensor forward(Tensor x) {
            Tensor gu = gate_up_proj.forward(x);
            long last = gu.dim() - 1;
            Tensor gate = gu.slice(last, new LongOptional(0),
                    new LongOptional(intermediateSize), 1);
            Tensor up = gu.slice(last, new LongOptional(intermediateSize),
                    new LongOptional(2L * intermediateSize), 1);
            return down_proj.forward(silu(gate).mul(up));
        }
    }

    /** GPT-2 MLP: GELU, names c_fc / c_proj, bias on. */
    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class GeluMlp extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl c_fc;
        public final LinearImpl c_proj;

        public GeluMlp(long hiddenSize, long intermediateSize) {
            super("GeluMlp");
            this.c_fc = register_module("c_fc", new LinearImpl(hiddenSize, intermediateSize));
            this.c_proj = register_module("c_proj", new LinearImpl(intermediateSize, hiddenSize));
        }

        @Override
        public Tensor forward(Tensor x) {
            return c_proj.forward(gelu(c_fc.forward(x)));
        }
    }

    /** Simple two-layer ReLU MLP. */
    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class ReluMlp extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl fc1;
        public final LinearImpl fc2;

        public ReluMlp(long hiddenSize, long intermediateSize) {
            super("ReluMlp");
            this.fc1 = register_module("fc1", new LinearImpl(hiddenSize, intermediateSize));
            this.fc2 = register_module("fc2", new LinearImpl(intermediateSize, hiddenSize));
        }

        @Override
        public Tensor forward(Tensor x) {
            return fc2.forward(relu(fc1.forward(x)));
        }
    }

    /** GeGLU: {@code down(gelu(gate(x)) * up(x))} — Gemma / some T5 variants. */
    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class GeGLU extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl gate_proj;
        public final LinearImpl up_proj;
        public final LinearImpl down_proj;

        public GeGLU(long hiddenSize, long intermediateSize) {
            super("GeGLU");
            this.gate_proj = register_module("gate_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, intermediateSize).bias(false)));
            this.up_proj = register_module("up_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, intermediateSize).bias(false)));
            this.down_proj = register_module("down_proj",
                    new LinearImpl(new LinearOptions(intermediateSize, hiddenSize).bias(false)));
        }

        @Override
        public Tensor forward(Tensor x) {
            return down_proj.forward(gelu(gate_proj.forward(x)).mul(up_proj.forward(x)));
        }
    }
}
