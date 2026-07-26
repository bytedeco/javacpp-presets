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
package org.bytedeco.pytorch.utils.transformers.modeling;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.gelu;
import static org.bytedeco.pytorch.global.torch.silu;

/**
 * MLP blocks used by causal LMs.
 *
 * <ul>
 *   <li>{@link SwiGLU} — Llama / Qwen2 / Mistral ({@code gate_proj}, {@code up_proj}, {@code down_proj})</li>
 *   <li>{@link GeluMlp} — GPT-2 style ({@code c_fc}, {@code c_proj})</li>
 * </ul>
 */
public final class ModelingMlp {

    private ModelingMlp() {}

    /** SwiGLU: {@code down(silu(gate(x)) * up(x))} with HF param names. */
    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class SwiGLU extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final LinearImpl gate_proj;
        public final LinearImpl up_proj;
        public final LinearImpl down_proj;

        public SwiGLU(long hiddenSize, long intermediateSize) {
            super("SwiGLU");
            // bias=false matches Llama/Qwen2
            this.gate_proj = register_module("gate_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, intermediateSize).bias(false)));
            this.up_proj = register_module("up_proj",
                    new LinearImpl(new LinearOptions(hiddenSize, intermediateSize).bias(false)));
            this.down_proj = register_module("down_proj",
                    new LinearImpl(new LinearOptions(intermediateSize, hiddenSize).bias(false)));
        }

        @Override
        public Tensor forward(Tensor x) {
            return down_proj.forward(silu(gate_proj.forward(x)).mul(up_proj.forward(x)));
        }
    }

    /** GPT-2 MLP with GELU; names {@code c_fc}/{@code c_proj}. */
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
}
