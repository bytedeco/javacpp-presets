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

package org.bytedeco.pytorch.llm.trl.spi;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;
import java.util.function.Function;

/**
 * Adapts {@link CausalLM} or a generic logits Module to {@link LlmForward}.
 */
public final class CausalLmForwardAdapter implements LlmForward {

    private final Function<Tensor, Tensor> forwardFn;

    public CausalLmForwardAdapter(CausalLM model) {
        Objects.requireNonNull(model, "model");
        this.forwardFn = model::forward;
    }

    public CausalLmForwardAdapter(Module module) {
        Objects.requireNonNull(module, "module");
        this.forwardFn = input -> {
            try {
                // prefer forward(Tensor) reflectively
                return (Tensor) module.getClass().getMethod("forward", Tensor.class).invoke(module, input);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Module has no forward(Tensor): " + module.getClass().getName(), e);
            }
        };
    }

    public CausalLmForwardAdapter(Function<Tensor, Tensor> forwardFn) {
        this.forwardFn = Objects.requireNonNull(forwardFn);
    }

    public static LlmForward of(CausalLM model) {
        return new CausalLmForwardAdapter(model);
    }

    public static LlmForward of(Module module) {
        if (module instanceof CausalLM clm) return of(clm);
        return new CausalLmForwardAdapter(module);
    }

    @Override
    public Tensor forward(Tensor inputIds, Tensor attentionMask) {
        // CausalLM.forward currently uses input ids only; mask reserved for future.
        return forwardFn.apply(inputIds);
    }
}
