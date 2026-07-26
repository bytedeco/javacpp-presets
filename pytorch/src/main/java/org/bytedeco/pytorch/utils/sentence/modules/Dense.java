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
package org.bytedeco.pytorch.utils.sentence.modules;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import static org.bytedeco.pytorch.global.torch.relu;
import static org.bytedeco.pytorch.global.torch.tanh;

/**
 * Dense projection with optional activation (Sentence-Transformers Dense module).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Dense extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final LinearImpl linear;
    private final String activation; // "tanh" | "relu" | "none"

    public Dense(int inFeatures, int outFeatures, String activation) {
        super("Dense");
        this.linear = register_module("linear", new LinearImpl(inFeatures, outFeatures));
        this.activation = activation == null ? "tanh" : activation.toLowerCase();
    }

    public Dense(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, "tanh");
    }

    public LinearImpl linear() { return linear; }
    public String activation() { return activation; }

    @Override
    public Tensor forward(Tensor x) {
        Tensor y = linear.forward(x);
        return switch (activation) {
            case "relu" -> relu(y);
            case "none", "linear", "" -> y;
            default -> tanh(y);
        };
    }
}
