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
package org.bytedeco.pytorch.llm.quantization;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.nn.Module;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Container produced by {@link TensorQuantizer#quantizeModel(Module)}: holds a
 * reference to the original module plus per-parameter {@link QuantizedLinear}
 * weights produced by the quantizer.
 *
 * <p>Does <strong>not</strong> close the quantizer (caller owns that).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class QuantizedModel implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final Module source;
    private final Map<String, QuantizedLinear> quantizedModules;
    private final TensorQuantizer quantizer;

    public QuantizedModel(
            Module source,
            Map<String, QuantizedLinear> quantizedModules,
            TensorQuantizer quantizer) {
        this.source = Objects.requireNonNull(source, "source");
        this.quantizedModules = Collections.unmodifiableMap(
                new LinkedHashMap<String, QuantizedLinear>(
                        Objects.requireNonNull(quantizedModules, "quantizedModules")));
        this.quantizer = Objects.requireNonNull(quantizer, "quantizer");
    }

    public Module getSource() { return source; }
    public Map<String, QuantizedLinear> getQuantizedModules() { return quantizedModules; }
    public TensorQuantizer getQuantizer() { return quantizer; }

    public QuantizedLinear get(String name) {
        return quantizedModules.get(name);
    }

    public int size() {
        return quantizedModules.size();
    }

    @Override
    public void close() {
        for (QuantizedLinear ql : quantizedModules.values()) {
            ql.close();
        }
    }

    @Override
    public String toString() {
        return "QuantizedModel{modules=" + quantizedModules.size()
                + ", dtype=" + quantizer.getDtype()
                + ", mode=" + quantizer.getMode() + '}';
    }
}
