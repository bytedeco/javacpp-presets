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
package org.bytedeco.pytorch.llm.sentence.modules;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.full;

/**
 * Sentence-Transformers style embedding + LayerNorm transformer tower (mini).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class TransformerModule extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final EmbeddingImpl embedding;
    private final LayerNormImpl norm;
    private final int hiddenSize;

    public TransformerModule(int vocabSize, int hiddenSize) {
        super("TransformerModule");
        this.hiddenSize = hiddenSize;
        this.embedding = register_module("emb", new EmbeddingImpl(vocabSize, hiddenSize));
        this.norm = register_module("norm", new LayerNormImpl(new LongVector().put((long) hiddenSize)));
    }

    public int hiddenSize() { return hiddenSize; }
    public EmbeddingImpl embedding() { return embedding; }

    public Tensor encodeIds(Tensor inputIds) {
        Tensor h = embedding.forward(inputIds);
        return norm.forward(h);
    }

    @Override
    public Tensor forward(Tensor input) {
        return encodeIds(input);
    }
}
