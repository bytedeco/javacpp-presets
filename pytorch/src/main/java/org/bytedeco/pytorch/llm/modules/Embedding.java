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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;

import static org.bytedeco.pytorch.global.torch.arange;

/**
 * Token + optional absolute positional embeddings (GPT-2 style) and
 * optional embedding dropout.
 *
 * <p>Llama / Qwen / DeepSeek typically use only token embeddings + RoPE
 * (no absolute pos table). GPT-2 uses both token and position tables.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Embedding extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final EmbeddingImpl token;
    /** Absolute position table; null when RoPE-only models. */
    public final EmbeddingImpl position;
    public final DropoutImpl drop;

    private final long vocabSize;
    private final long hiddenSize;
    private final long maxPositions;
    private final boolean useAbsolutePos;

    public Embedding(long vocabSize, long hiddenSize, long maxPositions,
                     boolean useAbsolutePos, double dropoutP) {
        super("Embedding");
        if (vocabSize <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException("vocabSize/hiddenSize must be > 0");
        }
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.maxPositions = Math.max(1, maxPositions);
        this.useAbsolutePos = useAbsolutePos;
        this.token = register_module("token", new EmbeddingImpl(vocabSize, hiddenSize));
        if (useAbsolutePos) {
            this.position = register_module("position",
                    new EmbeddingImpl(this.maxPositions, hiddenSize));
        } else {
            this.position = null;
        }
        this.drop = register_module("drop", new DropoutImpl(Math.max(0.0, dropoutP)));
    }

    /** RoPE-only token embedding (Llama / Qwen / DeepSeek). */
    public static Embedding ropeOnly(long vocabSize, long hiddenSize) {
        return new Embedding(vocabSize, hiddenSize, 1, false, 0.0);
    }

    /** GPT-2 style token + absolute position. */
    public static Embedding gpt2(long vocabSize, long hiddenSize, long maxPositions, double dropoutP) {
        return new Embedding(vocabSize, hiddenSize, maxPositions, true, dropoutP);
    }

    public long vocabSize() { return vocabSize; }
    public long hiddenSize() { return hiddenSize; }
    public long maxPositions() { return maxPositions; }
    public boolean useAbsolutePos() { return useAbsolutePos; }

    public Tensor weight() {
        return token.weight();
    }

    /**
     * @param inputIds [B, T] long token ids
     * @return [B, T, H] embeddings
     */
    @Override
    public Tensor forward(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        long T = ids.size(1);
        Tensor x = token.forward(ids);
        if (useAbsolutePos) {
            if (T > maxPositions) {
                throw new IllegalArgumentException(
                        "Sequence length " + T + " exceeds maxPositions=" + maxPositions);
            }
            Tensor pos = arange(new Scalar(0L), new Scalar(T), new Scalar(1L));
            x = x.add(position.forward(pos));
        }
        return drop.forward(x);
    }

    /**
     * Incremental decode with absolute position offset (GPT-2 cached path).
     *
     * @param positionOffset starting absolute index for this chunk
     */
    public Tensor forward(Tensor inputIds, long positionOffset) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        long T = ids.size(1);
        Tensor x = token.forward(ids);
        if (useAbsolutePos) {
            long end = positionOffset + T;
            if (end > maxPositions) {
                throw new IllegalArgumentException(
                        "Sequence end " + end + " exceeds maxPositions=" + maxPositions);
            }
            Tensor pos = arange(new Scalar(positionOffset), new Scalar(end), new Scalar(1L));
            x = x.add(position.forward(pos));
        }
        return drop.forward(x);
    }
}
