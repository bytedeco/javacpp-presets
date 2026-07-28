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
package org.bytedeco.pytorch.llm.vllm.runner;

import org.bytedeco.pytorch.llm.sentence.SentenceTransformer;

import java.util.List;

/**
 * Text embedding via {@link SentenceTransformer}.
 *
 * <p>For Phase 1, uses the pure-Java mini encoder.
 * Future: load a real HF sentence-transformers checkpoint via HfHub.
 */
public final class EmbeddingRunner {

    private final SentenceTransformer model;

    public EmbeddingRunner(SentenceTransformer model) {
        this.model = model;
    }

    /** Encode one text → float[embedDim]. */
    public float[] encode(String text) {
        return model.encode(text);
    }

    /** Batch encode. */
    public float[][] encodeBatch(List<String> texts) {
        return model.encode(texts);
    }

    /** Encode and L2-normalize (cosine similarity compatible). */
    public float[] encodeNormalized(String text) {
        float[] v = encode(text);
        l2Normalize(v);
        return v;
    }

    public int dimension() { return model.getEmbedDim(); }

    public void close() {
        try { model.close(); } catch (Exception e) { /* ignore */ }
    }

    private static void l2Normalize(float[] v) {
        double norm = 0;
        for (float f : v) norm += f * f;
        norm = Math.sqrt(norm);
        if (norm > 1e-8) {
            for (int i = 0; i < v.length; i++) v[i] = (float) (v[i] / norm);
        }
    }
}
