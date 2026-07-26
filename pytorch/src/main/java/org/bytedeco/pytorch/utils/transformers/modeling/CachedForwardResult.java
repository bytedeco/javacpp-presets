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

import org.bytedeco.pytorch.Tensor;

/**
 * Result of a cache-aware causal-LM forward.
 *
 * <ul>
 *   <li>{@link #hiddenOrLogits} — model: hidden [B,T,C]; top-level LM: logits [B,T,V]</li>
 *   <li>{@link #newKs} / {@link #newVs} — per-layer new K/V, each [B, nHeads, T, headDim]
 *       (GQA-repeated to nHeads for cache storage)</li>
 * </ul>
 */
public final class CachedForwardResult {

    public final Tensor hiddenOrLogits;
    public final Tensor[] newKs;
    public final Tensor[] newVs;

    public CachedForwardResult(Tensor hiddenOrLogits, Tensor[] newKs, Tensor[] newVs) {
        this.hiddenOrLogits = hiddenOrLogits;
        this.newKs = newKs;
        this.newVs = newVs;
    }

    public Tensor logits() { return hiddenOrLogits; }
    public Tensor hidden() { return hiddenOrLogits; }
    public int numLayers() { return newKs == null ? 0 : newKs.length; }
}
