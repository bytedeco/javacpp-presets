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
package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Tensor;

/**
 * Unified KV-cache surface for paged, dense, sliding, eviction, and compressed
 * backends under {@code org.bytedeco.pytorch.llm.kvcache}.
 *
 * <p>Layer tensors are typically {@code [H, D]} or {@code [1, H, D]} per token
 * (or {@code [T, H, D]} for multi-token append — implementations document which).
 * {@link #gather} returns compact contiguous {@code {K, V}} with no holes so
 * attention modules can run standard SDPA.
 */
public interface KvCache extends AutoCloseable {

    /** Allocate a new sequence id (or session). */
    long createSequence();

    /** Release all resources held by {@code seqId}. */
    void releaseSequence(long seqId);

    /**
     * Append one token (or a short span — see impl) of K/V for every layer.
     *
     * @param kLayers length = numLayers; each {@code [H,D]} or {@code [1,H,D]} or {@code [T,H,D]}
     * @param vLayers same shapes as kLayers
     */
    void append(long seqId, Tensor[] kLayers, Tensor[] vLayers);

    /**
     * Compact gather of retained K/V for one layer.
     *
     * @return {@code {K, V}} each shaped {@code [retainedLen, H, D]} (or impl-documented layout)
     */
    Tensor[] gather(long seqId, int layer);

    /** Total tokens ever appended (may exceed retained after eviction). */
    int sequenceLength(long seqId);

    /** Tokens currently held after eviction / windowing. */
    int retainedLength(long seqId);

    int numLayers();

    @Override
    void close();
}
