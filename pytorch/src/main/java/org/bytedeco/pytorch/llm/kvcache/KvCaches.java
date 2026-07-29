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
 * Catalog / factory for {@link KvCache} implementations and adapters over the
 * existing paged / sliding / hierarchical / dense-buffer backends.
 *
 * <h2>Policy map</h2>
 * <pre>
 *   Class                 Idea                         Paper / system
 *   ────────────────────  ───────────────────────────  ──────────────────────────
 *   PagedKvCache          block table + prefix radix   vLLM PagedAttention
 *   SlidingWindowKvCache  sink + recent window         Mistral SWA / StreamingLLM
 *   HierarchicalKvCache   hot device / cold host       TRT-LLM offload
 *   KvBufferCache         dense contiguous             simple baseline
 *   TokenLruKvCache       token-level LRU budget       multi-tenant pressure
 *   BlockLruKvCache       session/block LRU preempt    CoW / vLLM style
 *   H2OKvCache            heavy-hitter + recent        H2O (Zhang et al.)
 *   SnapKvCache           obs-window vote compress     SnapKV
 *   TovaKvCache           drop lowest latest attn      TOVA
 *   QuantizedKvCache      per-token int8 + scale       KIVI-lite
 *   CompressedKvCache     MLA latent c_kv + k_rope     DeepSeek MLA
 * </pre>
 */
public final class KvCaches {

    private KvCaches() {}

    public static TokenLruKvCache tokenLru(int layers, int heads, int headDim, int budget, int sink) {
        return new TokenLruKvCache(layers, heads, headDim, budget, sink);
    }

    public static BlockLruKvCache blockLru(int maxBlocks, int layers, int blockSize, int heads, int headDim) {
        return new BlockLruKvCache(maxBlocks, layers, blockSize, heads, headDim);
    }

    public static H2OKvCache h2o(int layers, int heads, int headDim, int heavy, int recent) {
        return new H2OKvCache(layers, heads, headDim, heavy, recent);
    }

    public static SnapKvCache snap(int layers, int heads, int headDim, int maxRetained, int obsWindow) {
        return new SnapKvCache(layers, heads, headDim, maxRetained, obsWindow);
    }

    public static TovaKvCache tova(int layers, int heads, int headDim, int budget) {
        return new TovaKvCache(layers, heads, headDim, budget);
    }

    public static QuantizedKvCache quantized(int layers, int heads, int headDim, int maxLen) {
        return new QuantizedKvCache(layers, heads, headDim, maxLen);
    }

    public static CompressedKvCache compressed(int layers, int kvLoraRank, int ropeDim, int maxLen) {
        return new CompressedKvCache(layers, kvLoraRank, ropeDim, maxLen);
    }

    /** Adapter: {@link PagedKvCache} as {@link KvCache} (tokenId auto-assigned). */
    public static KvCache paged(PagedKvCache inner) {
        return new KvCache() {
            private int tokenCounter = 0;

            @Override
            public long createSequence() {
                return inner.createSequence();
            }

            @Override
            public void releaseSequence(long seqId) {
                inner.releaseSequence(seqId);
            }

            @Override
            public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
                int tokens = inferTokens(kLayers[0]);
                for (int t = 0; t < tokens; t++) {
                    Tensor[] kOne = new Tensor[kLayers.length];
                    Tensor[] vOne = new Tensor[vLayers.length];
                    for (int L = 0; L < kLayers.length; L++) {
                        kOne[L] = sliceToken(kLayers[L], t);
                        vOne[L] = sliceToken(vLayers[L], t);
                    }
                    inner.append(seqId, tokenCounter++, kOne, vOne);
                }
            }

            @Override
            public Tensor[] gather(long seqId, int layer) {
                return inner.gather(seqId, layer);
            }

            @Override
            public int sequenceLength(long seqId) {
                return inner.sequenceLength(seqId);
            }

            @Override
            public int retainedLength(long seqId) {
                return inner.sequenceLength(seqId);
            }

            @Override
            public int numLayers() {
                return inner.numLayers();
            }

            @Override
            public void close() {
                inner.close();
            }
        };
    }

    /** Adapter: {@link SlidingWindowKvCache}. */
    public static KvCache sliding(SlidingWindowKvCache inner) {
        return new KvCache() {
            @Override
            public long createSequence() {
                return inner.createSequence();
            }

            @Override
            public void releaseSequence(long seqId) {
                inner.releaseSequence(seqId);
            }

            @Override
            public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
                int tokens = inferTokens(kLayers[0]);
                for (int t = 0; t < tokens; t++) {
                    Tensor[] kOne = new Tensor[kLayers.length];
                    Tensor[] vOne = new Tensor[vLayers.length];
                    for (int L = 0; L < kLayers.length; L++) {
                        kOne[L] = sliceToken(kLayers[L], t);
                        vOne[L] = sliceToken(vLayers[L], t);
                    }
                    inner.append(seqId, kOne, vOne);
                }
            }

            @Override
            public Tensor[] gather(long seqId, int layer) {
                return inner.gather(seqId, layer);
            }

            @Override
            public int sequenceLength(long seqId) {
                return inner.sequenceLength(seqId);
            }

            @Override
            public int retainedLength(long seqId) {
                return inner.retainedLength(seqId);
            }

            @Override
            public int numLayers() {
                return inner.numLayers();
            }

            @Override
            public void close() {
                inner.close();
            }
        };
    }

    /** Adapter: {@link HierarchicalKvCache}. */
    public static KvCache hierarchical(HierarchicalKvCache inner) {
        return new KvCache() {
            @Override
            public long createSequence() {
                return inner.createSequence();
            }

            @Override
            public void releaseSequence(long seqId) {
                inner.releaseSequence(seqId);
            }

            @Override
            public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
                int tokens = inferTokens(kLayers[0]);
                for (int t = 0; t < tokens; t++) {
                    Tensor[] kOne = new Tensor[kLayers.length];
                    Tensor[] vOne = new Tensor[vLayers.length];
                    for (int L = 0; L < kLayers.length; L++) {
                        kOne[L] = sliceToken(kLayers[L], t);
                        vOne[L] = sliceToken(vLayers[L], t);
                    }
                    inner.append(seqId, kOne, vOne);
                }
            }

            @Override
            public Tensor[] gather(long seqId, int layer) {
                return inner.gather(seqId, layer);
            }

            @Override
            public int sequenceLength(long seqId) {
                return inner.sequenceLength(seqId);
            }

            @Override
            public int retainedLength(long seqId) {
                return inner.sequenceLength(seqId);
            }

            @Override
            public int numLayers() {
                return inner.numLayers();
            }

            @Override
            public void close() {
                inner.close();
            }
        };
    }

    /**
     * Adapter: {@link KvBufferCache} — uses string session keys derived from seq id.
     * Append expects full-layer token writes via {@link KvBufferCache.KvBuffer#appendToken}.
     */
    public static KvCache buffer(KvBufferCache inner) {
        return new KvCache() {
            @Override
            public long createSequence() {
                // KvBufferCache is session-string based; allocate numeric id and materialize buffer
                long id = System.nanoTime() ^ inner.size();
                inner.getKvBuffer(Long.toString(id));
                return id;
            }

            @Override
            public void releaseSequence(long seqId) {
                inner.release(Long.toString(seqId));
            }

            @Override
            public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
                KvBufferCache.KvBuffer buf = inner.getKvBuffer(Long.toString(seqId));
                int tokens = inferTokens(kLayers[0]);
                for (int t = 0; t < tokens; t++) {
                    Tensor[] kOne = new Tensor[kLayers.length];
                    Tensor[] vOne = new Tensor[vLayers.length];
                    for (int L = 0; L < kLayers.length; L++) {
                        // buffer wants [kvWidth] flat or [H*D] — pass [H,D] reshaped
                        Tensor k = sliceToken(kLayers[L], t);
                        Tensor v = sliceToken(vLayers[L], t);
                        kOne[L] = k.reshape(-1);
                        vOne[L] = v.reshape(-1);
                    }
                    buf.appendToken(kOne, vOne);
                }
            }

            @Override
            public Tensor[] gather(long seqId, int layer) {
                KvBufferCache.KvBuffer buf = inner.getKvBuffer(Long.toString(seqId));
                Tensor k = buf.getTensorsUpToCurrent(layer, 0);
                Tensor v = buf.getTensorsUpToCurrent(layer, 1);
                // reshape [pos, kvWidth] → try [pos, H, D] if square-ish unknown — return as [pos, 1, kvWidth]
                if (k.dim() == 2) {
                    long T = k.size(0);
                    long W = k.size(1);
                    k = k.view(T, 1, W);
                    v = v.view(T, 1, W);
                }
                return new Tensor[]{k, v};
            }

            @Override
            public int sequenceLength(long seqId) {
                return inner.getKvBuffer(Long.toString(seqId)).getCurrentPosition();
            }

            @Override
            public int retainedLength(long seqId) {
                return sequenceLength(seqId);
            }

            @Override
            public int numLayers() {
                return inner.numLayers();
            }

            @Override
            public void close() {
                inner.close();
            }
        };
    }

    static int inferTokens(Tensor t) {
        if (t.dim() == 2) return 1;
        if (t.dim() == 3) return (int) t.size(0);
        if (t.dim() == 4) return (int) t.size(2);
        if (t.dim() == 1) return 1;
        throw new IllegalArgumentException("rank " + t.dim());
    }

    static Tensor sliceToken(Tensor t, int tIdx) {
        if (t.dim() == 1 || t.dim() == 2) return t;
        if (t.dim() == 3) return t.select(0, tIdx);
        if (t.dim() == 4) return t.select(0, 0).select(1, tIdx);
        throw new IllegalArgumentException("rank");
    }
}
