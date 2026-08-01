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
package org.bytedeco.pytorch.llm.ktransformers.attention;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.ktransformers.cache.KtCacheManager;
import org.bytedeco.pytorch.llm.ktransformers.util.KtPreconditions;
import org.bytedeco.pytorch.llm.kvcache.PagedBlockManager;
import org.bytedeco.pytorch.llm.modules.PagedAttention;
import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;

/**
 * KT wrapper around {@link PagedAttention} with long-context policy + optional
 * three-tier prefix awareness.
 *
 * <p>Default inference path when MLA is not selected. Composes existing
 * {@code modules.PagedAttention}; does not reimplement paged gather math.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class KtPagedAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final PagedAttention inner;
    private final LongContextPolicy policy;
    private final KtCacheManager cacheManager; // optional, may be null
    private long forwardCalls;
    private long pagedCalls;

    public KtPagedAttention(PagedAttention inner, LongContextPolicy policy,
                            KtCacheManager cacheManager) {
        super("KtPagedAttention");
        this.inner = Objects.requireNonNull(inner, "inner");
        this.policy = policy != null ? policy : LongContextPolicy.mini();
        this.cacheManager = cacheManager;
        register_module("paged", inner);
    }

    public KtPagedAttention(PagedAttention inner, LongContextPolicy policy) {
        this(inner, policy, null);
    }

    public static KtPagedAttention gqa(long hiddenSize, int nHeads, int nKvHeads,
                                       double ropeTheta, LongContextPolicy policy) {
        return new KtPagedAttention(
                PagedAttention.gqa(hiddenSize, nHeads, nKvHeads, ropeTheta),
                policy, null);
    }

    /** Mini CI defaults. */
    public static KtPagedAttention mini(long hiddenSize, int nHeads) {
        KtPreconditions.checkPositive((int) hiddenSize, "hiddenSize");
        KtPreconditions.checkPositive(nHeads, "nHeads");
        int nKv = Math.max(1, nHeads / 2);
        return gqa(hiddenSize, nHeads, nKv, 10000.0, LongContextPolicy.mini());
    }

    public PagedAttention inner() { return inner; }
    public LongContextPolicy policy() { return policy; }
    public KtCacheManager cacheManager() { return cacheManager; }
    public long forwardCalls() { return forwardCalls; }
    public long pagedCalls() { return pagedCalls; }

    public int nHeads() { return inner.nHeads(); }
    public int nKvHeads() { return inner.nKvHeads(); }
    public int headDim() { return inner.headDim(); }

    @Override
    public Tensor forward(Tensor x) {
        LongContextPolicy.Decision d = policy.plan((int) x.size(1));
        if (!d.allowed()) {
            throw new IllegalStateException("KtPagedAttention refused: " + d.reason);
        }
        forwardCalls++;
        return inner.forward(x);
    }

    /** Contiguous past-K/V path. */
    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
        int past = pastK != null ? (int) pastK.size(1) : 0;
        int total = past + (int) x.size(1);
        LongContextPolicy.Decision d = policy.plan(total);
        if (!d.allowed()) {
            throw new IllegalStateException("KtPagedAttention refused: " + d.reason);
        }
        forwardCalls++;
        return inner.forwardCached(x, positionOffset, pastK, pastV);
    }

    /**
     * Paged path over a {@link PagedBlockManager} block table.
     * See {@link PagedAttention#forwardPaged}.
     */
    public Tensor[] forwardPaged(Tensor x, long positionOffset,
                                 PagedBlockManager pool, int[] blockTable, int ctxLen, int layer) {
        LongContextPolicy.Decision d = policy.plan(ctxLen + (int) x.size(1));
        if (!d.allowed()) {
            throw new IllegalStateException("KtPagedAttention refused: " + d.reason);
        }
        forwardCalls++;
        pagedCalls++;
        return inner.forwardPaged(x, positionOffset, pool, blockTable, ctxLen, layer);
    }

    /** Estimated dense KV bytes for {@code tokens} (single layer accounting). */
    public long estimateCacheBytes(int tokens) {
        // K+V, nKvHeads, headDim, 2 bytes
        long per = 2L * inner.nKvHeads() * inner.headDim() * 2L;
        return Math.max(0, tokens) * per;
    }
}
