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
import org.bytedeco.pytorch.llm.ktransformers.util.KtPreconditions;
import org.bytedeco.pytorch.llm.modules.MultiLatentAttention;
import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;

/**
 * KT wrapper around {@link MultiLatentAttention} with optional long-context policy
 * and compressed-KV cache bookkeeping.
 *
 * <p>Aligns with upstream DeepSeek-style MLA serving: keep latent {@code c_kv}
 * compact on GPU, and let {@link LongContextPolicy} decide when to demote /
 * refuse under VRAM budgets. Does not reimplement MLA math — composes
 * {@code modules.MultiLatentAttention}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class KtMlaAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final MultiLatentAttention inner;
    private final LongContextPolicy policy;
    private long forwardCalls;
    private long cachedTokens;

    public KtMlaAttention(MultiLatentAttention inner, LongContextPolicy policy) {
        super("KtMlaAttention");
        this.inner = Objects.requireNonNull(inner, "inner");
        this.policy = policy != null ? policy : LongContextPolicy.mini();
        register_module("mla", inner);
    }

    public KtMlaAttention(long hiddenSize, int nHeads,
                          int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
                          int kvLoraRank, int qLoraRank, double ropeTheta,
                          LongContextPolicy policy) {
        this(new MultiLatentAttention(hiddenSize, nHeads, qkNopeHeadDim, qkRopeHeadDim,
                        vHeadDim, kvLoraRank, qLoraRank, ropeTheta),
                policy);
    }

    /** Small CI-friendly MLA (fits mini demo dims). */
    public static KtMlaAttention mini(long hiddenSize, int nHeads) {
        KtPreconditions.checkPositive((int) hiddenSize, "hiddenSize");
        KtPreconditions.checkPositive(nHeads, "nHeads");
        int headDim = Math.max(8, (int) (hiddenSize / nHeads));
        int rope = Math.min(16, headDim / 2);
        int nope = Math.max(8, headDim - rope);
        int v = headDim;
        int kvLora = Math.max(8, headDim);
        int qLora = Math.max(0, (int) Math.min(hiddenSize, Math.max(kvLora, hiddenSize / 4)));
        return new KtMlaAttention(hiddenSize, nHeads, nope, rope, v, kvLora, qLora, 10000.0,
                LongContextPolicy.mini());
    }

    public static KtMlaAttention deepseekLike(long hiddenSize, int nHeads, int kvLoraRank) {
        return new KtMlaAttention(
                MultiLatentAttention.deepseek(hiddenSize, nHeads, kvLoraRank, 10000.0),
                LongContextPolicy.consumer24g());
    }

    public MultiLatentAttention inner() { return inner; }
    public LongContextPolicy policy() { return policy; }
    public long forwardCalls() { return forwardCalls; }
    public long cachedTokens() { return cachedTokens; }

    public int nHeads() { return inner.nHeads(); }
    public int kvLoraRank() { return inner.kvLoraRank(); }
    public long hiddenSize() { return inner.hiddenSize(); }

    @Override
    public Tensor forward(Tensor x) {
        LongContextPolicy.Decision d = policy.plan((int) x.size(1));
        if (!d.allowed()) {
            throw new IllegalStateException("KtMlaAttention refused by LongContextPolicy: " + d.reason);
        }
        forwardCalls++;
        cachedTokens += x.size(1);
        return inner.forward(x);
    }

    /**
     * Cached path: returns {@code {out, newCkv, newKr}} — same contract as
     * {@link MultiLatentAttention#forwardCached}.
     */
    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastCkv, Tensor pastKr) {
        int past = pastCkv != null ? (int) pastCkv.size(1) : 0;
        int total = past + (int) x.size(1);
        LongContextPolicy.Decision d = policy.plan(total);
        if (!d.allowed()) {
            throw new IllegalStateException("KtMlaAttention refused by LongContextPolicy: " + d.reason);
        }
        forwardCalls++;
        cachedTokens = total;
        return inner.forwardCached(x, positionOffset, pastCkv, pastKr);
    }

    /** Estimated compressed KV bytes for {@code tokens} under this layer. */
    public long estimateCacheBytes(int tokens) {
        // c_kv [B,T,kv_lora] + k_rope [B,T,rope] ≈ 2 * kv_lora * 2 bytes (fp16-ish)
        long per = 2L * inner.kvLoraRank() * 2L + 2L * inner.qkRopeHeadDim() * 2L;
        return Math.max(0, tokens) * per;
    }
}
