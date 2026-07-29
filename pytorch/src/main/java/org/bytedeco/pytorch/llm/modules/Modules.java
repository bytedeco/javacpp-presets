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

/**
 * Catalog / factory entry-point for reusable LLM building blocks in
 * {@code org.bytedeco.pytorch.llm.modules}.
 *
 * <h2>Layer map by model family</h2>
 * <pre>
 *   Family          Norm     Pos enc     Attention          FFN
 *   ──────────────  ───────  ──────────  ─────────────────  ────────────────
 *   Llama/Mistral   RMSNorm  RoPE        GQA                SwiGLU
 *   Qwen2           RMSNorm  RoPE        GQA + qkv bias     SwiGLU
 *   Qwen3           RMSNorm  RoPE        GQA + QK-Norm      SwiGLU
 *   GPT-2 / GPT     LayerNorm absolute   MHA                GELU MLP
 *   GLM / ChatGLM   RMSNorm  RoPE        GQA                FusedSwiGLU
 *   Gemma           RMSNorm  RoPE        MHA/GQA            GeGLU
 *   DeepSeek-V2/V3  RMSNorm  RoPE(MLA)   MultiLatentAttn    MoE + shared
 *   Mixtral         RMSNorm  RoPE        GQA                MoE top-2
 *   Falcon/Bloom    LayerNorm ALiBi      MQA/MHA            GELU
 * </pre>
 *
 * <h2>Paper-level attention variants</h2>
 * <pre>
 *   Class                   Paper / system              Cache shape
 *   ──────────────────────  ──────────────────────────  ────────────────────
 *   Attention               dense MHA/GQA/MQA/SWA/ALiBi standard K/V
 *   FlashAttention          Dao et al. online softmax   standard K/V
 *   PagedAttention          vLLM block-table attend     pages + table
 *   CrossAttention          Vaswani encoder–decoder     memory K/V
 *   DifferentialAttention   Diff Transformer (MS)       dual-group K/V
 *   LinearAttention         Performer / Linear Trans.   S, Z states
 *   SparseAttention         Longformer / BigBird        standard / window
 *   InfiniAttention         Infini-attention (lite)     local + mem
 *   RetentionAttention      RetNet / Lightning          recurrent S
 *   StreamingSinkAttention  StreamingLLM                sink∥window
 *   NativeSparseAttention   DeepSeek NSA (lite)         selected blocks
 *   GatedAttention          gated attn output           standard K/V
 *   H2OAttention            H2O mass side-channel       standard + mass
 *   MultiLatentAttention    DeepSeek MLA                c_kv + k_rope
 * </pre>
 *
 * <p>Shared helpers: {@link org.bytedeco.pytorch.llm.modules.attn.AttentionOps}.
 *
 * <h2>Package contents</h2>
 * <ul>
 *   <li>{@link RMSNorm}, {@link LayerNorm} — normalization</li>
 *   <li>{@link RotaryEmbedding} — RoPE + GQA repeat + interleaved variant</li>
 *   <li>{@link Embedding} — token (+ optional absolute position)</li>
 *   <li>{@link Mlp} — SwiGLU / FusedSwiGLU / GeluMlp / ReluMlp / GeGLU</li>
 *   <li>{@link Attention} — MHA / GQA / MQA / sliding-window / ALiBi / QK-Norm</li>
 *   <li>{@link FlashAttention}, {@link PagedAttention}, {@link CrossAttention},
 *       {@link DifferentialAttention}, {@link LinearAttention}, {@link SparseAttention},
 *       {@link InfiniAttention}, {@link RetentionAttention}, {@link StreamingSinkAttention},
 *       {@link NativeSparseAttention}, {@link GatedAttention}, {@link H2OAttention}</li>
 *   <li>{@link MultiLatentAttention} — DeepSeek MLA compressed KV</li>
 *   <li>{@link MoE} — sparse experts + optional shared expert + aux loss</li>
 *   <li>{@link DecoderLayer} — pre/post-norm residual block</li>
 *   <li>{@link MlaDecoderLayer} — MLA + SwiGLU/MoE block</li>
 *   <li>{@link ParallelLinear} — column/row TP + ParallelSwiGLU (single-rank OK)</li>
 *   <li>{@link MiniCausalLM} — compose full tiny LMs (llama/qwen/gpt2/glm/moe…)</li>
 * </ul>
 *
 * <p>For HF weight-compatible full models see
 * {@code org.bytedeco.pytorch.llm.transformers.modeling} (Llama/Qwen2/Qwen3/GLM).
 * This package is the reusable kit those (and custom nets) can share.
 *
 * <p>KV-cache policies live in {@code org.bytedeco.pytorch.llm.kvcache}
 * ({@code KvCache}, {@code KvCaches}, TokenLru / H2O / Snap / TOVA / Quantized / Compressed…).
 *
 * <p>Multi-dim accuracy benches:
 * {@code samples.BenchmarkLlmAttention}, {@code samples.BenchmarkLlmKvCache},
 * {@code samples.BenchmarkLlmModules}.
 */
public final class Modules {

    private Modules() {}

    /** Tiny Llama-shaped net for smoke tests. */
    public static MiniCausalLM tinyLlama() {
        return MiniCausalLM.tiny();
    }

    public static MiniCausalLM tinyGpt2() {
        return MiniCausalLM.gpt2(128, 64, 2, 4);
    }

    public static MiniCausalLM tinyQwen3() {
        return MiniCausalLM.qwen3(128, 64, 2, 4, 2, 16);
    }

    public static MiniCausalLM tinyDeepseekMoe() {
        return MiniCausalLM.deepseekMoe(128, 64, 4, 4, 2, 4, 2);
    }

    public static MiniCausalLM tinyMixtral() {
        return MiniCausalLM.mixtral(128, 64, 2, 4, 2, 4, 2);
    }

    public static MiniCausalLM tinyGlm() {
        return MiniCausalLM.glm(128, 64, 2, 4, 2);
    }

    // ---- attention factories (thin shortcuts) ----

    public static FlashAttention flashMha(long hidden, int heads, double ropeTheta) {
        return FlashAttention.mha(hidden, heads, ropeTheta);
    }

    public static FlashAttention flashGqa(long hidden, int heads, int kvHeads, double ropeTheta) {
        return FlashAttention.gqa(hidden, heads, kvHeads, ropeTheta);
    }

    public static CrossAttention crossMha(long hidden, int heads) {
        return CrossAttention.mha(hidden, heads);
    }

    public static DifferentialAttention diffAttn(long hidden, int heads, double ropeTheta) {
        return DifferentialAttention.paperDefault(hidden, heads, ropeTheta);
    }

    public static LinearAttention linearMha(long hidden, int heads) {
        return LinearAttention.mha(hidden, heads);
    }

    public static SparseAttention longformer(long hidden, int heads, double ropeTheta, int window, int nGlobal) {
        return SparseAttention.longformer(hidden, heads, ropeTheta, window, nGlobal);
    }

    public static InfiniAttention infini(long hidden, int heads, double ropeTheta, int window) {
        return InfiniAttention.paperDefault(hidden, heads, ropeTheta);
    }

    public static RetentionAttention retention(long hidden, int heads) {
        return RetentionAttention.mha(hidden, heads);
    }

    public static StreamingSinkAttention streamingSink(long hidden, int heads, double ropeTheta,
                                                       int sink, int window) {
        return StreamingSinkAttention.gqa(hidden, heads, heads, ropeTheta, sink, window);
    }

    public static NativeSparseAttention nsa(long hidden, int heads, double ropeTheta) {
        return NativeSparseAttention.paperDefault(hidden, heads, ropeTheta);
    }

    public static GatedAttention gatedGqa(long hidden, int heads, int kvHeads, double ropeTheta) {
        return GatedAttention.gqa(hidden, heads, kvHeads, ropeTheta);
    }

    public static H2OAttention h2oGqa(long hidden, int heads, int kvHeads, double ropeTheta) {
        return H2OAttention.gqa(hidden, heads, kvHeads, ropeTheta);
    }

    public static PagedAttention pagedGqa(long hidden, int heads, int kvHeads, double ropeTheta) {
        return PagedAttention.gqa(hidden, heads, kvHeads, ropeTheta);
    }
}
