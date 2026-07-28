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
 * <h2>Package contents</h2>
 * <ul>
 *   <li>{@link RMSNorm}, {@link LayerNorm} — normalization</li>
 *   <li>{@link RotaryEmbedding} — RoPE + GQA repeat + interleaved variant</li>
 *   <li>{@link Embedding} — token (+ optional absolute position)</li>
 *   <li>{@link Mlp} — SwiGLU / FusedSwiGLU / GeluMlp / ReluMlp / GeGLU</li>
 *   <li>{@link Attention} — MHA / GQA / MQA / sliding-window / ALiBi / QK-Norm</li>
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
}
