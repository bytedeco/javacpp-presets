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

/**
 * Pure-Java port of the KTransformers product surface
 * ({@code https://github.com/kvcache-ai/ktransformers}).
 *
 * <p>Upstream exposes two user-facing capabilities from the <strong>kt-kernel</strong>
 * tree: <em>Inference</em> (CPU–GPU heterogeneous serving) and <em>SFT</em>
 * (fine-tuning, including LLaMA-Factory integration). This package mirrors those
 * entry points in pure Java on top of JavaCPP PyTorch bindings.
 *
 * <h2>What lives here</h2>
 * <ul>
 *   <li>{@code config} — immutable builders for inference / SFT / quant / MoE / cache</li>
 *   <li>{@code kernel} — quant linear backends (INT4/8, FP8-channel, AMX-like GEMM semantics)</li>
 *   <li>{@code moe} — CPU–GPU expert scheduling, NUMA-aware placement, routed MoE</li>
 *   <li>{@code cache} — three-tier (GPU–CPU–Disk) prefix cache reuse</li>
 *   <li>{@code attention} — MLA / paged paths composed with {@code llm.modules}</li>
 *   <li>{@code inject} — layer injection plans per model family</li>
 *   <li>{@code inference} — {@link org.bytedeco.pytorch.llm.ktransformers.inference.KtInferenceEngine}</li>
 *   <li>{@code sft} — heterogeneous fine-tuning session + TRL/PEFT bridges</li>
 *   <li>{@code adapter} — {@link org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter} SPI for host meshes</li>
 *   <li>{@code monitor} — TensorBoard / WandB / LlamaBoard metrics for visual training</li>
 * </ul>
 *
 * <h2>Composition (do not reimplement)</h2>
 * <ul>
 *   <li>{@code org.bytedeco.pytorch.llm.modules} — MoE, MLA, RoPE, RMSNorm, Attention</li>
 *   <li>{@code org.bytedeco.pytorch.llm.kvcache} — paged / hierarchical / prefix radix</li>
 *   <li>{@code org.bytedeco.pytorch.llm.peft} / {@code trl} / {@code factory} — LoRA &amp; trainers</li>
 *   <li>{@code org.bytedeco.pytorch.llm.vllm} — high-throughput serving hooks</li>
 *   <li>{@code org.bytedeco.pytorch.llm.quantization} / {@code unsloth} / {@code bitsandbytes}</li>
 *   <li>{@code org.bytedeco.pytorch.plot.*} — visualization backends</li>
 * </ul>
 *
 * <h2>Honest hardware boundary</h2>
 * Hand-written AMX/AVX/CUDA kernels from upstream are represented as
 * <strong>algorithmically equivalent</strong> torch reference backends
 * ({@link org.bytedeco.pytorch.llm.ktransformers.kernel.CpuRefKernelBackend}).
 * Claims of native AMX linkage are never made unless a real native backend is registered.
 *
 * <p>Host platforms should depend on {@link org.bytedeco.pytorch.llm.ktransformers.KTransformers}
 * and {@link org.bytedeco.pytorch.llm.ktransformers.adapter.KTransformersFinetuneAdapter}.
 *
 * @see org.bytedeco.pytorch.llm.ktransformers.KTransformers
 */
package org.bytedeco.pytorch.llm.ktransformers;
