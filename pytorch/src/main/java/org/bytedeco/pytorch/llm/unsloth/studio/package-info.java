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
 * Pure-Java Unsloth Studio product surface for local LLM inference, fine-tuning,
 * export, OpenAI/Anthropic-compatible serving, data recipes, MCP control, and
 * visual training boards.
 *
 * <p>Behaviour is aligned with
 * <a href="https://github.com/unslothai/unsloth/tree/main/studio">unslothai/unsloth studio</a>
 * (search/download/run, LoRA/QLoRA/full/RL, GGUF controls, tool calling, recipes,
 * observability) but implemented entirely in Java on top of this repository's
 * existing modules:
 * <ul>
 *   <li>{@code org.bytedeco.pytorch.llm.unsloth} — FastLanguageModel / FastConfig</li>
 *   <li>{@code org.bytedeco.pytorch.llm.peft} / {@code trl} / {@code bitsandbytes}</li>
 *   <li>{@code org.bytedeco.pytorch.llm.vllm} — high-throughput inference</li>
 *   <li>{@code org.bytedeco.pytorch.llm.hub} — Hugging Face cache / transfer</li>
 *   <li>{@code org.bytedeco.pytorch.llm.factory} — LLaMA-Factory orchestration bridge</li>
 *   <li>{@code org.bytedeco.pytorch.plot.*} — TensorBoard / WandB / SwanLab sinks</li>
 * </ul>
 *
 * <p>Host platforms (ByteDance / Taobao / Tencent style training meshes) should
 * depend on {@link org.bytedeco.pytorch.llm.unsloth.studio.UnslothStudio} and
 * {@link org.bytedeco.pytorch.llm.unsloth.studio.StudioAdapter} only.
 *
 * <h2>Subpackages</h2>
 * <ul>
 *   <li>{@code model} — immutable request/response DTOs and enums</li>
 *   <li>{@code hardware} — device probe, VRAM estimate, GGUF hardware controls</li>
 *   <li>{@code hub} — model/dataset inventory and download orchestration</li>
 *   <li>{@code inference} — load/run/chat, tools, providers, compare, multimodal</li>
 *   <li>{@code train} — LoRA/QLoRA/full/RL orchestrator and progress bus</li>
 *   <li>{@code data} — datasets, format detection, visual-node data recipes</li>
 *   <li>{@code export} — safetensors / peft-merge / GGUF export planning</li>
 *   <li>{@code rag} — PDF/web chunking helpers for grounded chat</li>
 *   <li>{@code api} — OpenAI/Anthropic-compatible HTTP server</li>
 *   <li>{@code mcp} — Model Context Protocol control endpoint</li>
 *   <li>{@code observe} — live metrics, GPU sampling, plot sinks</li>
 *   <li>{@code webui} — pure-Java visual training board (SSE + SVG)</li>
 *   <li>{@code auth} — optional local token gate</li>
 *   <li>{@code util} — validation, paths, JSON helpers</li>
 * </ul>
 *
 * @see org.bytedeco.pytorch.llm.unsloth.studio.UnslothStudio
 */
package org.bytedeco.pytorch.llm.unsloth.studio;
