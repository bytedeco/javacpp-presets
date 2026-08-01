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
 * Pure-Java port of the LLaMA-Factory product surface
 * ({@code https://github.com/hiyouga/LLaMA-Factory}).
 *
 * <p>This package is the <strong>orchestration + product facade</strong> for
 * multi-stage LLM fine-tuning. Algorithmic building blocks live in sibling
 * packages and are composed here:
 * <ul>
 *   <li>{@code org.bytedeco.pytorch.llm.peft} — LoRA / QLoRA / DoRA / OFT / …</li>
 *   <li>{@code org.bytedeco.pytorch.llm.trl} — SFT / DPO / KTO / PPO / ORPO / RM / GRPO</li>
 *   <li>{@code org.bytedeco.pytorch.llm.transformers} — AutoModel, CausalLM, registry</li>
 *   <li>{@code org.bytedeco.pytorch.llm.unsloth} — fast-path LoRA training</li>
 *   <li>{@code org.bytedeco.pytorch.llm.vllm} — high-throughput inference workers</li>
 *   <li>{@code org.bytedeco.pytorch.llm.accelerate} / {@code deepspeed} — distributed</li>
 *   <li>{@code org.bytedeco.pytorch.plot.*} — TensorBoard / WandB / SwanLab / MLflow</li>
 * </ul>
 *
 * <p>Host LLM platforms should depend on {@link org.bytedeco.pytorch.llm.llamafactory.LlamaFactory}
 * and {@link org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter} only.
 *
 * <h2>Subpackages</h2>
 * <ul>
 *   <li>{@code hparams} — typed arguments, stage / finetuning / quant enums</li>
 *   <li>{@code data} — templates, converters, collators, dataset builder</li>
 *   <li>{@code model} — loader, freeze, adapter, rope, value head, patches</li>
 *   <li>{@code train} — workflow, trainer factory, checkpoint, bridges</li>
 *   <li>{@code chat} — multi-turn chat engine</li>
 *   <li>{@code api} — OpenAI-compatible HTTP server</li>
 *   <li>{@code webui} — LlamaBoard visual training dashboard</li>
 *   <li>{@code extras} — advanced optims, quant loaders, monitors, misc</li>
 *   <li>{@code eval} — lightweight evaluation harness</li>
 *   <li>{@code export} — merge / dtype-cast / save</li>
 * </ul>
 */
package org.bytedeco.pytorch.llm.llamafactory;
