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
 * Enterprise pure-Java llama.cpp behaviour surface for GGUF inference.
 *
 * <p>Two backends:
 * <ul>
 *   <li>{@link org.bytedeco.pytorch.llm.llamacpp.InProcessLlamaEngine} — load GGUF via
 *       {@code data.gguf}, dequantize, run transformer blocks on libtorch</li>
 *   <li>{@link org.bytedeco.pytorch.llm.llamacpp.ProcessLlamaRuntime} — spawn official
 *       {@code llama-server} / {@code llama-cli} and speak OpenAI-compatible HTTP</li>
 * </ul>
 *
 * <p>Aligned with <a href="https://github.com/ggerganov/llama.cpp">ggerganov/llama.cpp</a>
 * product knobs (n_ctx, n_gpu_layers, sampling, chat) — not a line-by-line C++ port.
 */
package org.bytedeco.pytorch.llm.llamacpp;
