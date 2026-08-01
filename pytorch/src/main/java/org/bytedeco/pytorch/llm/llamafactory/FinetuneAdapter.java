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
package org.bytedeco.pytorch.llm.llamafactory;

import org.bytedeco.pytorch.llm.llamafactory.chat.ChatEngine;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ExportArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningArgs;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

/**
 * SPI for host LLM platforms (ByteDance / Taobao / Tencent style training meshes).
 *
 * <p>Depend on this interface — not on peft/trl internals — so the factory can
 * evolve without breaking outer systems.
 *
 * <pre>{@code
 * FactoryArgs args = FactoryArgs.parse(Map.of(
 *     "model_name_or_path", "Qwen/Qwen2-0.5B",
 *     "stage", "sft",
 *     "finetuning_type", "lora",
 *     "dataset", "alpaca_en_demo",
 *     "output_dir", "saves/qwen-sft"));
 * try (FinetuneAdapter job = LlamaFactory.open(args)) {
 *     job.train();
 *     Path exported = job.export(Path.of("export/qwen"), ExportArgs.builder().build());
 *     String reply = job.chat().chat("Hello");
 * }
 * }</pre>
 */
public interface FinetuneAdapter extends AutoCloseable {

    /** Immutable args snapshot used to open this job. */
    FactoryArgs args();

    /**
     * Blocking multi-stage / single-stage train according to
     * {@link FinetuningArgs#stage()}.
     */
    void train();

    /**
     * Cooperative cancel for board / API driven runs. Default is best-effort no-op
     * until a workflow is attached.
     */
    default void requestStop() {}

    /** Whether {@link #requestStop()} was observed. */
    default boolean stopRequested() {
        return false;
    }

    /**
     * Merge adapters (if any) and write weights + config under {@code dir}.
     *
     * @return directory written
     */
    Path export(Path dir, ExportArgs exportArgs);

    /**
     * Chat engine bound to the latest trained / loaded weights.
     * May load from {@code output_dir} if train has not been called in-process.
     */
    ChatEngine chat();

    /** Live board state when LlamaBoard is enabled; otherwise {@code null}. */
    default BoardState board() {
        return null;
    }

    /** Last logged scalar metrics (loss, lr, …); empty before first step. */
    default Map<String, Double> lastMetrics() {
        return Collections.emptyMap();
    }

    /** Global optimizer step after train (0 if not started). */
    default int globalStep() {
        return 0;
    }

    @Override
    void close();
}
