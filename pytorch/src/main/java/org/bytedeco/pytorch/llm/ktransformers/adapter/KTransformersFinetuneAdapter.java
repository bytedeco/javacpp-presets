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
package org.bytedeco.pytorch.llm.ktransformers.adapter;

import org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter;
import org.bytedeco.pytorch.llm.llamafactory.chat.ChatEngine;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ExportArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.sft.KtSftSession;

import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Host-mesh SPI implementation: KTransformers as a {@link FinetuneAdapter}.
 *
 * <pre>{@code
 * try (FinetuneAdapter job = KTransformersFinetuneAdapter.open(factoryArgs)) {
 *     job.train();
 *     job.export(Path.of("export/kt"), ExportArgs.builder().build());
 *     System.out.println(job.chat().chat("hi"));
 * }
 * }</pre>
 */
public final class KTransformersFinetuneAdapter implements FinetuneAdapter {

    private final FactoryArgs args;
    private final KtConfig ktConfig;
    private final KtSftSession session;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public KTransformersFinetuneAdapter(FactoryArgs args, KtConfig ktConfig) {
        this.args = Objects.requireNonNull(args, "args");
        this.ktConfig = Objects.requireNonNull(ktConfig, "ktConfig");
        this.session = KtSftSession.open(ktConfig);
    }

    public static KTransformersFinetuneAdapter open(FactoryArgs args) {
        Objects.requireNonNull(args, "args");
        Map<String, Object> flat = args.toFlatMap();
        KtConfig kt = KtConfig.fromMap(flat);
        // Prefer factory training max_steps / output when present
        try {
            if (args.training() != null && args.training().maxSteps() > 0) {
                // rebuild with override via map
                flat.put("kt_max_steps", args.training().maxSteps());
                kt = KtConfig.fromMap(flat);
            }
        } catch (Throwable ignored) {
        }
        return new KTransformersFinetuneAdapter(args, kt);
    }

    public static KTransformersFinetuneAdapter openMini() {
        FactoryArgs fa = FactoryArgs.parse(Map.of(
                "model_name_or_path", "kt-mini-moe",
                "stage", "sft",
                "finetuning_type", "lora",
                "output_dir", "saves/kt-mini",
                "kt_max_steps", 4,
                "kt_visual_board", true));
        return open(fa);
    }

    public KtConfig ktConfig() { return ktConfig; }
    public KtSftSession session() { return session; }

    @Override
    public FactoryArgs args() {
        return args;
    }

    @Override
    public void train() {
        ensureOpen();
        session.train();
    }

    @Override
    public void requestStop() {
        session.requestStop();
    }

    @Override
    public boolean stopRequested() {
        return session.stopRequested();
    }

    @Override
    public Path export(Path dir, ExportArgs exportArgs) {
        ensureOpen();
        Path target = dir;
        if (target == null && exportArgs != null && exportArgs.exportDir() != null) {
            target = Path.of(exportArgs.exportDir());
        }
        if (target == null) {
            target = Path.of("export", "kt");
        }
        return session.export(target);
    }

    @Override
    public ChatEngine chat() {
        ensureOpen();
        return session.chat();
    }

    @Override
    public BoardState board() {
        return session.board();
    }

    @Override
    public Map<String, Double> lastMetrics() {
        return session.lastMetrics();
    }

    @Override
    public int globalStep() {
        return session.globalStep();
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KTransformersFinetuneAdapter closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        session.close();
    }
}
