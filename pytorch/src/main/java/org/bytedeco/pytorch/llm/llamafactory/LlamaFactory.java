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
import org.bytedeco.pytorch.llm.llamafactory.chat.ChatModel;
import org.bytedeco.pytorch.llm.llamafactory.eval.EvalResult;
import org.bytedeco.pytorch.llm.llamafactory.eval.EvalRunner;
import org.bytedeco.pytorch.llm.llamafactory.export.ModelExporter;
import org.bytedeco.pytorch.llm.llamafactory.hparams.EvaluationArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ExportArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.InferArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.llamafactory.webui.LlamaBoard;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Main LLaMA-Factory entry point and product surface.
 *
 * <p>Static factory methods mirror the high-level API from the plan:
 * <ul>
 *   <li>{@code open(FactoryArgs)} — returns a {@link FinetuneAdapter} for training / export / chat</li>
 *   <li>{@code train(FactoryArgs)} — convenience for non-interactive SFT / PT runs</li>
 *   <li>{@code chat(InferArgs)} — lightweight inference surface</li>
 *   <li>{@code serveApi(InferArgs)} — OpenAI-compatible HTTP (when api package present)</li>
 *   <li>{@code serveBoard(FactoryArgs)} — embedded {@link LlamaBoard}</li>
 * </ul>
 *
 * <p>Usage example (tiny synthetic model):
 * <pre>{@code
 * FactoryArgs args = FactoryArgs.builder()
 *     .model(ModelArgs.builder().modelNameOrPath("tiny-gpt2").build())
 *     .data(d -> d.dataset("alpaca_en_demo").maxSamples(4))
 *     .finetuning(f -> f.stage(Stage.SFT).finetuningType(FinetuningType.LORA))
 *     .training(t -> t.outputDir("saves/tiny").perDeviceTrainBatchSize(1)
 *         .learningRate(5e-5).maxSteps(2).build())
 *     .build();
 *
 * try (FinetuneAdapter job = LlamaFactory.open(args)) {
 *     job.train();
 *     Path merged = job.export(Path.of("export"), ExportArgs.builder().mergeAdapters(true).build());
 *     String reply = job.chat().chat("Hello world");
 * }
 * }</pre>
 */
public final class LlamaFactory {

    private static final Logger LOG = Logger.getLogger(LlamaFactory.class.getName());

    private LlamaFactory() {}

    /**
     * Opens a training / export / chat job with the given arguments.
     *
     * @param args aggregated hyper-parameters (model / data / finetuning / …)
     * @return FinetuneAdapter instance (AutoCloseable)
     */
    public static FinetuneAdapter open(FactoryArgs args) {
        Objects.requireNonNull(args, "args");
        args.validate();
        return new DefaultFinetuneJob(args);
    }

    /**
     * Opens a job with host-supplied raw dataset rows (skips demo rows).
     */
    public static FinetuneAdapter open(FactoryArgs args, List<Map<String, Object>> rawRows) {
        Objects.requireNonNull(args, "args");
        args.validate();
        return new DefaultFinetuneJob(args, rawRows);
    }

    /**
     * Opens a job bound to an existing board (train + visualize).
     */
    public static FinetuneAdapter open(FactoryArgs args, List<Map<String, Object>> rawRows, BoardState board) {
        Objects.requireNonNull(args, "args");
        args.validate();
        return new DefaultFinetuneJob(args, rawRows, board);
    }

    /**
     * Convenience for non-interactive training only (no board / API).
     */
    public static void train(FactoryArgs args) {
        try (FinetuneAdapter job = open(args)) {
            job.train();
        }
    }

    /**
     * Train with explicit rows.
     */
    public static void train(FactoryArgs args, List<Map<String, Object>> rawRows) {
        try (FinetuneAdapter job = open(args, rawRows)) {
            job.train();
        }
    }

    /**
     * Lightweight inference surface (no training). Loads model via {@link ModelLoader}.
     */
    public static ChatEngine chat(InferArgs args) {
        Objects.requireNonNull(args, "args");
        ModelArgs modelArgs = ModelArgs.builder()
                .modelNameOrPath(args.modelNameOrPath())
                .adapterNameOrPath(args.adapterNameOrPath())
                .quantizationMethod(args.quantizationMethod())
                .flashAttn(args.flashAttn())
                .useUnsloth(args.useUnsloth())
                .build();
        FactoryArgs fa = FactoryArgs.builder()
                .model(modelArgs)
                .generating(args.generating())
                .infer(args)
                .build();
        LoadedModel loaded = ModelLoader.load(fa);
        return ChatModel.fromLoaded(loaded, args);
    }

    /**
     * Starts OpenAI-compatible HTTP server.
     *
     * <p>Delegates to {@code factory.api.OpenAiServer} when present; otherwise throws
     * with a clear message so hosts know to enable the api package.
     */
    public static AutoCloseable serveApi(InferArgs args) {
        Objects.requireNonNull(args, "args");
        try {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.llm.llamafactory.api.OpenAiServer");
            var m = cls.getMethod("start", InferArgs.class);
            Object server = m.invoke(null, args);
            LOG.info("OpenAI API server started host=" + args.host() + " port=" + args.port());
            return (AutoCloseable) server;
        } catch (ClassNotFoundException e) {
            throw new UnsupportedOperationException(
                    "factory.api.OpenAiServer not on classpath yet — implement api package "
                            + "or use LlamaFactory.chat(InferArgs) in-process", e);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("Failed to start OpenAiServer: " + e.getMessage(), e);
        }
    }

    /**
     * Starts embedded training dashboard (LlamaBoard) and returns the board handle.
     * Training is <em>not</em> started automatically — use {@link #open} + board, or
     * {@link LlamaBoard#attach(FinetuneAdapter)}.
     */
    public static LlamaBoard serveBoard(FactoryArgs args) {
        Objects.requireNonNull(args, "args");
        args.validate();
        int port = args.training().boardPort() > 0 ? args.training().boardPort() : 7860;
        try {
            LlamaBoard board = LlamaBoard.start(port, args);
            LOG.info("LlamaBoard UI at " + board.uiUrl());
            return board;
        } catch (Exception e) {
            throw new RuntimeException("Failed to start LlamaBoard: " + e.getMessage(), e);
        }
    }

    /**
     * Export / merge adapters + weights (safetensors / config).
     */
    public static Path export(FactoryArgs args, ExportArgs exportArgs) {
        try (FinetuneAdapter job = open(args)) {
            ExportArgs ex = exportArgs == null ? ExportArgs.defaults() : exportArgs;
            return job.export(Paths.get(ex.exportDir()), ex);
        }
    }

    /**
     * Evaluation harness (MMLU-style multi-choice).
     */
    public static EvalResult eval(EvaluationArgs args) {
        Objects.requireNonNull(args, "args");
        return EvalRunner.run(args);
    }

    /**
     * Direct export from an already-open loaded model (advanced).
     */
    public static Path exportLoaded(FactoryArgs args, LoadedModel loaded, ExportArgs exportArgs) {
        try {
            return ModelExporter.export(args, loaded, exportArgs);
        } catch (Exception e) {
            throw new RuntimeException("export failed: " + e.getMessage(), e);
        }
    }

    /**
     * Factory version string (for logging / compatibility).
     */
    public static String version() {
        return FactoryVersion.VERSION;
    }
}
