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

package org.bytedeco.pytorch.llm.llamacpp;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Facade for enterprise llama.cpp Java surface.
 *
 * <pre>{@code
 * try (LlamaEngine eng = LlamaCpp.open(LlamaRuntimeConfig.builder()
 *         .modelPath("model.gguf")
 *         .backend(LlamaBackend.IN_PROCESS)
 *         .nCtx(2048)
 *         .build())) {
 *     eng.load();
 *     System.out.println(eng.complete("Hello", LlamaSamplingParams.greedy(32)));
 * }
 * }</pre>
 */
public final class LlamaCpp {

    public static final String VERSION = "1.0.0-enterprise";

    private LlamaCpp() {}

    public static String version() { return VERSION; }

    public static LlamaEngine open(LlamaRuntimeConfig config) {
        Objects.requireNonNull(config, "config");
        LlamaBackend backend = config.backend();
        if (backend == LlamaBackend.AUTO) {
            backend = LlamaProcessManager.findLlamaServer(config) != null
                    ? LlamaBackend.PROCESS_SERVER
                    : LlamaBackend.IN_PROCESS;
        }
        return switch (backend) {
            case PROCESS_SERVER -> new ProcessLlamaRuntime(config);
            case IN_PROCESS, AUTO -> new InProcessLlamaEngine(
                    config.backend() == LlamaBackend.AUTO
                            ? withBackend(config, LlamaBackend.IN_PROCESS)
                            : config);
        };
    }

    public static LlamaEngine openInProcess(Path model) {
        return open(LlamaRuntimeConfig.builder()
                .modelPath(model)
                .backend(LlamaBackend.IN_PROCESS)
                .build());
    }

    public static LlamaEngine openServer(Path model, Path llamaServerBin) {
        return open(LlamaRuntimeConfig.builder()
                .modelPath(model)
                .backend(LlamaBackend.PROCESS_SERVER)
                .llamaServerBin(llamaServerBin)
                .serverPort(0)
                .build());
    }

    private static LlamaRuntimeConfig withBackend(LlamaRuntimeConfig cfg, LlamaBackend b) {
        return LlamaRuntimeConfig.builder()
                .modelPath(cfg.modelPath())
                .backend(b)
                .nCtx(cfg.nCtx())
                .nBatch(cfg.nBatch())
                .nThreads(cfg.nThreads())
                .nGpuLayers(cfg.nGpuLayers())
                .offloadMoeExperts(cfg.offloadMoeExperts())
                .tensorSplit(cfg.tensorSplit())
                .mainGpu(cfg.mainGpu())
                .flashAttn(cfg.flashAttn())
                .cacheTypeK(cfg.cacheTypeK().orElse(null))
                .cacheTypeV(cfg.cacheTypeV().orElse(null))
                .llamaServerBin(cfg.llamaServerBin().orElse(null))
                .llamaCliBin(cfg.llamaCliBin().orElse(null))
                .serverHost(cfg.serverHost())
                .serverPort(cfg.serverPort())
                .serverStartTimeoutMs(cfg.serverStartTimeoutMs())
                .chatTemplate(cfg.chatTemplate().orElse(null))
                .verbose(cfg.verbose())
                .extraServerArgs(cfg.extraServerArgs())
                .useMmap(cfg.useMmap())
                .useMlock(cfg.useMlock())
                .build();
    }
}
