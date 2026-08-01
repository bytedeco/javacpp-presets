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

package org.bytedeco.pytorch.llm.unsloth.studio.inference;

import org.bytedeco.pytorch.llm.unsloth.studio.StudioOptions;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioModelDownloader;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioModelRegistry;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.providers.ExternalProvider;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.tools.SelfHealingToolCaller;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.tools.ToolCallParser;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.tools.ToolLoopController;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.LoadRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ModelCard;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Studio inference facade: load/run local engines, external providers, tool loop, compare.
 */
public final class InferenceOrchestrator implements AutoCloseable {

    private final StudioOptions options;
    private final StudioModelRegistry registry;
    private final StudioModelDownloader downloader;
    private final ChatTemplateService templates = new ChatTemplateService();
    private final LocalInferenceEngine local;
    private final Map<String, ExternalProvider> providers = new ConcurrentHashMap<>();
    private final ToolCallParser toolParser = new ToolCallParser();
    private final SelfHealingToolCaller selfHealing = new SelfHealingToolCaller();
    private final ToolLoopController toolLoop;
    private volatile InferenceEngine active;
    private volatile String activeModelId;

    public InferenceOrchestrator(StudioOptions options, StudioModelRegistry registry,
                                 StudioModelDownloader downloader) {
        this.options = Objects.requireNonNull(options);
        this.registry = Objects.requireNonNull(registry);
        this.downloader = Objects.requireNonNull(downloader);
        this.local = new LocalInferenceEngine(templates);
        this.active = local;
        this.toolLoop = new ToolLoopController(toolParser, selfHealing);
    }

    public ChatTemplateService templates() { return templates; }
    public LocalInferenceEngine local() { return local; }
    public ToolCallParser toolParser() { return toolParser; }
    public ToolLoopController toolLoop() { return toolLoop; }

    public void registerProvider(String name, ExternalProvider provider) {
        providers.put(name, provider);
    }

    public Optional<ExternalProvider> provider(String name) {
        return Optional.ofNullable(providers.get(name));
    }

    public synchronized void load(LoadRequest request) throws Exception {
        ModelCard card = registry.resolve(request.modelPath());
        card = downloader.ensureLocal(card);
        String path = card.localPath().map(PathStr -> PathStr.toString()).orElse(request.modelPath());
        LoadRequest resolved = LoadRequest.builder()
                .modelPath(path)
                .hfToken(request.hfToken().orElse(options.hfToken().orElse(null)))
                .maxSeqLength(request.maxSeqLength())
                .loadIn4bit(request.loadIn4bit())
                .loadIn8bit(request.loadIn8bit())
                .isLora(request.isLora())
                .loraPath(request.loraPath().orElse(null))
                .ggufVariant(request.ggufVariant().orElse(null))
                .trustRemoteCode(request.trustRemoteCode() && options.allowRemoteCode())
                .chatTemplateOverride(request.chatTemplateOverride().orElse(null))
                .cacheTypeKv(request.cacheTypeKv().orElse(null))
                .gpuIds(request.gpuIds())
                .speculativeType(request.speculativeType().orElse(null))
                .nParallel(request.nParallel().orElse(null))
                .nGpuLayers(request.nGpuLayers().orElse(null))
                .tensorParallel(request.tensorParallel())
                .offloadMoeExperts(request.offloadMoeExperts())
                .build();
        local.load(resolved);
        active = local;
        final String modelId = card.id();
        activeModelId = modelId;
        request.chatTemplateOverride().ifPresent(t -> templates.setOverride(modelId, t));
    }

    public boolean isLoaded() {
        return active != null && active.isLoaded();
    }

    public Optional<String> loadedModelId() {
        return Optional.ofNullable(activeModelId);
    }

    public ChatCompletionResponse chatCompletions(ChatCompletionRequest request) throws Exception {
        // External provider by model prefix: "openai:gpt-4o", "anthropic:claude-..."
        String model = request.model().orElse(activeModelId);
        if (model != null && model.contains(":")) {
            String[] parts = model.split(":", 2);
            ExternalProvider p = providers.get(parts[0]);
            if (p != null) {
                return p.chatCompletions(request);
            }
        }
        if (active == null || !active.isLoaded()) {
            // auto-load tiny for convenience
            load(LoadRequest.builder().modelPath("studio/tiny-gpt2").loadIn4bit(false).build());
        }
        ChatCompletionResponse base = active.chatCompletions(request);
        if (!request.tools().isEmpty()) {
            return toolLoop.run(active, request, base, options.allowCodeExecution());
        }
        return base;
    }

    public CompareSession compare(String modelA, String modelB) {
        return new CompareSession(this, modelA, modelB);
    }

    public Map<String, Object> stats() {
        return active != null ? active.stats() : Map.of("loaded", false);
    }

    public synchronized void unload() {
        if (active != null) active.unload();
        activeModelId = null;
    }

    @Override
    public void close() {
        unload();
        for (ExternalProvider p : providers.values()) {
            try { p.close(); } catch (Exception ignored) {}
        }
    }
}
