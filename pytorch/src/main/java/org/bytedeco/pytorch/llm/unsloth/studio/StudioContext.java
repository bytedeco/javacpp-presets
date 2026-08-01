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

package org.bytedeco.pytorch.llm.unsloth.studio;

import org.bytedeco.pytorch.llm.unsloth.studio.api.StudioServer;
import org.bytedeco.pytorch.llm.unsloth.studio.data.RecipeService;
import org.bytedeco.pytorch.llm.unsloth.studio.export.ExportOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioInventory;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioModelDownloader;
import org.bytedeco.pytorch.llm.unsloth.studio.hub.StudioModelRegistry;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.InferenceOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.mcp.McpServer;
import org.bytedeco.pytorch.llm.unsloth.studio.mcp.McpToolRegistry;
import org.bytedeco.pytorch.llm.unsloth.studio.observe.LiveGraphBuffer;
import org.bytedeco.pytorch.llm.unsloth.studio.observe.MetricsSink;
import org.bytedeco.pytorch.llm.unsloth.studio.observe.TensorBoardSink;
import org.bytedeco.pytorch.llm.unsloth.studio.rag.RagPipeline;
import org.bytedeco.pytorch.llm.unsloth.studio.train.StudioTrainingOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.train.TrainingProgressBus;
import org.bytedeco.pytorch.llm.unsloth.studio.train.TrainingRunStore;
import org.bytedeco.pytorch.llm.unsloth.studio.util.StudioPaths;
import org.bytedeco.pytorch.llm.unsloth.studio.webui.StudioBoard;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Wired runtime graph shared by {@link UnslothStudio}. */
public final class StudioContext implements AutoCloseable {

    private final StudioOptions options;
    private final StudioModelRegistry registry;
    private final StudioModelDownloader downloader;
    private final StudioInventory inventory;
    private final InferenceOrchestrator inference;
    private final TrainingProgressBus progressBus;
    private final TrainingRunStore runStore;
    private final LiveGraphBuffer graphs;
    private final StudioTrainingOrchestrator training;
    private final ExportOrchestrator export;
    private final RecipeService recipes;
    private final RagPipeline rag;
    private final McpToolRegistry mcpTools;
    private final McpServer mcp;
    private final StudioServer server;
    private final StudioBoard board;
    private final List<MetricsSink> sinks;

    public StudioContext(StudioOptions options) throws Exception {
        this.options = Objects.requireNonNull(options);
        StudioPaths.ensureLayout(options);

        this.registry = new StudioModelRegistry(options.modelsDir());
        this.downloader = new StudioModelDownloader(options, registry);
        this.inventory = new StudioInventory(options.modelsDir(), options.datasetsDir(), registry);
        this.inference = new InferenceOrchestrator(options, registry, downloader);
        this.progressBus = new TrainingProgressBus();
        this.runStore = new TrainingRunStore(options.runsDir());
        this.graphs = new LiveGraphBuffer();

        this.sinks = new ArrayList<>();
        this.sinks.add(graphs);
        if (options.tensorBoardSink()) {
            this.sinks.add(new TensorBoardSink(options.tensorBoardLogDir()));
        }

        this.training = new StudioTrainingOrchestrator(options, runStore, progressBus, sinks);
        this.export = new ExportOrchestrator();
        this.recipes = new RecipeService(options.recipesDir());
        this.rag = new RagPipeline(null, null, null);

        this.mcpTools = new McpToolRegistry();
        org.bytedeco.pytorch.llm.unsloth.studio.mcp.McpStudioTools.registerAll(
                mcpTools, registry, training, export);
        this.mcp = new McpServer(mcpTools);

        this.server = new StudioServer(options, inference, training, export, registry, inventory);
        this.board = new StudioBoard(options, graphs, progressBus);
    }

    public StudioOptions options() { return options; }
    public StudioModelRegistry registry() { return registry; }
    public StudioModelDownloader downloader() { return downloader; }
    public StudioInventory inventory() { return inventory; }
    public InferenceOrchestrator inference() { return inference; }
    public StudioTrainingOrchestrator training() { return training; }
    public ExportOrchestrator export() { return export; }
    public RecipeService recipes() { return recipes; }
    public RagPipeline rag() { return rag; }
    public McpServer mcp() { return mcp; }
    public McpToolRegistry mcpTools() { return mcpTools; }
    public StudioServer server() { return server; }
    public StudioBoard board() { return board; }
    public LiveGraphBuffer graphs() { return graphs; }
    public TrainingProgressBus progressBus() { return progressBus; }

    @Override
    public void close() {
        try { board.close(); } catch (Exception ignored) {}
        try { server.close(); } catch (Exception ignored) {}
        try { mcp.close(); } catch (Exception ignored) {}
        try { training.close(); } catch (Exception ignored) {}
        try { inference.close(); } catch (Exception ignored) {}
        for (MetricsSink s : sinks) {
            try { s.close(); } catch (Exception ignored) {}
        }
    }
}
