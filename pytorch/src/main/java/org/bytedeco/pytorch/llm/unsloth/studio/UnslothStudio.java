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
import org.bytedeco.pytorch.llm.unsloth.studio.observe.LiveGraphBuffer;
import org.bytedeco.pytorch.llm.unsloth.studio.rag.RagPipeline;
import org.bytedeco.pytorch.llm.unsloth.studio.train.StudioTrainingOrchestrator;
import org.bytedeco.pytorch.llm.unsloth.studio.webui.StudioBoard;

/**
 * Top-level Unsloth Studio facade (pure Java).
 *
 * <pre>{@code
 * try (UnslothStudio studio = UnslothStudio.open(StudioOptions.builder()
 *         .dataRoot(Path.of("studio-data"))
 *         .enableApi(true).apiPort(0)
 *         .enableBoard(true).boardPort(0)
 *         .build())) {
 *     studio.hub().resolve("studio/tiny-gpt2");
 *     studio.inference().load(LoadRequest.builder().modelPath("studio/tiny-gpt2").loadIn4bit(false).build());
 *     var resp = studio.inference().chatCompletions(ChatCompletionRequest.of(null, "Hello"));
 *     String runId = studio.train().start(TrainingStartRequest.builder()
 *             .modelName("studio/tiny-gpt2").maxSteps(3).loadIn4bit(false).build());
 *     studio.train().await(runId);
 * }
 * }</pre>
 */
public final class UnslothStudio implements AutoCloseable {

    private final StudioContext ctx;

    private UnslothStudio(StudioContext ctx) {
        this.ctx = ctx;
    }

    public static UnslothStudio open() throws Exception {
        return open(StudioOptions.defaults());
    }

    public static UnslothStudio open(StudioOptions options) throws Exception {
        StudioContext ctx = new StudioContext(options);
        UnslothStudio studio = new UnslothStudio(ctx);
        if (options.enableApi()) {
            ctx.server().start();
        }
        if (options.enableBoard()) {
            ctx.board().open();
        }
        if (options.enableMcp()) {
            ctx.mcp().start();
        }
        return studio;
    }

    public static StudioAdapter openAdapter(StudioOptions options) throws Exception {
        return new DefaultStudioAdapter(open(options));
    }

    public String version() { return StudioVersion.full(); }
    public StudioOptions options() { return ctx.options(); }
    public StudioContext context() { return ctx; }

    public StudioModelRegistry hub() { return ctx.registry(); }
    public StudioModelDownloader downloader() { return ctx.downloader(); }
    public StudioInventory inventory() { return ctx.inventory(); }
    public InferenceOrchestrator inference() { return ctx.inference(); }
    public StudioTrainingOrchestrator train() { return ctx.training(); }
    public ExportOrchestrator export() { return ctx.export(); }
    public RecipeService recipes() { return ctx.recipes(); }
    public RagPipeline rag() { return ctx.rag(); }
    public StudioServer server() { return ctx.server(); }
    public McpServer mcp() { return ctx.mcp(); }
    public StudioBoard board() { return ctx.board(); }
    public LiveGraphBuffer graphs() { return ctx.graphs(); }

    public StudioAdapter asAdapter() {
        return new DefaultStudioAdapter(this);
    }

    @Override
    public void close() {
        ctx.close();
    }
}
