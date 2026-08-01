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
import org.bytedeco.pytorch.llm.llamafactory.export.ModelExporter;
import org.bytedeco.pytorch.llm.llamafactory.hparams.ExportArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.llamafactory.train.TrainWorkflow;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Default {@link FinetuneAdapter} implementation — owns model load, train workflow,
 * export and in-process chat.
 *
 * <p>Host platforms obtain instances via {@link LlamaFactory#open(FactoryArgs)}.
 */
public final class DefaultFinetuneJob implements FinetuneAdapter {

    private static final Logger LOG = Logger.getLogger(DefaultFinetuneJob.class.getName());

    private final FactoryArgs args;
    private final LoadedModel loaded;
    private final BoardState board;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private TrainWorkflow workflow;
    private ChatEngine chatEngine;
    private int globalStep;
    private Map<String, Double> lastMetrics = Collections.emptyMap();
    private final List<Map<String, Object>> rawRows;

    public DefaultFinetuneJob(FactoryArgs args) {
        this(args, null, null);
    }

    public DefaultFinetuneJob(FactoryArgs args, List<Map<String, Object>> rawRows) {
        this(args, rawRows, null);
    }

    public DefaultFinetuneJob(FactoryArgs args, List<Map<String, Object>> rawRows, BoardState board) {
        this.args = Objects.requireNonNull(args, "args");
        this.args.validate();
        this.rawRows = rawRows;
        this.loaded = ModelLoader.load(args);
        BoardState b = board;
        if (b == null && args.training().boardEnabled()) {
            b = new BoardState();
        }
        this.board = b;
        this.workflow = new TrainWorkflow(args, loaded, this.board);
        LOG.info("DefaultFinetuneJob opened model=" + args.model().modelNameOrPath()
                + " stage=" + args.finetuning().stage().wireName()
                + " type=" + args.finetuning().finetuningType().wireName());
    }

    /** Package-private: already-loaded model (tests / advanced hosts). */
    DefaultFinetuneJob(FactoryArgs args, LoadedModel loaded, BoardState board,
                       List<Map<String, Object>> rawRows) {
        this.args = Objects.requireNonNull(args, "args");
        this.loaded = Objects.requireNonNull(loaded, "loaded");
        this.rawRows = rawRows;
        this.board = board;
        this.workflow = new TrainWorkflow(args, loaded, board);
    }

    @Override
    public FactoryArgs args() {
        return args;
    }

    public LoadedModel loaded() {
        return loaded;
    }

    @Override
    public void train() {
        ensureOpen();
        globalStep = workflow.run(rawRows);
        lastMetrics = workflow.lastMetrics();
        // refresh chat engine against post-train weights
        chatEngine = null;
    }

    @Override
    public void requestStop() {
        if (workflow != null) {
            workflow.requestStop();
        }
        if (board != null) {
            board.requestStop();
        }
    }

    @Override
    public boolean stopRequested() {
        if (workflow != null && workflow.stopRequested()) {
            return true;
        }
        return board != null && board.stopRequested();
    }

    @Override
    public Path export(Path dir, ExportArgs exportArgs) {
        ensureOpen();
        ExportArgs ex = exportArgs == null ? ExportArgs.defaults() : exportArgs;
        if (dir != null) {
            ex = ExportArgs.builder()
                    .exportDir(dir.toString())
                    .exportSize(ex.exportSize())
                    .exportDevice(ex.exportDevice())
                    .exportDtype(ex.exportDtype())
                    .exportLegacyFormat(ex.exportLegacyFormat())
                    .exportHubModelId(ex.exportHubModelId())
                    .mergeAdapters(ex.mergeAdapters())
                    .build();
        }
        try {
            return ModelExporter.export(args, loaded, ex);
        } catch (IOException e) {
            throw new RuntimeException("export failed: " + e.getMessage(), e);
        }
    }

    @Override
    public ChatEngine chat() {
        ensureOpen();
        if (chatEngine == null) {
            chatEngine = ChatModel.fromLoaded(loaded, args.generating(), args.data().template());
        }
        return chatEngine;
    }

    @Override
    public BoardState board() {
        if (board != null) {
            return board;
        }
        return workflow == null ? null : workflow.board();
    }

    @Override
    public Map<String, Double> lastMetrics() {
        if (lastMetrics != null && !lastMetrics.isEmpty()) {
            return lastMetrics;
        }
        return workflow == null ? Collections.emptyMap() : workflow.lastMetrics();
    }

    @Override
    public int globalStep() {
        if (globalStep > 0) {
            return globalStep;
        }
        return workflow == null ? 0 : workflow.globalStep();
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        if (chatEngine != null) {
            try { chatEngine.close(); } catch (Exception ignored) {}
            chatEngine = null;
        }
        if (workflow != null) {
            workflow.closeMonitors();
        }
        if (loaded != null) {
            try { loaded.close(); } catch (Exception ignored) {}
        }
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("FinetuneAdapter already closed");
        }
    }
}
