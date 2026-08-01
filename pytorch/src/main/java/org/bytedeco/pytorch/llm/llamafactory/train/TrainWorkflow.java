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
package org.bytedeco.pytorch.llm.llamafactory.train;

import org.bytedeco.pytorch.llm.llamafactory.data.DataLoaderFactory;
import org.bytedeco.pytorch.llm.llamafactory.data.DatasetBuilder;
import org.bytedeco.pytorch.llm.llamafactory.extras.monitor.MonitorBundle;
import org.bytedeco.pytorch.llm.llamafactory.extras.monitor.TrainerMonitorCallback;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.Stage;
import org.bytedeco.pytorch.llm.llamafactory.hparams.TrainingArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningArgs;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Stage dispatch + resume + checkpoint orchestration for LLaMA-Factory train.
 *
 * <p>Does <strong>not</strong> reimplement TRL loss math — builds data, trainer,
 * monitors, then runs {@link BaseTrainer#train(BaseTrainer.BatchSupplier)}.
 */
public final class TrainWorkflow {

    private static final Logger LOG = Logger.getLogger(TrainWorkflow.class.getName());

    private final FactoryArgs args;
    private final LoadedModel loaded;
    private final CheckpointManager checkpoints;
    private final BoardState board;
    private final MonitorBundle monitors;
    private final AtomicBoolean stop = new AtomicBoolean(false);
    private final Map<String, Double> lastMetrics = new LinkedHashMap<>();

    private BaseTrainer trainer;
    private int globalStep;
    private boolean trained;

    public TrainWorkflow(FactoryArgs args, LoadedModel loaded) {
        this(args, loaded, null);
    }

    public TrainWorkflow(FactoryArgs args, LoadedModel loaded, BoardState board) {
        this.args = Objects.requireNonNull(args, "args");
        this.loaded = Objects.requireNonNull(loaded, "loaded");
        this.checkpoints = CheckpointManager.from(args);
        BoardState b = board;
        if (b == null && args.training().boardEnabled()) {
            b = new BoardState();
        }
        this.board = b;
        this.monitors = MonitorBundle.open(args, this.board);
    }

    public static TrainWorkflow create(FactoryArgs args) {
        args.validate();
        LoadedModel model = ModelLoader.load(args);
        return new TrainWorkflow(args, model);
    }

    public FactoryArgs args() { return args; }
    public LoadedModel loaded() { return loaded; }
    public BoardState board() { return board != null ? board : monitors.board(); }
    public CheckpointManager checkpoints() { return checkpoints; }
    public int globalStep() { return globalStep; }
    public boolean trained() { return trained; }
    public BaseTrainer trainer() { return trainer; }

    public Map<String, Double> lastMetrics() {
        synchronized (lastMetrics) {
            if (!lastMetrics.isEmpty()) {
                return Map.copyOf(lastMetrics);
            }
        }
        return monitors.lastMetrics();
    }

    public void requestStop() {
        stop.set(true);
        BoardState b = board();
        if (b != null) {
            b.requestStop();
        }
    }

    public boolean stopRequested() {
        if (stop.get()) return true;
        BoardState b = board();
        return b != null && b.stopRequested();
    }

    /**
     * Blocking train for {@link FinetuningArgs#stage()}.
     *
     * @return final global optimizer step
     */
    public int run() {
        return run(null);
    }

    /**
     * @param rawRows optional host-supplied rows; when null, demo rows for the stage are used
     *                (offline / CI path). Production hosts always pass real rows or a path loader.
     */
    public int run(List<Map<String, Object>> rawRows) {
        args.validate();
        Stage stage = args.finetuning().stage();
        TrainingArgs t = args.training();

        List<Map<String, Object>> rows = rawRows;
        if (rows == null || rows.isEmpty()) {
            rows = demoRowsFor(stage);
            LOG.info("TrainWorkflow using demo rows for stage=" + stage.wireName()
                    + " count=" + rows.size()
                    + " (pass rawRows for production data)");
        }

        DatasetBuilder builder = DatasetBuilder.from(args.data(), stage);
        List<Map<String, Object>> features = builder.buildFeatures(rows);
        if (features.isEmpty()) {
            throw new IllegalStateException("No training features produced for stage=" + stage
                    + " dataset=" + args.data().dataset());
        }

        int batchSize = Math.max(1, t.perDeviceTrainBatchSize());
        int maxSteps = t.effectiveMaxSteps(features.size());
        // resume: bump effective steps so we still do work after restore marker
        Path resumeDir = checkpoints.resolveResumeDir();
        int resumeStep = 0;
        if (resumeDir != null) {
            try {
                resumeStep = checkpoints.loadGlobalStep(resumeDir);
                LOG.info("Resume marker at step=" + resumeStep + " from " + resumeDir);
            } catch (IOException e) {
                LOG.log(Level.WARNING, "Failed reading resume state: " + e.getMessage());
            }
        }

        DataLoaderFactory loader = new DataLoaderFactory(
                features, builder.collator(), batchSize, true, false, t.dataSeed());

        // enough micro-batches for remaining steps * grad accum (with slack)
        int accum = Math.max(1, t.gradientAccumulationSteps());
        int remainingOptSteps = Math.max(1, maxSteps); // trainer counts from 0 each open
        int maxBatches = remainingOptSteps * accum + accum;

        trainer = TrainerFactory.create(args, loaded, remainingOptSteps);

        CallbackHub hub = new CallbackHub();
        hub.add(monitors.asCallback());
        TrainerMonitorCallback monitorCb = new TrainerMonitorCallback(hub, step -> {
            try {
                Map<String, Double> m = lastMetrics();
                checkpoints.save(loaded, step, m);
            } catch (IOException e) {
                throw new RuntimeException("checkpoint save failed: " + e.getMessage(), e);
            }
        }, Math.max(0, t.saveSteps()));
        trainer.addCallback(monitorCb);

        AtomicBoolean cancel = stop;
        BoardState b = board();
        BaseTrainer.BatchSupplier supplier = DataLoaderFactory.cancellable(
                loader.cycling(maxBatches),
                cancel);

        // also cancel when board stop flag flips
        BaseTrainer.BatchSupplier gated = () -> {
            if (b != null && b.stopRequested()) {
                cancel.set(true);
                return null;
            }
            return supplier.next();
        };

        LOG.info(String.format(Locale.ROOT,
                "TrainWorkflow stage=%s type=%s steps=%d batch=%d features=%d resume_step=%d",
                stage.wireName(),
                args.finetuning().finetuningType().wireName(),
                remainingOptSteps,
                batchSize,
                features.size(),
                resumeStep));

        try {
            trainer.train(gated);
        } catch (RuntimeException e) {
            if (b != null) {
                b.setStatus(BoardState.Status.FAILED);
                b.setMessage(e.getMessage() == null ? e.toString() : e.getMessage());
                b.log("[train] FAILED " + b.message());
            }
            throw e;
        }

        globalStep = trainer.globalStep();
        // expose resume offset for hosts that track cumulative steps
        if (resumeStep > 0) {
            globalStep = resumeStep + trainer.globalStep();
        }
        Map<String, Double> metrics = new LinkedHashMap<>(monitorCb.lastMetrics());
        if (metrics.isEmpty()) {
            metrics.putAll(monitors.lastMetrics());
        }
        metrics.put("global_step", (double) globalStep);
        synchronized (lastMetrics) {
            lastMetrics.clear();
            lastMetrics.putAll(metrics);
        }

        try {
            if (trainer.globalStep() > 0) {
                checkpoints.save(loaded, Math.max(1, trainer.globalStep()), metrics);
            }
        } catch (IOException e) {
            LOG.log(Level.WARNING, "Final checkpoint save failed: " + e.getMessage());
        }

        trained = true;
        return globalStep;
    }

    public void closeMonitors() {
        try {
            monitors.close();
        } catch (Exception ignored) {
        }
    }

    static List<Map<String, Object>> demoRowsFor(Stage stage) {
        return switch (stage) {
            case PT -> DatasetBuilder.demoPretrainRows();
            case DPO, ORPO, RM -> DatasetBuilder.demoPreferenceRows();
            case KTO -> DatasetBuilder.demoKtoRows();
            default -> DatasetBuilder.demoAlpacaRows();
        };
    }

    /** Materialize a few collated batches (unit / bench helper). */
    public static List<Map<String, org.bytedeco.pytorch.Tensor>> previewBatches(
            FactoryArgs args, List<Map<String, Object>> rawRows, int maxBatches) {
        Stage stage = args.finetuning().stage();
        DatasetBuilder builder = DatasetBuilder.from(args.data(), stage);
        List<Map<String, Object>> rows = rawRows == null || rawRows.isEmpty()
                ? demoRowsFor(stage) : rawRows;
        List<Map<String, Object>> feats = builder.buildFeatures(rows);
        DataLoaderFactory loader = new DataLoaderFactory(
                feats, builder.collator(),
                Math.max(1, args.training().perDeviceTrainBatchSize()),
                false, false, args.training().dataSeed());
        List<Map<String, org.bytedeco.pytorch.Tensor>> out = new ArrayList<>();
        BaseTrainer.BatchSupplier s = loader.cycling(Math.max(1, maxBatches));
        Map<String, org.bytedeco.pytorch.Tensor> b;
        while ((b = s.next()) != null) {
            out.add(b);
        }
        return out;
    }
}
