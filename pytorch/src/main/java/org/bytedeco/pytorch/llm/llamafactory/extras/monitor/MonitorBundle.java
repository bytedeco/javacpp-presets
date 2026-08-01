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
package org.bytedeco.pytorch.llm.llamafactory.extras.monitor;

import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.TrainingArgs;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;
import org.bytedeco.pytorch.plot.swanlab.SwanLabClient;
import org.bytedeco.pytorch.plot.swanlab.SwanLabLocalServer;
import org.bytedeco.pytorch.plot.swanlab.SwanLabTrainingMonitor;
import org.bytedeco.pytorch.plot.tensorboard.SummaryWriter;
import org.bytedeco.pytorch.plot.wandb.WandbClient;
import org.bytedeco.pytorch.plot.wandb.WandbLocalServer;
import org.bytedeco.pytorch.plot.wandb.WandbTrainingMonitor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Multi-backend training monitor bundle (TensorBoard + WandB + SwanLab + Board).
 *
 * <p>Created from {@code TrainingArgs.reportTo} (comma-separated:
 * {@code none|tensorboard|wandb|swanlab|all}) and optional {@link BoardState}.
 *
 * <p>WandB / SwanLab open via their offline local-server builders when available;
 * failures are logged and that backend is skipped so training still runs.
 */
public final class MonitorBundle implements AutoCloseable, TrainerCallback {

    private static final Logger LOG = Logger.getLogger(MonitorBundle.class.getName());

    private final SummaryWriter tensorboard;
    private final WandbTrainingMonitor wandb;
    private final SwanLabTrainingMonitor swanlab;
    private final BoardState board;
    private final WandbLocalServer wandbServer;
    private final SwanLabLocalServer swanServer;
    private final boolean ownsTb;
    private final boolean ownsWandb;
    private final boolean ownsSwan;
    private final Map<String, Double> lastMetrics = new LinkedHashMap<>();

    private MonitorBundle(
            SummaryWriter tensorboard,
            boolean ownsTb,
            WandbTrainingMonitor wandb,
            boolean ownsWandb,
            WandbLocalServer wandbServer,
            SwanLabTrainingMonitor swanlab,
            boolean ownsSwan,
            SwanLabLocalServer swanServer,
            BoardState board) {
        this.tensorboard = tensorboard;
        this.ownsTb = ownsTb;
        this.wandb = wandb;
        this.ownsWandb = ownsWandb;
        this.wandbServer = wandbServer;
        this.swanlab = swanlab;
        this.ownsSwan = ownsSwan;
        this.swanServer = swanServer;
        this.board = board;
    }

    public static MonitorBundle open(FactoryArgs args, BoardState board) {
        Objects.requireNonNull(args, "args");
        TrainingArgs t = args.training();
        String report = t.reportTo() == null ? "none" : t.reportTo().trim().toLowerCase(Locale.ROOT);
        boolean all = report.contains("all");
        boolean wantTb = all || report.contains("tensorboard") || report.contains("tb");
        boolean wantWandb = all || report.contains("wandb");
        boolean wantSwan = all || report.contains("swanlab") || report.contains("swan");

        SummaryWriter tb = null;
        boolean ownsTb = false;
        if (wantTb) {
            try {
                Path dir = Path.of(t.outputDir() == null ? "saves/factory" : t.outputDir(), "runs");
                Files.createDirectories(dir);
                tb = new SummaryWriter(dir.toString());
                ownsTb = true;
            } catch (Exception e) {
                LOG.log(Level.WARNING, "TensorBoard SummaryWriter open failed: " + e.getMessage());
            }
        }

        WandbLocalServer wbServer = null;
        WandbTrainingMonitor wb = null;
        boolean ownsWb = false;
        if (wantWandb) {
            try {
                wbServer = WandbLocalServer.start(0);
                WandbClient client = WandbClient.newBuilder()
                        .offline(wbServer)
                        .build();
                String run = t.runName() == null || t.runName().isBlank() ? "factory-run" : t.runName();
                wb = new WandbTrainingMonitor(client, run, safeFlat(args), true);
                ownsWb = true;
            } catch (Throwable e) {
                LOG.log(Level.WARNING, "WandB monitor open failed: " + e.getMessage());
                if (wbServer != null) {
                    try { wbServer.close(); } catch (Exception ignored) {}
                    wbServer = null;
                }
            }
        }

        SwanLabLocalServer swanServer = null;
        SwanLabTrainingMonitor swan = null;
        boolean ownsSwan = false;
        if (wantSwan) {
            try {
                swanServer = SwanLabLocalServer.start(0);
                SwanLabClient client = SwanLabClient.newBuilder()
                        .offline(swanServer)
                        .build();
                swan = new SwanLabTrainingMonitor(client, safeFlat(args), true);
                ownsSwan = true;
            } catch (Throwable e) {
                LOG.log(Level.WARNING, "SwanLab monitor open failed: " + e.getMessage());
                if (swanServer != null) {
                    try { swanServer.close(); } catch (Exception ignored) {}
                    swanServer = null;
                }
            }
        }

        BoardState b = board;
        if (b == null && t.boardEnabled()) {
            b = new BoardState();
        }
        return new MonitorBundle(tb, ownsTb, wb, ownsWb, wbServer, swan, ownsSwan, swanServer, b);
    }

    /** Board-only / no external reporters. */
    public static MonitorBundle boardOnly(BoardState board) {
        return new MonitorBundle(null, false, null, false, null, null, false, null, board);
    }

    public BoardState board() {
        return board;
    }

    public Map<String, Double> lastMetrics() {
        synchronized (lastMetrics) {
            return Collections.unmodifiableMap(new LinkedHashMap<>(lastMetrics));
        }
    }

    public TrainerCallback asCallback() {
        return this;
    }

    public List<TrainerCallback> callbacks() {
        List<TrainerCallback> list = new ArrayList<>(1);
        list.add(this);
        return list;
    }

    @Override
    public void onTrainBegin(BaseTrainer trainer) {
        if (board != null) {
            board.clearStop();
            board.setStatus(BoardState.Status.RUNNING);
            board.setMessage("train begin");
            board.log("[train] begin max_steps=" + trainer.config().maxSteps());
        }
    }

    @Override
    public void onTrainEnd(BaseTrainer trainer) {
        if (board != null) {
            if (board.stopRequested()) {
                board.setStatus(BoardState.Status.STOPPED);
                board.setMessage("stopped");
            } else {
                board.setStatus(BoardState.Status.COMPLETED);
                board.setMessage("completed step=" + trainer.globalStep());
            }
            board.setGlobalStep(trainer.globalStep());
            board.log("[train] end step=" + trainer.globalStep());
        }
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        record(step, metrics);
        if (board != null) {
            board.setGlobalStep(step);
            if (metrics != null && metrics.containsKey("loss") && metrics.get("loss") != null) {
                board.recordLoss(metrics.get("loss"));
            }
            board.putMetrics(metrics);
        }
    }

    @Override
    public void onLog(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        record(step, metrics);
        if (board != null) {
            board.putMetrics(metrics);
            if (metrics != null && metrics.containsKey("loss") && metrics.get("loss") != null) {
                board.log(String.format(Locale.ROOT, "step=%d loss=%.6f", step, metrics.get("loss")));
            }
        }
    }

    private void record(int step, Map<String, Double> metrics) {
        if (metrics == null || metrics.isEmpty()) {
            return;
        }
        synchronized (lastMetrics) {
            lastMetrics.clear();
            lastMetrics.putAll(metrics);
            lastMetrics.put("global_step", (double) step);
        }
        if (tensorboard != null) {
            try {
                for (Map.Entry<String, Double> e : metrics.entrySet()) {
                    if (e.getKey() != null && e.getValue() != null) {
                        tensorboard.add_scalar(e.getKey(), e.getValue(), step);
                    }
                }
            } catch (IOException ex) {
                LOG.log(Level.FINE, "tb write failed", ex);
            }
        }
        if (wandb != null) {
            try {
                wandb.setStep(step);
                wandb.log(metrics);
            } catch (Exception ex) {
                LOG.log(Level.FINE, "wandb log failed", ex);
            }
        }
        if (swanlab != null) {
            try {
                swanlab.setStep(step);
                swanlab.log(metrics);
            } catch (Exception ex) {
                LOG.log(Level.FINE, "swanlab log failed", ex);
            }
        }
    }

    @Override
    public void close() {
        if (ownsTb && tensorboard != null) {
            try { tensorboard.close(); } catch (Exception ignored) {}
        }
        if (ownsWandb && wandb != null) {
            try { wandb.close(); } catch (Exception ignored) {}
        }
        if (wandbServer != null) {
            try { wandbServer.close(); } catch (Exception ignored) {}
        }
        if (ownsSwan && swanlab != null) {
            try { swanlab.close(); } catch (Exception ignored) {}
        }
        if (swanServer != null) {
            try { swanServer.close(); } catch (Exception ignored) {}
        }
    }

    private static Map<String, Object> safeFlat(FactoryArgs args) {
        try {
            return new LinkedHashMap<>(args.toFlatMap());
        } catch (Throwable t) {
            return Map.of();
        }
    }
}
