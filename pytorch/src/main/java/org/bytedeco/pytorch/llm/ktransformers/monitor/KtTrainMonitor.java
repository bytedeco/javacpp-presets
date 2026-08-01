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
package org.bytedeco.pytorch.llm.ktransformers.monitor;

import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.ktransformers.cache.PrefixHitStats;
import org.bytedeco.pytorch.llm.ktransformers.config.KtSftConfig;
import org.bytedeco.pytorch.llm.ktransformers.moe.ExpertLoadBalanceMetrics;
import org.bytedeco.pytorch.plot.tensorboard.SummaryWriter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Unified train/infer monitor: Board + optional TensorBoard event file.
 *
 * <p>Works without a web UI — demos can poll {@link #board()} / {@link #metrics()}.
 */
public final class KtTrainMonitor implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(KtTrainMonitor.class.getName());

    private final KtMetrics metrics = new KtMetrics();
    private final BoardState board = new BoardState();
    private final BoardStateBridge bridge;
    private final ExpertHeatmapLogger expertLogger;
    private final CacheHitDashboard cacheDashboard;
    private final boolean enableTb;
    private SummaryWriter writer;
    private final Path tbDir;

    public KtTrainMonitor(KtSftConfig sft) {
        Objects.requireNonNull(sft, "sft");
        this.bridge = new BoardStateBridge(board, metrics);
        this.expertLogger = new ExpertHeatmapLogger(metrics);
        this.cacheDashboard = new CacheHitDashboard(metrics);
        this.enableTb = sft.tensorboard() || sft.visualBoard();
        Path out = sft.outputDir() != null ? sft.outputDir() : Path.of("runs", "kt-default");
        this.tbDir = out.resolve("tensorboard");
        if (enableTb) {
            try {
                Files.createDirectories(tbDir);
                this.writer = new SummaryWriter(tbDir.toString());
            } catch (IOException e) {
                LOG.log(Level.WARNING, "TensorBoard writer disabled: " + e.getMessage());
                this.writer = null;
            }
        }
    }

    public static KtTrainMonitor forDemo() {
        return new KtTrainMonitor(KtSftConfig.sftLoraDemo());
    }

    public KtMetrics metrics() { return metrics; }
    public BoardState board() { return board; }
    public BoardStateBridge bridge() { return bridge; }
    public Path tensorboardDir() { return tbDir; }
    public boolean tensorboardActive() { return writer != null; }

    public void onTrainStep(int step, double loss, double lr, double gradNorm) {
        bridge.onTrainStep(step, loss, lr, gradNorm);
        if (writer != null) {
            try {
                writer.add_scalar("train/loss", loss, step);
                writer.add_scalar("train/lr", lr, step);
                writer.add_scalar("train/grad_norm", gradNorm, step);
            } catch (IOException e) {
                LOG.log(Level.FINE, "tb write failed", e);
            }
        }
    }

    public void onExperts(ExpertLoadBalanceMetrics load) {
        expertLogger.log(load);
        if (writer != null && load != null) {
            try {
                double[] f = load.frequency();
                for (int i = 0; i < Math.min(f.length, 16); i++) {
                    writer.add_scalar("moe/expert_freq/" + i, f[i], (long) metrics.trainSteps());
                }
            } catch (IOException e) {
                LOG.log(Level.FINE, "tb moe write failed", e);
            }
        }
    }

    public void onCache(PrefixHitStats stats) {
        cacheDashboard.update(stats);
        if (writer != null && stats != null) {
            try {
                long step = metrics.generateCalls() + metrics.trainSteps();
                writer.add_scalar("cache/hit_rate", stats.hitRate(), step);
                writer.add_scalar("cache/l1_hit_rate", stats.l1HitRate(), step);
                writer.add_scalar("cache/l2_hit_rate", stats.l2HitRate(), step);
                writer.add_scalar("cache/l3_hit_rate", stats.l3HitRate(), step);
            } catch (IOException e) {
                LOG.log(Level.FINE, "tb cache write failed", e);
            }
        }
    }

    public void publish(Map<String, Double> extra) {
        bridge.publishAll(extra);
        if (writer != null && extra != null) {
            long step = Math.max(1, metrics.trainSteps());
            for (Map.Entry<String, Double> e : extra.entrySet()) {
                try {
                    writer.add_scalar(e.getKey().replace(':', '/'), e.getValue(), step);
                } catch (IOException ignored) {
                }
            }
        }
    }

    public void markRunning() { bridge.markRunning(); }
    public void markCompleted() { bridge.markCompleted(); }
    public void markFailed(String reason) { bridge.markFailed(reason); }

    /** Console-friendly status line for demos without a web server. */
    public String statusLine() {
        return String.format("status=%s step=%d loss_hist=%d metrics=%d tb=%s",
                board.status(), board.globalStep(), board.lossHistory().size(),
                board.metrics().size(), writer != null);
    }

    @Override
    public void close() {
        if (writer != null) {
            try {
                writer.close();
            } catch (Exception ignored) {
            }
            writer = null;
        }
    }
}
