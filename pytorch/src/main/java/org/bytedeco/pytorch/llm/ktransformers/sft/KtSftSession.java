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
package org.bytedeco.pytorch.llm.ktransformers.sft;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.llm.llamafactory.chat.ChatEngine;
import org.bytedeco.pytorch.llm.llamafactory.webui.BoardState;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtSftConfig;
import org.bytedeco.pytorch.llm.ktransformers.model.KtMiniMoECausalLM;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtTrainMonitor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * KT SFT session: mini (or injected) model + peft/TRL bridge + visual monitor.
 *
 * <p>Upstream SFT Quick Start entry for pure-Java host meshes.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class KtSftSession implements AutoCloseable {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final KtConfig config;
    private final KtMiniMoECausalLM model;
    private final boolean ownsModel;
    private final KtTrainMonitor monitor;
    private final KtSftBridge bridge;
    private final HeterogeneousTrainerHooks hooks;
    private final FreezeAndOffloadPolicy offload;
    private final AtomicBoolean stop = new AtomicBoolean(false);
    private final AtomicInteger globalStep = new AtomicInteger(0);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private volatile double lastLoss = Double.NaN;
    private volatile Map<String, Double> lastMetrics = Collections.emptyMap();

    public KtSftSession(KtConfig config) {
        this.config = Objects.requireNonNull(config, "config");
        this.model = new KtMiniMoECausalLM(config);
        this.ownsModel = true;
        this.monitor = new KtTrainMonitor(config.sft());
        this.bridge = new KtSftBridge(config, monitor);
        this.offload = FreezeAndOffloadPolicy.from(config);
        this.hooks = new HeterogeneousTrainerHooks(offload, monitor, stop::get);
        bridge.maybeAttachPeft(model);
    }

    public static KtSftSession openMini() {
        return new KtSftSession(KtConfig.miniDemo());
    }

    public static KtSftSession open(KtConfig config) {
        return new KtSftSession(config);
    }

    public KtConfig config() { return config; }
    public KtMiniMoECausalLM model() { return model; }
    public KtTrainMonitor monitor() { return monitor; }
    public BoardState board() { return monitor.board(); }
    public int globalStep() { return globalStep.get(); }
    public double lastLoss() { return lastLoss; }
    public Map<String, Double> lastMetrics() { return lastMetrics; }

    public void requestStop() {
        stop.set(true);
        monitor.board().requestStop();
    }

    public boolean stopRequested() {
        return stop.get() || monitor.board().stopRequested();
    }

    /**
     * Run SFT or DPO synthetic steps according to {@link KtSftConfig#stage()}.
     */
    public void train() {
        ensureOpen();
        stop.set(false);
        monitor.board().clearStop();
        KtSftConfig sft = config.sft();
        int steps = Math.max(1, sft.maxSteps());
        int seqLen = Math.min(32, Math.max(8, (int) config.inference().maxSeqLen() / 4));
        double loss;
        if (sft.stage() == KtSftConfig.Stage.DPO) {
            DpoWithKtRuntime dpo = new DpoWithKtRuntime(config, monitor);
            loss = dpo.runSyntheticSteps(model, hooks, steps, seqLen, 42L);
        } else {
            loss = bridge.runSyntheticSteps(model, hooks, steps, seqLen, 42L);
        }
        lastLoss = loss;
        globalStep.set(monitor.board().globalStep());
        lastMetrics = monitor.metrics().snapshot();
    }

    /** Export a tiny marker + metrics snapshot under {@code dir}. */
    public Path export(Path dir) {
        ensureOpen();
        try {
            Files.createDirectories(dir);
            Path marker = dir.resolve("kt_export_ok.txt");
            String body = "KTransformers-Java export\n"
                    + "model=" + config.modelNameOrPath() + "\n"
                    + "step=" + globalStep.get() + "\n"
                    + "last_loss=" + lastLoss + "\n"
                    + "stage=" + config.sft().stage() + "\n";
            Files.writeString(marker, body);
            Path metricsFile = dir.resolve("metrics.txt");
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, Double> e : lastMetrics.entrySet()) {
                sb.append(e.getKey()).append('=').append(e.getValue()).append('\n');
            }
            Files.writeString(metricsFile, sb.toString());
            return dir;
        } catch (Exception e) {
            throw new IllegalStateException("export failed: " + e.getMessage(), e);
        }
    }

    public ChatEngine chat() {
        ensureOpen();
        return new ChatEngine() {
            @Override
            public String chat(String userMessage) {
                if (userMessage == null) userMessage = "";
                int[] prompt = new int[Math.min(8, Math.max(1, userMessage.length()))];
                for (int i = 0; i < prompt.length; i++) {
                    prompt[i] = Math.floorMod(userMessage.charAt(i % userMessage.length()),
                            model.vocabSize());
                }
                if (prompt.length == 0) prompt = new int[]{1};
                int[] out = model.generateGreedy(prompt, 4);
                StringBuilder sb = new StringBuilder("kt-tokens:");
                for (int i = 0; i < out.length; i++) {
                    if (i > 0) sb.append(',');
                    sb.append(out[i]);
                }
                return sb.toString();
            }
        };
    }

    private void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KtSftSession closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        try {
            monitor.close();
        } finally {
            if (ownsModel) {
                try {
                    model.close();
                } catch (Throwable ignored) {
                }
            }
        }
    }
}
