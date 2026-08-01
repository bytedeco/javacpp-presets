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
package org.bytedeco.pytorch.llm.ktransformers.config;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Supervised fine-tuning / preference-tuning configuration for KT × factory/TRL.
 *
 * <p>Maps to upstream "SFT Quick Start" and LLaMA-Factory / RL-DPO integration:
 * LoRA/QLoRA, stage selection, offload policy, and visual board enablement.
 */
public final class KtSftConfig {

    public enum Stage {
        SFT,
        DPO,
        KTO,
        ORPO,
        PPO,
        GRPO,
        RM
    }

    public enum PeftKind {
        NONE,
        LORA,
        QLORA,
        DORA
    }

    private final Stage stage;
    private final PeftKind peftKind;
    private final int loraR;
    private final double loraAlpha;
    private final double loraDropout;
    private final int maxSteps;
    private final int loggingSteps;
    private final double learningRate;
    private final int batchSize;
    private final int gradAccum;
    private final boolean offloadFrozen;
    private final boolean visualBoard;
    private final boolean tensorboard;
    private final boolean wandb;
    private final Path outputDir;
    private final Path datasetPath;
    private final String reportTo;

    private KtSftConfig(Builder b) {
        this.stage = Objects.requireNonNull(b.stage, "stage");
        this.peftKind = Objects.requireNonNull(b.peftKind, "peftKind");
        if (b.loraR < 0) throw new IllegalArgumentException("loraR must be >= 0");
        if (b.maxSteps < 1) throw new IllegalArgumentException("maxSteps must be >= 1");
        if (b.batchSize < 1) throw new IllegalArgumentException("batchSize must be >= 1");
        if (b.gradAccum < 1) throw new IllegalArgumentException("gradAccum must be >= 1");
        this.loraR = b.loraR;
        this.loraAlpha = b.loraAlpha;
        this.loraDropout = b.loraDropout;
        this.maxSteps = b.maxSteps;
        this.loggingSteps = Math.max(1, b.loggingSteps);
        this.learningRate = b.learningRate;
        this.batchSize = b.batchSize;
        this.gradAccum = b.gradAccum;
        this.offloadFrozen = b.offloadFrozen;
        this.visualBoard = b.visualBoard;
        this.tensorboard = b.tensorboard;
        this.wandb = b.wandb;
        this.outputDir = b.outputDir;
        this.datasetPath = b.datasetPath;
        this.reportTo = b.reportTo;
    }

    public Stage stage() { return stage; }
    public PeftKind peftKind() { return peftKind; }
    public int loraR() { return loraR; }
    public double loraAlpha() { return loraAlpha; }
    public double loraDropout() { return loraDropout; }
    public int maxSteps() { return maxSteps; }
    public int loggingSteps() { return loggingSteps; }
    public double learningRate() { return learningRate; }
    public int batchSize() { return batchSize; }
    public int gradAccum() { return gradAccum; }
    public boolean offloadFrozen() { return offloadFrozen; }
    public boolean visualBoard() { return visualBoard; }
    public boolean tensorboard() { return tensorboard; }
    public boolean wandb() { return wandb; }
    public Path outputDir() { return outputDir; }
    public Path datasetPath() { return datasetPath; }
    public String reportTo() { return reportTo; }

    public static Builder builder() { return new Builder(); }

    public static KtSftConfig sftLoraDemo() {
        return builder()
                .stage(Stage.SFT)
                .peftKind(PeftKind.LORA)
                .loraR(8)
                .maxSteps(20)
                .visualBoard(true)
                .tensorboard(true)
                .build();
    }

    public static final class Builder {
        private Stage stage = Stage.SFT;
        private PeftKind peftKind = PeftKind.LORA;
        private int loraR = 16;
        private double loraAlpha = 32.0;
        private double loraDropout = 0.05;
        private int maxSteps = 100;
        private int loggingSteps = 1;
        private double learningRate = 2e-4;
        private int batchSize = 1;
        private int gradAccum = 1;
        private boolean offloadFrozen = true;
        private boolean visualBoard = true;
        private boolean tensorboard = true;
        private boolean wandb = false;
        private Path outputDir = Path.of("saves/kt-sft");
        private Path datasetPath = null;
        private String reportTo = "tensorboard";

        public Builder stage(Stage v) { this.stage = v; return this; }
        public Builder peftKind(PeftKind v) { this.peftKind = v; return this; }
        public Builder loraR(int v) { this.loraR = v; return this; }
        public Builder loraAlpha(double v) { this.loraAlpha = v; return this; }
        public Builder loraDropout(double v) { this.loraDropout = v; return this; }
        public Builder maxSteps(int v) { this.maxSteps = v; return this; }
        public Builder loggingSteps(int v) { this.loggingSteps = v; return this; }
        public Builder learningRate(double v) { this.learningRate = v; return this; }
        public Builder batchSize(int v) { this.batchSize = v; return this; }
        public Builder gradAccum(int v) { this.gradAccum = v; return this; }
        public Builder offloadFrozen(boolean v) { this.offloadFrozen = v; return this; }
        public Builder visualBoard(boolean v) { this.visualBoard = v; return this; }
        public Builder tensorboard(boolean v) { this.tensorboard = v; return this; }
        public Builder wandb(boolean v) { this.wandb = v; return this; }
        public Builder outputDir(Path v) { this.outputDir = v; return this; }
        public Builder datasetPath(Path v) { this.datasetPath = v; return this; }
        public Builder reportTo(String v) { this.reportTo = v; return this; }

        public KtSftConfig build() { return new KtSftConfig(this); }
    }
}
