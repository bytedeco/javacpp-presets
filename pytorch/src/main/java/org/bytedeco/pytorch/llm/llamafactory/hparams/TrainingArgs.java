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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.LinkedHashMap;
import java.util.Map;

/** Optimizer loop / logging / distributed training args. */
public final class TrainingArgs {
    private final String outputDir;
    private final boolean overwriteOutputDir;
    private final double numTrainEpochs;
    private final int maxSteps;
    private final int perDeviceTrainBatchSize;
    private final int perDeviceEvalBatchSize;
    private final int gradientAccumulationSteps;
    private final double learningRate;
    private final double weightDecay;
    private final String lrSchedulerType;
    private final double warmupRatio;
    private final int warmupSteps;
    private final int loggingSteps;
    private final int saveSteps;
    private final int evalSteps;
    private final int saveTotalLimit;
    private final long seed;
    private final long dataSeed;
    private final boolean fp16;
    private final boolean bf16;
    private final double maxGradNorm;
    private final boolean gradientCheckpointing;
    private final int dataloaderNumWorkers;
    private final boolean dataloaderPinMemory;
    private final boolean removeUnusedColumns;
    private final String reportTo;
    private final String runName;
    private final String projectName;
    private final String deepspeed;
    private final boolean fsdp;
    private final int ddpTimeout;
    private final String resumeFromCheckpoint;
    private final boolean plotLoss;
    private final boolean includeNumInputTokensSeen;
    private final int maxNewTokens;
    private final boolean predictWithGenerate;
    private final boolean doTrain;
    private final boolean doEval;
    private final boolean doPredict;
    private final int boardPort;
    private final boolean boardEnabled;
    private final boolean pushToHub;

    private TrainingArgs(Builder b) {
        this.outputDir = b.outputDir == null ? "saves/factory" : b.outputDir;
        this.overwriteOutputDir = b.overwriteOutputDir;
        this.numTrainEpochs = b.numTrainEpochs;
        this.maxSteps = b.maxSteps;
        this.perDeviceTrainBatchSize = b.perDeviceTrainBatchSize;
        this.perDeviceEvalBatchSize = b.perDeviceEvalBatchSize;
        this.gradientAccumulationSteps = b.gradientAccumulationSteps;
        this.learningRate = b.learningRate;
        this.weightDecay = b.weightDecay;
        this.lrSchedulerType = b.lrSchedulerType == null ? "cosine" : b.lrSchedulerType;
        this.warmupRatio = b.warmupRatio;
        this.warmupSteps = b.warmupSteps;
        this.loggingSteps = b.loggingSteps;
        this.saveSteps = b.saveSteps;
        this.evalSteps = b.evalSteps;
        this.saveTotalLimit = b.saveTotalLimit;
        this.seed = b.seed;
        this.dataSeed = b.dataSeed;
        this.fp16 = b.fp16;
        this.bf16 = b.bf16;
        this.maxGradNorm = b.maxGradNorm;
        this.gradientCheckpointing = b.gradientCheckpointing;
        this.dataloaderNumWorkers = b.dataloaderNumWorkers;
        this.dataloaderPinMemory = b.dataloaderPinMemory;
        this.removeUnusedColumns = b.removeUnusedColumns;
        this.reportTo = b.reportTo == null ? "none" : b.reportTo;
        this.runName = b.runName;
        this.projectName = b.projectName == null ? "llamafactory" : b.projectName;
        this.deepspeed = b.deepspeed;
        this.fsdp = b.fsdp;
        this.ddpTimeout = b.ddpTimeout;
        this.resumeFromCheckpoint = b.resumeFromCheckpoint;
        this.plotLoss = b.plotLoss;
        this.includeNumInputTokensSeen = b.includeNumInputTokensSeen;
        this.maxNewTokens = b.maxNewTokens;
        this.predictWithGenerate = b.predictWithGenerate;
        this.doTrain = b.doTrain;
        this.doEval = b.doEval;
        this.doPredict = b.doPredict;
        this.boardPort = b.boardPort;
        this.boardEnabled = b.boardEnabled;
        this.pushToHub = b.pushToHub;
    }

    public String outputDir() { return outputDir; }
    public boolean overwriteOutputDir() { return overwriteOutputDir; }
    public double numTrainEpochs() { return numTrainEpochs; }
    public int maxSteps() { return maxSteps; }
    public int perDeviceTrainBatchSize() { return perDeviceTrainBatchSize; }
    public int perDeviceEvalBatchSize() { return perDeviceEvalBatchSize; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public double learningRate() { return learningRate; }
    public double weightDecay() { return weightDecay; }
    public String lrSchedulerType() { return lrSchedulerType; }
    public double warmupRatio() { return warmupRatio; }
    public int warmupSteps() { return warmupSteps; }
    public int loggingSteps() { return loggingSteps; }
    public int saveSteps() { return saveSteps; }
    public int evalSteps() { return evalSteps; }
    public int saveTotalLimit() { return saveTotalLimit; }
    public long seed() { return seed; }
    public long dataSeed() { return dataSeed; }
    public boolean fp16() { return fp16; }
    public boolean bf16() { return bf16; }
    public double maxGradNorm() { return maxGradNorm; }
    public boolean gradientCheckpointing() { return gradientCheckpointing; }
    public int dataloaderNumWorkers() { return dataloaderNumWorkers; }
    public boolean dataloaderPinMemory() { return dataloaderPinMemory; }
    public boolean removeUnusedColumns() { return removeUnusedColumns; }
    public String reportTo() { return reportTo; }
    public String runName() { return runName; }
    public String projectName() { return projectName; }
    public String deepspeed() { return deepspeed; }
    public boolean fsdp() { return fsdp; }
    public int ddpTimeout() { return ddpTimeout; }
    public String resumeFromCheckpoint() { return resumeFromCheckpoint; }
    public boolean plotLoss() { return plotLoss; }
    public boolean includeNumInputTokensSeen() { return includeNumInputTokensSeen; }
    public int maxNewTokens() { return maxNewTokens; }
    public boolean predictWithGenerate() { return predictWithGenerate; }
    public boolean doTrain() { return doTrain; }
    public boolean doEval() { return doEval; }
    public boolean doPredict() { return doPredict; }
    public int boardPort() { return boardPort; }
    public boolean boardEnabled() { return boardEnabled; }
    public boolean pushToHub() { return pushToHub; }

    /** Resolve max steps from epochs when {@code maxSteps < 0}. */
    public int effectiveMaxSteps(int datasetSize) {
        if (maxSteps > 0) return maxSteps;
        int bs = Math.max(1, perDeviceTrainBatchSize) * Math.max(1, gradientAccumulationSteps);
        int stepsPerEpoch = Math.max(1, (Math.max(1, datasetSize) + bs - 1) / bs);
        return Math.max(1, (int) Math.ceil(numTrainEpochs * stepsPerEpoch));
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "output_dir", outputDir);
        HparamsMaps.put(m, "overwrite_output_dir", overwriteOutputDir);
        HparamsMaps.put(m, "num_train_epochs", numTrainEpochs);
        HparamsMaps.put(m, "max_steps", maxSteps);
        HparamsMaps.put(m, "per_device_train_batch_size", perDeviceTrainBatchSize);
        HparamsMaps.put(m, "per_device_eval_batch_size", perDeviceEvalBatchSize);
        HparamsMaps.put(m, "gradient_accumulation_steps", gradientAccumulationSteps);
        HparamsMaps.put(m, "learning_rate", learningRate);
        HparamsMaps.put(m, "weight_decay", weightDecay);
        HparamsMaps.put(m, "lr_scheduler_type", lrSchedulerType);
        HparamsMaps.put(m, "warmup_ratio", warmupRatio);
        HparamsMaps.put(m, "warmup_steps", warmupSteps);
        HparamsMaps.put(m, "logging_steps", loggingSteps);
        HparamsMaps.put(m, "save_steps", saveSteps);
        HparamsMaps.put(m, "eval_steps", evalSteps);
        HparamsMaps.put(m, "save_total_limit", saveTotalLimit);
        HparamsMaps.put(m, "seed", seed);
        HparamsMaps.put(m, "data_seed", dataSeed);
        HparamsMaps.put(m, "fp16", fp16);
        HparamsMaps.put(m, "bf16", bf16);
        HparamsMaps.put(m, "max_grad_norm", maxGradNorm);
        HparamsMaps.put(m, "gradient_checkpointing", gradientCheckpointing);
        HparamsMaps.put(m, "dataloader_num_workers", dataloaderNumWorkers);
        HparamsMaps.put(m, "dataloader_pin_memory", dataloaderPinMemory);
        HparamsMaps.put(m, "remove_unused_columns", removeUnusedColumns);
        HparamsMaps.put(m, "report_to", reportTo);
        HparamsMaps.put(m, "run_name", runName);
        HparamsMaps.put(m, "project_name", projectName);
        HparamsMaps.put(m, "deepspeed", deepspeed);
        HparamsMaps.put(m, "fsdp", fsdp);
        HparamsMaps.put(m, "ddp_timeout", ddpTimeout);
        HparamsMaps.put(m, "resume_from_checkpoint", resumeFromCheckpoint);
        HparamsMaps.put(m, "plot_loss", plotLoss);
        HparamsMaps.put(m, "include_num_input_tokens_seen", includeNumInputTokensSeen);
        HparamsMaps.put(m, "max_new_tokens", maxNewTokens);
        HparamsMaps.put(m, "predict_with_generate", predictWithGenerate);
        HparamsMaps.put(m, "do_train", doTrain);
        HparamsMaps.put(m, "do_eval", doEval);
        HparamsMaps.put(m, "do_predict", doPredict);
        HparamsMaps.put(m, "board_port", boardPort);
        HparamsMaps.put(m, "board_enabled", boardEnabled);
        HparamsMaps.put(m, "push_to_hub", pushToHub);
        return m;
    }

    public static TrainingArgs defaults() { return builder().build(); }

    public static TrainingArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        b.outputDir(HparamsMaps.str(m, b.outputDir, "output_dir"));
        b.overwriteOutputDir(HparamsMaps.bool(m, b.overwriteOutputDir, "overwrite_output_dir"));
        b.numTrainEpochs(HparamsMaps.dbl(m, b.numTrainEpochs, "num_train_epochs", "epochs"));
        b.maxSteps(HparamsMaps.integer(m, b.maxSteps, "max_steps"));
        b.perDeviceTrainBatchSize(HparamsMaps.integer(m, b.perDeviceTrainBatchSize, "per_device_train_batch_size", "batch_size"));
        b.perDeviceEvalBatchSize(HparamsMaps.integer(m, b.perDeviceEvalBatchSize, "per_device_eval_batch_size", "eval_batch_size"));
        b.gradientAccumulationSteps(HparamsMaps.integer(m, b.gradientAccumulationSteps, "gradient_accumulation_steps", "grad_accum"));
        b.learningRate(HparamsMaps.dbl(m, b.learningRate, "learning_rate", "lr"));
        b.weightDecay(HparamsMaps.dbl(m, b.weightDecay, "weight_decay"));
        b.lrSchedulerType(HparamsMaps.str(m, b.lrSchedulerType, "lr_scheduler_type", "lr_scheduler"));
        b.warmupRatio(HparamsMaps.dbl(m, b.warmupRatio, "warmup_ratio"));
        b.warmupSteps(HparamsMaps.integer(m, b.warmupSteps, "warmup_steps"));
        b.loggingSteps(HparamsMaps.integer(m, b.loggingSteps, "logging_steps"));
        b.saveSteps(HparamsMaps.integer(m, b.saveSteps, "save_steps"));
        b.evalSteps(HparamsMaps.integer(m, b.evalSteps, "eval_steps"));
        b.saveTotalLimit(HparamsMaps.integer(m, b.saveTotalLimit, "save_total_limit"));
        b.seed(HparamsMaps.lng(m, b.seed, "seed"));
        b.dataSeed(HparamsMaps.lng(m, b.dataSeed, "data_seed"));
        b.fp16(HparamsMaps.bool(m, b.fp16, "fp16"));
        b.bf16(HparamsMaps.bool(m, b.bf16, "bf16"));
        b.maxGradNorm(HparamsMaps.dbl(m, b.maxGradNorm, "max_grad_norm"));
        b.gradientCheckpointing(HparamsMaps.bool(m, b.gradientCheckpointing, "gradient_checkpointing"));
        b.dataloaderNumWorkers(HparamsMaps.integer(m, b.dataloaderNumWorkers, "dataloader_num_workers"));
        b.dataloaderPinMemory(HparamsMaps.bool(m, b.dataloaderPinMemory, "dataloader_pin_memory"));
        b.removeUnusedColumns(HparamsMaps.bool(m, b.removeUnusedColumns, "remove_unused_columns"));
        b.reportTo(HparamsMaps.str(m, b.reportTo, "report_to"));
        b.runName(HparamsMaps.strOrNull(m, "run_name"));
        b.projectName(HparamsMaps.str(m, b.projectName, "project_name", "project"));
        b.deepspeed(HparamsMaps.strOrNull(m, "deepspeed"));
        b.fsdp(HparamsMaps.bool(m, b.fsdp, "fsdp"));
        b.ddpTimeout(HparamsMaps.integer(m, b.ddpTimeout, "ddp_timeout"));
        b.resumeFromCheckpoint(HparamsMaps.strOrNull(m, "resume_from_checkpoint", "resume_from"));
        b.plotLoss(HparamsMaps.bool(m, b.plotLoss, "plot_loss"));
        b.includeNumInputTokensSeen(HparamsMaps.bool(m, b.includeNumInputTokensSeen, "include_num_input_tokens_seen"));
        b.maxNewTokens(HparamsMaps.integer(m, b.maxNewTokens, "max_new_tokens"));
        b.predictWithGenerate(HparamsMaps.bool(m, b.predictWithGenerate, "predict_with_generate"));
        b.doTrain(HparamsMaps.bool(m, b.doTrain, "do_train"));
        b.doEval(HparamsMaps.bool(m, b.doEval, "do_eval"));
        b.doPredict(HparamsMaps.bool(m, b.doPredict, "do_predict"));
        b.boardPort(HparamsMaps.integer(m, b.boardPort, "board_port", "llamaboard_port"));
        b.boardEnabled(HparamsMaps.bool(m, b.boardEnabled, "board_enabled", "llamaboard"));
        b.pushToHub(HparamsMaps.bool(m, b.pushToHub, "push_to_hub"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String outputDir = "saves/factory";
        private boolean overwriteOutputDir;
        private double numTrainEpochs = 3.0;
        private int maxSteps = -1;
        private int perDeviceTrainBatchSize = 1;
        private int perDeviceEvalBatchSize = 1;
        private int gradientAccumulationSteps = 1;
        private double learningRate = 5e-5;
        private double weightDecay;
        private String lrSchedulerType = "cosine";
        private double warmupRatio;
        private int warmupSteps;
        private int loggingSteps = 5;
        private int saveSteps = 100;
        private int evalSteps = 100;
        private int saveTotalLimit = 3;
        private long seed = 42L;
        private long dataSeed = 42L;
        private boolean fp16;
        private boolean bf16;
        private double maxGradNorm = 1.0;
        private boolean gradientCheckpointing;
        private int dataloaderNumWorkers;
        private boolean dataloaderPinMemory;
        private boolean removeUnusedColumns = true;
        private String reportTo = "none";
        private String runName;
        private String projectName = "llamafactory";
        private String deepspeed;
        private boolean fsdp;
        private int ddpTimeout = 1800;
        private String resumeFromCheckpoint;
        private boolean plotLoss = true;
        private boolean includeNumInputTokensSeen;
        private int maxNewTokens;
        private boolean predictWithGenerate;
        private boolean doTrain = true;
        private boolean doEval;
        private boolean doPredict;
        private int boardPort = 7860;
        private boolean boardEnabled;
        private boolean pushToHub;

        public Builder outputDir(String v) { this.outputDir = v; return this; }
        public Builder overwriteOutputDir(boolean v) { this.overwriteOutputDir = v; return this; }
        public Builder numTrainEpochs(double v) { this.numTrainEpochs = v; return this; }
        public Builder maxSteps(int v) { this.maxSteps = v; return this; }
        public Builder perDeviceTrainBatchSize(int v) { this.perDeviceTrainBatchSize = v; return this; }
        public Builder perDeviceEvalBatchSize(int v) { this.perDeviceEvalBatchSize = v; return this; }
        public Builder gradientAccumulationSteps(int v) { this.gradientAccumulationSteps = v; return this; }
        public Builder learningRate(double v) { this.learningRate = v; return this; }
        public Builder weightDecay(double v) { this.weightDecay = v; return this; }
        public Builder lrSchedulerType(String v) { this.lrSchedulerType = v; return this; }
        public Builder warmupRatio(double v) { this.warmupRatio = v; return this; }
        public Builder warmupSteps(int v) { this.warmupSteps = v; return this; }
        public Builder loggingSteps(int v) { this.loggingSteps = v; return this; }
        public Builder saveSteps(int v) { this.saveSteps = v; return this; }
        public Builder evalSteps(int v) { this.evalSteps = v; return this; }
        public Builder saveTotalLimit(int v) { this.saveTotalLimit = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }
        public Builder dataSeed(long v) { this.dataSeed = v; return this; }
        public Builder fp16(boolean v) { this.fp16 = v; return this; }
        public Builder bf16(boolean v) { this.bf16 = v; return this; }
        public Builder maxGradNorm(double v) { this.maxGradNorm = v; return this; }
        public Builder gradientCheckpointing(boolean v) { this.gradientCheckpointing = v; return this; }
        public Builder dataloaderNumWorkers(int v) { this.dataloaderNumWorkers = v; return this; }
        public Builder dataloaderPinMemory(boolean v) { this.dataloaderPinMemory = v; return this; }
        public Builder removeUnusedColumns(boolean v) { this.removeUnusedColumns = v; return this; }
        public Builder reportTo(String v) { this.reportTo = v; return this; }
        public Builder runName(String v) { this.runName = v; return this; }
        public Builder projectName(String v) { this.projectName = v; return this; }
        public Builder deepspeed(String v) { this.deepspeed = v; return this; }
        public Builder fsdp(boolean v) { this.fsdp = v; return this; }
        public Builder ddpTimeout(int v) { this.ddpTimeout = v; return this; }
        public Builder resumeFromCheckpoint(String v) { this.resumeFromCheckpoint = v; return this; }
        public Builder plotLoss(boolean v) { this.plotLoss = v; return this; }
        public Builder includeNumInputTokensSeen(boolean v) { this.includeNumInputTokensSeen = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder predictWithGenerate(boolean v) { this.predictWithGenerate = v; return this; }
        public Builder doTrain(boolean v) { this.doTrain = v; return this; }
        public Builder doEval(boolean v) { this.doEval = v; return this; }
        public Builder doPredict(boolean v) { this.doPredict = v; return this; }
        public Builder boardPort(int v) { this.boardPort = v; return this; }
        public Builder boardEnabled(boolean v) { this.boardEnabled = v; return this; }
        public Builder pushToHub(boolean v) { this.pushToHub = v; return this; }
        public TrainingArgs build() { return new TrainingArgs(this); }
    }
}
