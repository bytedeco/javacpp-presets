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
package org.bytedeco.pytorch.llm.unsloth.studio.model;

import org.bytedeco.pytorch.llm.unsloth.studio.util.Validate;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Request to start a Studio training run. Field names and limits follow
 * upstream {@code TrainingStartRequest}.
 */
public final class TrainingStartRequest {

    private final String modelName;
    private final String projectName;
    private final TrainingType trainingType;
    private final String hfToken;
    private final boolean loadIn4bit;
    private final boolean loadIn8bit;
    private final int maxSeqLength;
    private final Integer visionImageSize;
    private final boolean trustRemoteCode;
    private final int loraR;
    private final int loraAlpha;
    private final double loraDropout;
    private final List<String> targetModules;
    private final double learningRate;
    private final int batchSize;
    private final int gradientAccumulationSteps;
    private final int maxSteps;
    private final double numTrainEpochs;
    private final String dataset;
    private final String datasetPath;
    private final long datasetSkip;
    private final long datasetTake;
    private final int seed;
    private final boolean gradientCheckpointing;
    private final String optim;
    private final double weightDecay;
    private final int warmupSteps;
    private final String lrSchedulerType;
    private final String outputDir;
    private final boolean packing;
    private final String rlAlgorithm;
    private final Map<String, Object> extra;

    private TrainingStartRequest(Builder b) {
        this.modelName = Objects.requireNonNull(b.modelName, "model_name");
        Validate.requireNonBlank("model_name", b.modelName);
        this.projectName = Validate.projectName(b.projectName);
        this.trainingType = Objects.requireNonNull(b.trainingType, "training_type");
        this.hfToken = b.hfToken;
        this.loadIn4bit = b.loadIn4bit;
        this.loadIn8bit = b.loadIn8bit;
        this.maxSeqLength = Validate.maxSeqLength(b.maxSeqLength);
        this.visionImageSize = Validate.visionImageSize(b.visionImageSize);
        this.trustRemoteCode = b.trustRemoteCode;
        this.loraR = Validate.loraR(b.loraR);
        this.loraAlpha = Validate.loraAlpha(b.loraAlpha);
        this.loraDropout = b.loraDropout;
        this.targetModules = List.copyOf(b.targetModules);
        this.learningRate = Validate.learningRate(b.learningRate);
        this.batchSize = Validate.batchSize(b.batchSize);
        this.gradientAccumulationSteps = Validate.gradAccum(b.gradientAccumulationSteps);
        this.maxSteps = b.maxSteps > 0 ? Validate.maxSteps(b.maxSteps) : b.maxSteps;
        this.numTrainEpochs = b.numTrainEpochs > 0 ? Validate.epochs(b.numTrainEpochs) : b.numTrainEpochs;
        if (this.maxSteps <= 0 && this.numTrainEpochs <= 0) {
            throw new org.bytedeco.pytorch.llm.unsloth.studio.util.StudioValidationException(
                    "either max_steps or num_train_epochs must be positive");
        }
        this.dataset = b.dataset;
        this.datasetPath = b.datasetPath;
        this.datasetSkip = Validate.datasetSliceIndex(b.datasetSkip);
        this.datasetTake = b.datasetTake < 0 ? -1 : Validate.datasetSliceIndex(b.datasetTake);
        this.seed = b.seed;
        this.gradientCheckpointing = b.gradientCheckpointing;
        this.optim = b.optim != null ? b.optim : "adamw_torch";
        this.weightDecay = b.weightDecay;
        this.warmupSteps = Math.max(0, b.warmupSteps);
        this.lrSchedulerType = b.lrSchedulerType != null ? b.lrSchedulerType : "linear";
        this.outputDir = b.outputDir;
        this.packing = b.packing;
        this.rlAlgorithm = b.rlAlgorithm;
        this.extra = Map.copyOf(b.extra);
    }

    public static Builder builder() {
        return new Builder();
    }

    public String modelName() { return modelName; }
    public Optional<String> projectName() { return Optional.ofNullable(projectName); }
    public TrainingType trainingType() { return trainingType; }
    public Optional<String> hfToken() { return Optional.ofNullable(hfToken); }
    public boolean loadIn4bit() { return loadIn4bit; }
    public boolean loadIn8bit() { return loadIn8bit; }
    public int maxSeqLength() { return maxSeqLength; }
    public Optional<Integer> visionImageSize() { return Optional.ofNullable(visionImageSize); }
    public boolean trustRemoteCode() { return trustRemoteCode; }
    public int loraR() { return loraR; }
    public int loraAlpha() { return loraAlpha; }
    public double loraDropout() { return loraDropout; }
    public List<String> targetModules() { return targetModules; }
    public double learningRate() { return learningRate; }
    public int batchSize() { return batchSize; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public int maxSteps() { return maxSteps; }
    public double numTrainEpochs() { return numTrainEpochs; }
    public Optional<String> dataset() { return Optional.ofNullable(dataset); }
    public Optional<String> datasetPath() { return Optional.ofNullable(datasetPath); }
    public long datasetSkip() { return datasetSkip; }
    public long datasetTake() { return datasetTake; }
    public int seed() { return seed; }
    public boolean gradientCheckpointing() { return gradientCheckpointing; }
    public String optim() { return optim; }
    public double weightDecay() { return weightDecay; }
    public int warmupSteps() { return warmupSteps; }
    public String lrSchedulerType() { return lrSchedulerType; }
    public Optional<String> outputDir() { return Optional.ofNullable(outputDir); }
    public boolean packing() { return packing; }
    public Optional<String> rlAlgorithm() { return Optional.ofNullable(rlAlgorithm); }
    public Map<String, Object> extra() { return extra; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_name", modelName);
        if (projectName != null) m.put("project_name", projectName);
        m.put("training_type", trainingType.label());
        m.put("load_in_4bit", loadIn4bit);
        m.put("load_in_8bit", loadIn8bit);
        m.put("max_seq_length", maxSeqLength);
        if (visionImageSize != null) m.put("vision_image_size", visionImageSize);
        m.put("trust_remote_code", trustRemoteCode);
        m.put("lora_r", loraR);
        m.put("lora_alpha", loraAlpha);
        m.put("lora_dropout", loraDropout);
        m.put("target_modules", targetModules);
        m.put("learning_rate", learningRate);
        m.put("batch_size", batchSize);
        m.put("gradient_accumulation_steps", gradientAccumulationSteps);
        m.put("max_steps", maxSteps);
        m.put("num_train_epochs", numTrainEpochs);
        if (dataset != null) m.put("dataset", dataset);
        if (datasetPath != null) m.put("dataset_path", datasetPath);
        m.put("dataset_skip", datasetSkip);
        m.put("dataset_take", datasetTake);
        m.put("seed", seed);
        m.put("gradient_checkpointing", gradientCheckpointing);
        m.put("optim", optim);
        m.put("weight_decay", weightDecay);
        m.put("warmup_steps", warmupSteps);
        m.put("lr_scheduler_type", lrSchedulerType);
        if (outputDir != null) m.put("output_dir", outputDir);
        m.put("packing", packing);
        if (rlAlgorithm != null) m.put("rl_algorithm", rlAlgorithm);
        if (!extra.isEmpty()) m.put("extra", extra);
        return m;
    }

    @SuppressWarnings("unchecked")
    public static TrainingStartRequest fromMap(Map<String, Object> m) {
        Builder b = builder();
        if (m.containsKey("model_name")) b.modelName(String.valueOf(m.get("model_name")));
        if (m.containsKey("project_name") && m.get("project_name") != null) {
            b.projectName(String.valueOf(m.get("project_name")));
        }
        if (m.containsKey("training_type")) {
            b.trainingType(TrainingType.fromLabel(String.valueOf(m.get("training_type"))));
        }
        if (m.containsKey("load_in_4bit")) b.loadIn4bit(asBool(m.get("load_in_4bit"), true));
        if (m.containsKey("load_in_8bit")) b.loadIn8bit(asBool(m.get("load_in_8bit"), false));
        if (m.containsKey("max_seq_length")) b.maxSeqLength(asInt(m.get("max_seq_length"), 2048));
        if (m.containsKey("vision_image_size") && m.get("vision_image_size") != null) {
            b.visionImageSize(asInt(m.get("vision_image_size"), 512));
        }
        if (m.containsKey("trust_remote_code")) b.trustRemoteCode(asBool(m.get("trust_remote_code"), false));
        if (m.containsKey("lora_r")) b.loraR(asInt(m.get("lora_r"), 16));
        if (m.containsKey("lora_alpha")) b.loraAlpha(asInt(m.get("lora_alpha"), 16));
        if (m.containsKey("lora_dropout")) b.loraDropout(asDouble(m.get("lora_dropout"), 0.0));
        if (m.containsKey("learning_rate")) b.learningRate(Validate.parseLearningRate(m.get("learning_rate")));
        if (m.containsKey("batch_size")) b.batchSize(asInt(m.get("batch_size"), 2));
        if (m.containsKey("gradient_accumulation_steps")) {
            b.gradientAccumulationSteps(asInt(m.get("gradient_accumulation_steps"), 4));
        }
        if (m.containsKey("max_steps")) b.maxSteps(asInt(m.get("max_steps"), 0));
        if (m.containsKey("num_train_epochs")) b.numTrainEpochs(asDouble(m.get("num_train_epochs"), 0));
        if (m.containsKey("dataset") && m.get("dataset") != null) b.dataset(String.valueOf(m.get("dataset")));
        if (m.containsKey("dataset_path") && m.get("dataset_path") != null) {
            b.datasetPath(String.valueOf(m.get("dataset_path")));
        }
        if (m.containsKey("seed")) b.seed(asInt(m.get("seed"), 42));
        if (m.containsKey("gradient_checkpointing")) {
            b.gradientCheckpointing(asBool(m.get("gradient_checkpointing"), true));
        }
        if (m.containsKey("target_modules") && m.get("target_modules") instanceof List<?> list) {
            List<String> tm = new ArrayList<>();
            for (Object o : list) tm.add(String.valueOf(o));
            b.targetModules(tm);
        }
        if (m.containsKey("output_dir") && m.get("output_dir") != null) {
            b.outputDir(String.valueOf(m.get("output_dir")));
        }
        if (m.containsKey("rl_algorithm") && m.get("rl_algorithm") != null) {
            b.rlAlgorithm(String.valueOf(m.get("rl_algorithm")));
        }
        return b.build();
    }

    private static int asInt(Object v, int d) {
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(String.valueOf(v)); } catch (Exception e) { return d; }
    }

    private static double asDouble(Object v, double d) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(String.valueOf(v)); } catch (Exception e) { return d; }
    }

    private static boolean asBool(Object v, boolean d) {
        if (v instanceof Boolean) return (Boolean) v;
        if (v == null) return d;
        return Boolean.parseBoolean(String.valueOf(v));
    }

    public static final class Builder {
        private String modelName;
        private String projectName;
        private TrainingType trainingType = TrainingType.LORA_QLORA;
        private String hfToken;
        private boolean loadIn4bit = true;
        private boolean loadIn8bit = false;
        private int maxSeqLength = 2048;
        private Integer visionImageSize;
        private boolean trustRemoteCode = false;
        private int loraR = 16;
        private int loraAlpha = 16;
        private double loraDropout = 0.0;
        private List<String> targetModules = List.of("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj");
        private double learningRate = 2e-4;
        private int batchSize = 2;
        private int gradientAccumulationSteps = 4;
        private int maxSteps = 60;
        private double numTrainEpochs = 0;
        private String dataset = "alpaca_demo";
        private String datasetPath;
        private long datasetSkip = 0;
        private long datasetTake = -1;
        private int seed = 42;
        private boolean gradientCheckpointing = true;
        private String optim = "adamw_torch";
        private double weightDecay = 0.01;
        private int warmupSteps = 5;
        private String lrSchedulerType = "linear";
        private String outputDir;
        private boolean packing = false;
        private String rlAlgorithm;
        private Map<String, Object> extra = Map.of();

        public Builder modelName(String v) { this.modelName = v; return this; }
        public Builder projectName(String v) { this.projectName = v; return this; }
        public Builder trainingType(TrainingType v) { this.trainingType = v; return this; }
        public Builder trainingType(String v) { this.trainingType = TrainingType.fromLabel(v); return this; }
        public Builder hfToken(String v) { this.hfToken = v; return this; }
        public Builder loadIn4bit(boolean v) { this.loadIn4bit = v; return this; }
        public Builder loadIn8bit(boolean v) { this.loadIn8bit = v; return this; }
        public Builder maxSeqLength(int v) { this.maxSeqLength = v; return this; }
        public Builder visionImageSize(Integer v) { this.visionImageSize = v; return this; }
        public Builder trustRemoteCode(boolean v) { this.trustRemoteCode = v; return this; }
        public Builder loraR(int v) { this.loraR = v; return this; }
        public Builder loraAlpha(int v) { this.loraAlpha = v; return this; }
        public Builder loraDropout(double v) { this.loraDropout = v; return this; }
        public Builder targetModules(List<String> v) { this.targetModules = v != null ? v : List.of(); return this; }
        public Builder learningRate(double v) { this.learningRate = v; return this; }
        public Builder batchSize(int v) { this.batchSize = v; return this; }
        public Builder gradientAccumulationSteps(int v) { this.gradientAccumulationSteps = v; return this; }
        public Builder maxSteps(int v) { this.maxSteps = v; return this; }
        public Builder numTrainEpochs(double v) { this.numTrainEpochs = v; return this; }
        public Builder dataset(String v) { this.dataset = v; return this; }
        public Builder datasetPath(String v) { this.datasetPath = v; return this; }
        public Builder datasetSkip(long v) { this.datasetSkip = v; return this; }
        public Builder datasetTake(long v) { this.datasetTake = v; return this; }
        public Builder seed(int v) { this.seed = v; return this; }
        public Builder gradientCheckpointing(boolean v) { this.gradientCheckpointing = v; return this; }
        public Builder optim(String v) { this.optim = v; return this; }
        public Builder weightDecay(double v) { this.weightDecay = v; return this; }
        public Builder warmupSteps(int v) { this.warmupSteps = v; return this; }
        public Builder lrSchedulerType(String v) { this.lrSchedulerType = v; return this; }
        public Builder outputDir(String v) { this.outputDir = v; return this; }
        public Builder packing(boolean v) { this.packing = v; return this; }
        public Builder rlAlgorithm(String v) { this.rlAlgorithm = v; return this; }
        public Builder extra(Map<String, Object> v) { this.extra = v != null ? v : Map.of(); return this; }

        public TrainingStartRequest build() {
            return new TrainingStartRequest(this);
        }
    }
}
