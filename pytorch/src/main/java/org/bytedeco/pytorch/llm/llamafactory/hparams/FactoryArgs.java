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
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Aggregate LLaMA-Factory arguments: model + data + finetuning + training + …
 *
 * <p>Accepts both nested maps ({@code model}/{@code data}/…) and flat
 * snake_case HF / LLaMA-Factory keys via {@link #parse(Map)}.
 */
public final class FactoryArgs {
    private final ModelArgs model;
    private final DataArgs data;
    private final FinetuningArgs finetuning;
    private final TrainingArgs training;
    private final GeneratingArgs generating;
    private final EvaluationArgs evaluation;
    private final InferArgs infer;
    private final ExportArgs export;

    private FactoryArgs(Builder b) {
        this.model = b.model == null ? ModelArgs.defaults() : b.model;
        this.data = b.data == null ? DataArgs.defaults() : b.data;
        this.finetuning = b.finetuning == null ? FinetuningArgs.defaults() : b.finetuning;
        this.training = b.training == null ? TrainingArgs.defaults() : b.training;
        this.generating = b.generating == null ? GeneratingArgs.defaults() : b.generating;
        this.evaluation = b.evaluation == null ? EvaluationArgs.defaults() : b.evaluation;
        this.infer = b.infer == null ? InferArgs.defaults() : b.infer;
        this.export = b.export == null ? ExportArgs.defaults() : b.export;
    }

    public ModelArgs model() { return model; }
    public DataArgs data() { return data; }
    public FinetuningArgs finetuning() { return finetuning; }
    public TrainingArgs training() { return training; }
    public GeneratingArgs generating() { return generating; }
    public EvaluationArgs evaluation() { return evaluation; }
    public InferArgs infer() { return infer; }
    public ExportArgs export() { return export; }

    /** Cross-field validation (throws {@link IllegalArgumentException}). */
    public void validate() {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(finetuning, "finetuning");
        Objects.requireNonNull(training, "training");

        if (model.modelNameOrPath() == null || model.modelNameOrPath().isBlank()) {
            throw new IllegalArgumentException("model_name_or_path must be non-blank");
        }
        if (data.cutoffLen() <= 0) {
            throw new IllegalArgumentException("cutoff_len must be > 0, got " + data.cutoffLen());
        }
        if (training.learningRate() <= 0.0) {
            throw new IllegalArgumentException("learning_rate must be > 0, got " + training.learningRate());
        }
        if (training.perDeviceTrainBatchSize() <= 0) {
            throw new IllegalArgumentException("per_device_train_batch_size must be > 0");
        }
        if (training.gradientAccumulationSteps() <= 0) {
            throw new IllegalArgumentException("gradient_accumulation_steps must be > 0");
        }

        FinetuningType ft = finetuning.finetuningType();
        if (ft == FinetuningType.LORA || ft == FinetuningType.QLORA) {
            if (finetuning.loraRank() <= 0) {
                throw new IllegalArgumentException("lora_rank must be > 0 for " + ft.wireName());
            }
        }
        if (ft == FinetuningType.QLORA && !model.quantizationMethod().enabled()) {
            throw new IllegalArgumentException(
                    "finetuning_type=qlora requires quantization_method (bnb/gptq/awq/…)");
        }
        if (finetuning.useDora() && finetuning.useOft()) {
            throw new IllegalArgumentException("use_dora and use_oft are mutually exclusive");
        }
        if (finetuning.stage().needsRewardModel()) {
            String rm = finetuning.rewardModel();
            if (rm == null || rm.isBlank()) {
                throw new IllegalArgumentException(
                        "stage=ppo requires reward_model (path or 'dummy' for synthetic tests)");
            }
        }
        if (finetuning.useGalore() && finetuning.galoreRank() <= 0) {
            throw new IllegalArgumentException("galore_rank must be > 0 when use_galore");
        }
        if (finetuning.useApollo() && finetuning.apolloRank() <= 0) {
            throw new IllegalArgumentException("apollo_rank must be > 0 when use_apollo");
        }
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model", model.toMap());
        m.put("data", data.toMap());
        m.put("finetuning", finetuning.toMap());
        m.put("training", training.toMap());
        m.put("generating", generating.toMap());
        m.put("evaluation", evaluation.toMap());
        m.put("infer", infer.toMap());
        m.put("export", export.toMap());
        return m;
    }

    /** Flat snake_case view (LLaMA-Factory CLI style). */
    public Map<String, Object> toFlatMap() {
        Map<String, Object> flat = new LinkedHashMap<>();
        flat.putAll(model.toMap());
        flat.putAll(data.toMap());
        flat.putAll(finetuning.toMap());
        flat.putAll(training.toMap());
        // generating / eval / export kept namespaced to avoid key clashes
        flat.put("generating", generating.toMap());
        flat.put("evaluation", evaluation.toMap());
        flat.put("infer", infer.toMap());
        flat.put("export", export.toMap());
        return flat;
    }

    public static FactoryArgs defaults() {
        return builder().build();
    }

    /**
     * Parse nested or flat maps. Nested keys: {@code model}, {@code data},
     * {@code finetuning}, {@code training}, {@code generating}, {@code evaluation},
     * {@code infer}, {@code export}. Flat keys use LLaMA-Factory snake_case.
     */
    @SuppressWarnings("unchecked")
    public static FactoryArgs parse(Map<String, ?> raw) {
        if (raw == null || raw.isEmpty()) {
            return defaults();
        }
        Map<String, Object> m = new LinkedHashMap<>();
        for (Map.Entry<String, ?> e : raw.entrySet()) {
            if (e.getKey() != null) {
                m.put(e.getKey(), e.getValue());
            }
        }

        ModelArgs model = ModelArgs.fromMap(sectionOrFlat(m, "model"));
        DataArgs data = DataArgs.fromMap(sectionOrFlat(m, "data"));
        FinetuningArgs finetuning = FinetuningArgs.fromMap(sectionOrFlat(m, "finetuning"));
        TrainingArgs training = TrainingArgs.fromMap(sectionOrFlat(m, "training"));

        Map<String, Object> genSec = HparamsMaps.asMap(m.get("generating"));
        GeneratingArgs generating = genSec != null
                ? GeneratingArgs.fromMap(genSec)
                : GeneratingArgs.fromMap(m);

        Map<String, Object> evalSec = HparamsMaps.asMap(m.get("evaluation"));
        EvaluationArgs evaluation = evalSec != null
                ? EvaluationArgs.fromMap(evalSec)
                : EvaluationArgs.fromMap(m);

        Map<String, Object> inferSec = HparamsMaps.asMap(m.get("infer"));
        InferArgs infer = inferSec != null
                ? InferArgs.fromMap(inferSec)
                : InferArgs.fromMap(m);

        Map<String, Object> exportSec = HparamsMaps.asMap(m.get("export"));
        ExportArgs export = exportSec != null
                ? ExportArgs.fromMap(exportSec)
                : ExportArgs.fromMap(m);

        // If top-level model path set and nested model was empty-ish, prefer flat
        if (m.containsKey("model_name_or_path") && HparamsMaps.asMap(m.get("model")) == null) {
            model = ModelArgs.fromMap(m);
        }
        if (m.containsKey("dataset") && HparamsMaps.asMap(m.get("data")) == null) {
            data = DataArgs.fromMap(m);
        }
        if ((m.containsKey("stage") || m.containsKey("finetuning_type") || m.containsKey("lora_rank"))
                && HparamsMaps.asMap(m.get("finetuning")) == null) {
            finetuning = FinetuningArgs.fromMap(m);
        }
        if ((m.containsKey("output_dir") || m.containsKey("learning_rate") || m.containsKey("max_steps"))
                && HparamsMaps.asMap(m.get("training")) == null) {
            training = TrainingArgs.fromMap(m);
        }

        FactoryArgs args = builder()
                .model(model)
                .data(data)
                .finetuning(finetuning)
                .training(training)
                .generating(generating)
                .evaluation(evaluation)
                .infer(infer)
                .export(export)
                .build();
        args.validate();
        return args;
    }

    private static Map<String, ?> sectionOrFlat(Map<String, Object> m, String section) {
        Map<String, Object> nested = HparamsMaps.asMap(m.get(section));
        return nested != null ? nested : m;
    }

    public FactoryArgs withModel(ModelArgs v) {
        return builder().from(this).model(v).build();
    }

    public FactoryArgs withData(DataArgs v) {
        return builder().from(this).data(v).build();
    }

    public FactoryArgs withFinetuning(FinetuningArgs v) {
        return builder().from(this).finetuning(v).build();
    }

    public FactoryArgs withTraining(TrainingArgs v) {
        return builder().from(this).training(v).build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private ModelArgs model;
        private DataArgs data;
        private FinetuningArgs finetuning;
        private TrainingArgs training;
        private GeneratingArgs generating;
        private EvaluationArgs evaluation;
        private InferArgs infer;
        private ExportArgs export;

        public Builder from(FactoryArgs src) {
            if (src != null) {
                this.model = src.model;
                this.data = src.data;
                this.finetuning = src.finetuning;
                this.training = src.training;
                this.generating = src.generating;
                this.evaluation = src.evaluation;
                this.infer = src.infer;
                this.export = src.export;
            }
            return this;
        }

        public Builder model(ModelArgs v) { this.model = v; return this; }
        public Builder data(DataArgs v) { this.data = v; return this; }
        public Builder finetuning(FinetuningArgs v) { this.finetuning = v; return this; }
        public Builder training(TrainingArgs v) { this.training = v; return this; }
        public Builder generating(GeneratingArgs v) { this.generating = v; return this; }
        public Builder evaluation(EvaluationArgs v) { this.evaluation = v; return this; }
        public Builder infer(InferArgs v) { this.infer = v; return this; }
        public Builder export(ExportArgs v) { this.export = v; return this; }

        /**
         * Fluent nested builder: {@code .model(m -> m.modelNameOrPath("…").flashAttn(true))}.
         * Starts from defaults; successive calls replace the section.
         */
        public Builder model(Consumer<ModelArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "model consumer");
            ModelArgs.Builder b = ModelArgs.builder();
            consumer.accept(b);
            this.model = b.build();
            return this;
        }

        /** Fluent nested builder for {@link DataArgs}. */
        public Builder data(Consumer<DataArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "data consumer");
            DataArgs.Builder b = DataArgs.builder();
            consumer.accept(b);
            this.data = b.build();
            return this;
        }

        /** Fluent nested builder for {@link FinetuningArgs}. */
        public Builder finetuning(Consumer<FinetuningArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "finetuning consumer");
            FinetuningArgs.Builder b = FinetuningArgs.builder();
            consumer.accept(b);
            this.finetuning = b.build();
            return this;
        }

        /** Fluent nested builder for {@link TrainingArgs}. */
        public Builder training(Consumer<TrainingArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "training consumer");
            TrainingArgs.Builder b = TrainingArgs.builder();
            consumer.accept(b);
            this.training = b.build();
            return this;
        }

        /** Fluent nested builder for {@link GeneratingArgs}. */
        public Builder generating(Consumer<GeneratingArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "generating consumer");
            GeneratingArgs.Builder b = GeneratingArgs.builder();
            consumer.accept(b);
            this.generating = b.build();
            return this;
        }

        /** Fluent nested builder for {@link EvaluationArgs}. */
        public Builder evaluation(Consumer<EvaluationArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "evaluation consumer");
            EvaluationArgs.Builder b = EvaluationArgs.builder();
            consumer.accept(b);
            this.evaluation = b.build();
            return this;
        }

        /** Fluent nested builder for {@link InferArgs}. */
        public Builder infer(Consumer<InferArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "infer consumer");
            InferArgs.Builder b = InferArgs.builder();
            consumer.accept(b);
            this.infer = b.build();
            return this;
        }

        /** Fluent nested builder for {@link ExportArgs}. */
        public Builder export(Consumer<ExportArgs.Builder> consumer) {
            Objects.requireNonNull(consumer, "export consumer");
            ExportArgs.Builder b = ExportArgs.builder();
            consumer.accept(b);
            this.export = b.build();
            return this;
        }

        public FactoryArgs build() { return new FactoryArgs(this); }
    }
}

