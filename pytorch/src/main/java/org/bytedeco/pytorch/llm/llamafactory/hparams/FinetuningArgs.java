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

/** Stage / PEFT / preference / advanced-optim args (LLaMA-Factory finetuning section). */
public final class FinetuningArgs {
    private final Stage stage;
    private final FinetuningType finetuningType;
    private final int loraRank;
    private final int loraAlpha;
    private final double loraDropout;
    private final String loraTarget;
    private final String additionalTarget;
    private final double loraplusLrRatio;
    private final boolean useRslora;
    private final boolean useDora;
    private final boolean useOft;
    private final boolean pissaInit;
    private final int pissaIter;
    private final int loftqBits;
    private final boolean createNewAdapter;
    private final int freezeTrainableLayers;
    private final String freezeTrainableModules;
    private final String freezeExtraModules;
    private final boolean useGalore;
    private final int galoreRank;
    private final int galoreUpdateInterval;
    private final double galoreScale;
    private final String galoreTarget;
    private final boolean useApollo;
    private final int apolloRank;
    private final int apolloUpdateInterval;
    private final double apolloScale;
    private final boolean useBadam;
    private final String badamMode;
    private final String badamSwitchMode;
    private final int badamSwitchInterval;
    private final double badamUpdateRatio;
    private final boolean useAdamMini;
    private final boolean useMuon;
    private final boolean pureBf16;
    private final boolean useLlamaPro;
    private final boolean useMixtureOfDepths;
    private final double prefBeta;
    private final String prefLoss;
    private final double prefFtx;
    private final double ktoChosenWeight;
    private final double ktoRejectedWeight;
    private final int ppoBufferSize;
    private final int ppoEpochs;
    private final boolean ppoScoreNorm;
    private final boolean ppoWhitenRewards;
    private final String refModel;
    private final String refModelAdapters;
    private final String rewardModel;
    private final String rewardModelAdapters;
    private final String rewardModelType;
    private final double dpoLabelSmoothing;
    private final double simpoGamma;

    private FinetuningArgs(Builder b) {
        this.stage = b.stage == null ? Stage.SFT : b.stage;
        this.finetuningType = b.finetuningType == null ? FinetuningType.LORA : b.finetuningType;
        this.loraRank = b.loraRank;
        this.loraAlpha = b.loraAlpha;
        this.loraDropout = b.loraDropout;
        this.loraTarget = b.loraTarget == null ? "all" : b.loraTarget;
        this.additionalTarget = b.additionalTarget;
        this.loraplusLrRatio = b.loraplusLrRatio;
        this.useRslora = b.useRslora;
        this.useDora = b.useDora;
        this.useOft = b.useOft;
        this.pissaInit = b.pissaInit;
        this.pissaIter = b.pissaIter;
        this.loftqBits = b.loftqBits;
        this.createNewAdapter = b.createNewAdapter;
        this.freezeTrainableLayers = b.freezeTrainableLayers;
        this.freezeTrainableModules = b.freezeTrainableModules == null ? "all" : b.freezeTrainableModules;
        this.freezeExtraModules = b.freezeExtraModules;
        this.useGalore = b.useGalore;
        this.galoreRank = b.galoreRank;
        this.galoreUpdateInterval = b.galoreUpdateInterval;
        this.galoreScale = b.galoreScale;
        this.galoreTarget = b.galoreTarget == null ? "all" : b.galoreTarget;
        this.useApollo = b.useApollo;
        this.apolloRank = b.apolloRank;
        this.apolloUpdateInterval = b.apolloUpdateInterval;
        this.apolloScale = b.apolloScale;
        this.useBadam = b.useBadam;
        this.badamMode = b.badamMode == null ? "layer" : b.badamMode;
        this.badamSwitchMode = b.badamSwitchMode == null ? "ascending" : b.badamSwitchMode;
        this.badamSwitchInterval = b.badamSwitchInterval;
        this.badamUpdateRatio = b.badamUpdateRatio;
        this.useAdamMini = b.useAdamMini;
        this.useMuon = b.useMuon;
        this.pureBf16 = b.pureBf16;
        this.useLlamaPro = b.useLlamaPro;
        this.useMixtureOfDepths = b.useMixtureOfDepths;
        this.prefBeta = b.prefBeta;
        this.prefLoss = b.prefLoss == null ? "sigmoid" : b.prefLoss;
        this.prefFtx = b.prefFtx;
        this.ktoChosenWeight = b.ktoChosenWeight;
        this.ktoRejectedWeight = b.ktoRejectedWeight;
        this.ppoBufferSize = b.ppoBufferSize;
        this.ppoEpochs = b.ppoEpochs;
        this.ppoScoreNorm = b.ppoScoreNorm;
        this.ppoWhitenRewards = b.ppoWhitenRewards;
        this.refModel = b.refModel;
        this.refModelAdapters = b.refModelAdapters;
        this.rewardModel = b.rewardModel;
        this.rewardModelAdapters = b.rewardModelAdapters;
        this.rewardModelType = b.rewardModelType == null ? "lora" : b.rewardModelType;
        this.dpoLabelSmoothing = b.dpoLabelSmoothing;
        this.simpoGamma = b.simpoGamma;
    }

    public Stage stage() { return stage; }
    public FinetuningType finetuningType() { return finetuningType; }
    public int loraRank() { return loraRank; }
    public int loraAlpha() { return loraAlpha; }
    public int effectiveLoraAlpha() { return loraAlpha <= 0 ? Math.max(1, 2 * loraRank) : loraAlpha; }
    public double loraDropout() { return loraDropout; }
    public String loraTarget() { return loraTarget; }
    public String additionalTarget() { return additionalTarget; }
    public double loraplusLrRatio() { return loraplusLrRatio; }
    public boolean useRslora() { return useRslora; }
    public boolean useDora() { return useDora; }
    public boolean useOft() { return useOft; }
    public boolean pissaInit() { return pissaInit; }
    public int pissaIter() { return pissaIter; }
    public int loftqBits() { return loftqBits; }
    public boolean loftqEnabled() { return loftqBits > 0; }
    public boolean createNewAdapter() { return createNewAdapter; }
    public int freezeTrainableLayers() { return freezeTrainableLayers; }
    public String freezeTrainableModules() { return freezeTrainableModules; }
    public String freezeExtraModules() { return freezeExtraModules; }
    public boolean useGalore() { return useGalore; }
    public int galoreRank() { return galoreRank; }
    public int galoreUpdateInterval() { return galoreUpdateInterval; }
    public double galoreScale() { return galoreScale; }
    public String galoreTarget() { return galoreTarget; }
    public boolean useApollo() { return useApollo; }
    public int apolloRank() { return apolloRank; }
    public int apolloUpdateInterval() { return apolloUpdateInterval; }
    public double apolloScale() { return apolloScale; }
    public boolean useBadam() { return useBadam; }
    public String badamMode() { return badamMode; }
    public String badamSwitchMode() { return badamSwitchMode; }
    public int badamSwitchInterval() { return badamSwitchInterval; }
    public double badamUpdateRatio() { return badamUpdateRatio; }
    public boolean useAdamMini() { return useAdamMini; }
    public boolean useMuon() { return useMuon; }
    public boolean pureBf16() { return pureBf16; }
    public boolean useLlamaPro() { return useLlamaPro; }
    public boolean useMixtureOfDepths() { return useMixtureOfDepths; }
    public double prefBeta() { return prefBeta; }
    public String prefLoss() { return prefLoss; }
    public double prefFtx() { return prefFtx; }
    public double ktoChosenWeight() { return ktoChosenWeight; }
    public double ktoRejectedWeight() { return ktoRejectedWeight; }
    public int ppoBufferSize() { return ppoBufferSize; }
    public int ppoEpochs() { return ppoEpochs; }
    public boolean ppoScoreNorm() { return ppoScoreNorm; }
    public boolean ppoWhitenRewards() { return ppoWhitenRewards; }
    public String refModel() { return refModel; }
    public String refModelAdapters() { return refModelAdapters; }
    public String rewardModel() { return rewardModel; }
    public String rewardModelAdapters() { return rewardModelAdapters; }
    public String rewardModelType() { return rewardModelType; }
    public double dpoLabelSmoothing() { return dpoLabelSmoothing; }
    public double simpoGamma() { return simpoGamma; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "stage", stage.wireName());
        HparamsMaps.put(m, "finetuning_type", finetuningType.wireName());
        HparamsMaps.put(m, "lora_rank", loraRank);
        HparamsMaps.put(m, "lora_alpha", loraAlpha);
        HparamsMaps.put(m, "lora_dropout", loraDropout);
        HparamsMaps.put(m, "lora_target", loraTarget);
        HparamsMaps.put(m, "additional_target", additionalTarget);
        HparamsMaps.put(m, "loraplus_lr_ratio", loraplusLrRatio);
        HparamsMaps.put(m, "use_rslora", useRslora);
        HparamsMaps.put(m, "use_dora", useDora);
        HparamsMaps.put(m, "use_oft", useOft);
        HparamsMaps.put(m, "pissa_init", pissaInit);
        HparamsMaps.put(m, "pissa_iter", pissaIter);
        HparamsMaps.put(m, "loftq_bits", loftqBits);
        HparamsMaps.put(m, "create_new_adapter", createNewAdapter);
        HparamsMaps.put(m, "freeze_trainable_layers", freezeTrainableLayers);
        HparamsMaps.put(m, "freeze_trainable_modules", freezeTrainableModules);
        HparamsMaps.put(m, "freeze_extra_modules", freezeExtraModules);
        HparamsMaps.put(m, "use_galore", useGalore);
        HparamsMaps.put(m, "galore_rank", galoreRank);
        HparamsMaps.put(m, "galore_update_interval", galoreUpdateInterval);
        HparamsMaps.put(m, "galore_scale", galoreScale);
        HparamsMaps.put(m, "galore_target", galoreTarget);
        HparamsMaps.put(m, "use_apollo", useApollo);
        HparamsMaps.put(m, "apollo_rank", apolloRank);
        HparamsMaps.put(m, "apollo_update_interval", apolloUpdateInterval);
        HparamsMaps.put(m, "apollo_scale", apolloScale);
        HparamsMaps.put(m, "use_badam", useBadam);
        HparamsMaps.put(m, "badam_mode", badamMode);
        HparamsMaps.put(m, "badam_switch_mode", badamSwitchMode);
        HparamsMaps.put(m, "badam_switch_interval", badamSwitchInterval);
        HparamsMaps.put(m, "badam_update_ratio", badamUpdateRatio);
        HparamsMaps.put(m, "use_adam_mini", useAdamMini);
        HparamsMaps.put(m, "use_muon", useMuon);
        HparamsMaps.put(m, "pure_bf16", pureBf16);
        HparamsMaps.put(m, "use_llama_pro", useLlamaPro);
        HparamsMaps.put(m, "use_mixture_of_depths", useMixtureOfDepths);
        HparamsMaps.put(m, "pref_beta", prefBeta);
        HparamsMaps.put(m, "pref_loss", prefLoss);
        HparamsMaps.put(m, "pref_ftx", prefFtx);
        HparamsMaps.put(m, "kto_chosen_weight", ktoChosenWeight);
        HparamsMaps.put(m, "kto_rejected_weight", ktoRejectedWeight);
        HparamsMaps.put(m, "ppo_buffer_size", ppoBufferSize);
        HparamsMaps.put(m, "ppo_epochs", ppoEpochs);
        HparamsMaps.put(m, "ppo_score_norm", ppoScoreNorm);
        HparamsMaps.put(m, "ppo_whiten_rewards", ppoWhitenRewards);
        HparamsMaps.put(m, "ref_model", refModel);
        HparamsMaps.put(m, "ref_model_adapters", refModelAdapters);
        HparamsMaps.put(m, "reward_model", rewardModel);
        HparamsMaps.put(m, "reward_model_adapters", rewardModelAdapters);
        HparamsMaps.put(m, "reward_model_type", rewardModelType);
        HparamsMaps.put(m, "dpo_label_smoothing", dpoLabelSmoothing);
        HparamsMaps.put(m, "simpo_gamma", simpoGamma);
        return m;
    }

    public static FinetuningArgs defaults() { return builder().build(); }

    public static FinetuningArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        String stage = HparamsMaps.strOrNull(m, "stage");
        if (stage != null) b.stage(Stage.parse(stage));
        String ft = HparamsMaps.strOrNull(m, "finetuning_type", "finetuning");
        if (ft != null) b.finetuningType(FinetuningType.parse(ft));
        b.loraRank(HparamsMaps.integer(m, b.loraRank, "lora_rank", "lora_r", "r"));
        b.loraAlpha(HparamsMaps.integer(m, b.loraAlpha, "lora_alpha", "alpha"));
        b.loraDropout(HparamsMaps.dbl(m, b.loraDropout, "lora_dropout", "dropout"));
        b.loraTarget(HparamsMaps.str(m, b.loraTarget, "lora_target", "target_modules"));
        b.additionalTarget(HparamsMaps.strOrNull(m, "additional_target"));
        b.loraplusLrRatio(HparamsMaps.dbl(m, b.loraplusLrRatio, "loraplus_lr_ratio", "lora_plus_lr_ratio"));
        b.useRslora(HparamsMaps.bool(m, b.useRslora, "use_rslora", "rslora"));
        b.useDora(HparamsMaps.bool(m, b.useDora, "use_dora", "dora"));
        b.useOft(HparamsMaps.bool(m, b.useOft, "use_oft", "oft"));
        b.pissaInit(HparamsMaps.bool(m, b.pissaInit, "pissa_init", "use_pissa"));
        b.pissaIter(HparamsMaps.integer(m, b.pissaIter, "pissa_iter"));
        b.loftqBits(HparamsMaps.integer(m, b.loftqBits, "loftq_bits", "loftq_config"));
        b.createNewAdapter(HparamsMaps.bool(m, b.createNewAdapter, "create_new_adapter"));
        b.freezeTrainableLayers(HparamsMaps.integer(m, b.freezeTrainableLayers, "freeze_trainable_layers", "num_layer_trainable"));
        b.freezeTrainableModules(HparamsMaps.str(m, b.freezeTrainableModules, "freeze_trainable_modules", "name_module_trainable"));
        b.freezeExtraModules(HparamsMaps.strOrNull(m, "freeze_extra_modules"));
        b.useGalore(HparamsMaps.bool(m, b.useGalore, "use_galore", "galore"));
        b.galoreRank(HparamsMaps.integer(m, b.galoreRank, "galore_rank"));
        b.galoreUpdateInterval(HparamsMaps.integer(m, b.galoreUpdateInterval, "galore_update_interval"));
        b.galoreScale(HparamsMaps.dbl(m, b.galoreScale, "galore_scale"));
        b.galoreTarget(HparamsMaps.str(m, b.galoreTarget, "galore_target"));
        b.useApollo(HparamsMaps.bool(m, b.useApollo, "use_apollo", "apollo"));
        b.apolloRank(HparamsMaps.integer(m, b.apolloRank, "apollo_rank"));
        b.apolloUpdateInterval(HparamsMaps.integer(m, b.apolloUpdateInterval, "apollo_update_interval"));
        b.apolloScale(HparamsMaps.dbl(m, b.apolloScale, "apollo_scale"));
        b.useBadam(HparamsMaps.bool(m, b.useBadam, "use_badam", "badam"));
        b.badamMode(HparamsMaps.str(m, b.badamMode, "badam_mode"));
        b.badamSwitchMode(HparamsMaps.str(m, b.badamSwitchMode, "badam_switch_mode"));
        b.badamSwitchInterval(HparamsMaps.integer(m, b.badamSwitchInterval, "badam_switch_interval"));
        b.badamUpdateRatio(HparamsMaps.dbl(m, b.badamUpdateRatio, "badam_update_ratio"));
        b.useAdamMini(HparamsMaps.bool(m, b.useAdamMini, "use_adam_mini", "adam_mini"));
        b.useMuon(HparamsMaps.bool(m, b.useMuon, "use_muon", "muon"));
        b.pureBf16(HparamsMaps.bool(m, b.pureBf16, "pure_bf16"));
        b.useLlamaPro(HparamsMaps.bool(m, b.useLlamaPro, "use_llama_pro", "llama_pro"));
        b.useMixtureOfDepths(HparamsMaps.bool(m, b.useMixtureOfDepths, "use_mixture_of_depths", "mixture_of_depths"));
        b.prefBeta(HparamsMaps.dbl(m, b.prefBeta, "pref_beta", "dpo_beta", "beta"));
        b.prefLoss(HparamsMaps.str(m, b.prefLoss, "pref_loss", "dpo_loss"));
        b.prefFtx(HparamsMaps.dbl(m, b.prefFtx, "pref_ftx", "dpo_ftx"));
        b.ktoChosenWeight(HparamsMaps.dbl(m, b.ktoChosenWeight, "kto_chosen_weight"));
        b.ktoRejectedWeight(HparamsMaps.dbl(m, b.ktoRejectedWeight, "kto_rejected_weight"));
        b.ppoBufferSize(HparamsMaps.integer(m, b.ppoBufferSize, "ppo_buffer_size"));
        b.ppoEpochs(HparamsMaps.integer(m, b.ppoEpochs, "ppo_epochs"));
        b.ppoScoreNorm(HparamsMaps.bool(m, b.ppoScoreNorm, "ppo_score_norm"));
        b.ppoWhitenRewards(HparamsMaps.bool(m, b.ppoWhitenRewards, "ppo_whiten_rewards"));
        b.refModel(HparamsMaps.strOrNull(m, "ref_model"));
        b.refModelAdapters(HparamsMaps.strOrNull(m, "ref_model_adapters"));
        b.rewardModel(HparamsMaps.strOrNull(m, "reward_model"));
        b.rewardModelAdapters(HparamsMaps.strOrNull(m, "reward_model_adapters"));
        b.rewardModelType(HparamsMaps.str(m, b.rewardModelType, "reward_model_type"));
        b.dpoLabelSmoothing(HparamsMaps.dbl(m, b.dpoLabelSmoothing, "dpo_label_smoothing", "label_smoothing"));
        b.simpoGamma(HparamsMaps.dbl(m, b.simpoGamma, "simpo_gamma"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private Stage stage = Stage.SFT;
        private FinetuningType finetuningType = FinetuningType.LORA;
        private int loraRank = 8;
        private int loraAlpha = 16;
        private double loraDropout;
        private String loraTarget = "all";
        private String additionalTarget;
        private double loraplusLrRatio;
        private boolean useRslora;
        private boolean useDora;
        private boolean useOft;
        private boolean pissaInit;
        private int pissaIter = 16;
        private int loftqBits;
        private boolean createNewAdapter;
        private int freezeTrainableLayers = 2;
        private String freezeTrainableModules = "all";
        private String freezeExtraModules;
        private boolean useGalore;
        private int galoreRank = 128;
        private int galoreUpdateInterval = 50;
        private double galoreScale = 1.0;
        private String galoreTarget = "all";
        private boolean useApollo;
        private int apolloRank = 128;
        private int apolloUpdateInterval = 50;
        private double apolloScale = 1.0;
        private boolean useBadam;
        private String badamMode = "layer";
        private String badamSwitchMode = "ascending";
        private int badamSwitchInterval = 50;
        private double badamUpdateRatio = 0.05;
        private boolean useAdamMini;
        private boolean useMuon;
        private boolean pureBf16;
        private boolean useLlamaPro;
        private boolean useMixtureOfDepths;
        private double prefBeta = 0.1;
        private String prefLoss = "sigmoid";
        private double prefFtx;
        private double ktoChosenWeight = 1.0;
        private double ktoRejectedWeight = 1.0;
        private int ppoBufferSize = 1;
        private int ppoEpochs = 4;
        private boolean ppoScoreNorm;
        private boolean ppoWhitenRewards;
        private String refModel;
        private String refModelAdapters;
        private String rewardModel;
        private String rewardModelAdapters;
        private String rewardModelType = "lora";
        private double dpoLabelSmoothing;
        private double simpoGamma = 0.5;

        public Builder stage(Stage v) { this.stage = Objects.requireNonNull(v); return this; }
        public Builder finetuningType(FinetuningType v) { this.finetuningType = Objects.requireNonNull(v); return this; }
        public Builder loraRank(int v) { this.loraRank = v; return this; }
        public Builder loraAlpha(int v) { this.loraAlpha = v; return this; }
        public Builder loraDropout(double v) { this.loraDropout = v; return this; }
        public Builder loraTarget(String v) { this.loraTarget = v; return this; }
        public Builder additionalTarget(String v) { this.additionalTarget = v; return this; }
        public Builder loraplusLrRatio(double v) { this.loraplusLrRatio = v; return this; }
        public Builder useRslora(boolean v) { this.useRslora = v; return this; }
        public Builder useDora(boolean v) { this.useDora = v; return this; }
        public Builder useOft(boolean v) { this.useOft = v; return this; }
        public Builder pissaInit(boolean v) { this.pissaInit = v; return this; }
        public Builder pissaIter(int v) { this.pissaIter = v; return this; }
        public Builder loftqBits(int v) { this.loftqBits = v; return this; }
        public Builder createNewAdapter(boolean v) { this.createNewAdapter = v; return this; }
        public Builder freezeTrainableLayers(int v) { this.freezeTrainableLayers = v; return this; }
        public Builder freezeTrainableModules(String v) { this.freezeTrainableModules = v; return this; }
        public Builder freezeExtraModules(String v) { this.freezeExtraModules = v; return this; }
        public Builder useGalore(boolean v) { this.useGalore = v; return this; }
        public Builder galoreRank(int v) { this.galoreRank = v; return this; }
        public Builder galoreUpdateInterval(int v) { this.galoreUpdateInterval = v; return this; }
        public Builder galoreScale(double v) { this.galoreScale = v; return this; }
        public Builder galoreTarget(String v) { this.galoreTarget = v; return this; }
        public Builder useApollo(boolean v) { this.useApollo = v; return this; }
        public Builder apolloRank(int v) { this.apolloRank = v; return this; }
        public Builder apolloUpdateInterval(int v) { this.apolloUpdateInterval = v; return this; }
        public Builder apolloScale(double v) { this.apolloScale = v; return this; }
        public Builder useBadam(boolean v) { this.useBadam = v; return this; }
        public Builder badamMode(String v) { this.badamMode = v; return this; }
        public Builder badamSwitchMode(String v) { this.badamSwitchMode = v; return this; }
        public Builder badamSwitchInterval(int v) { this.badamSwitchInterval = v; return this; }
        public Builder badamUpdateRatio(double v) { this.badamUpdateRatio = v; return this; }
        public Builder useAdamMini(boolean v) { this.useAdamMini = v; return this; }
        public Builder useMuon(boolean v) { this.useMuon = v; return this; }
        public Builder pureBf16(boolean v) { this.pureBf16 = v; return this; }
        public Builder useLlamaPro(boolean v) { this.useLlamaPro = v; return this; }
        public Builder useMixtureOfDepths(boolean v) { this.useMixtureOfDepths = v; return this; }
        public Builder prefBeta(double v) { this.prefBeta = v; return this; }
        public Builder prefLoss(String v) { this.prefLoss = v; return this; }
        public Builder prefFtx(double v) { this.prefFtx = v; return this; }
        public Builder ktoChosenWeight(double v) { this.ktoChosenWeight = v; return this; }
        public Builder ktoRejectedWeight(double v) { this.ktoRejectedWeight = v; return this; }
        public Builder ppoBufferSize(int v) { this.ppoBufferSize = v; return this; }
        public Builder ppoEpochs(int v) { this.ppoEpochs = v; return this; }
        public Builder ppoScoreNorm(boolean v) { this.ppoScoreNorm = v; return this; }
        public Builder ppoWhitenRewards(boolean v) { this.ppoWhitenRewards = v; return this; }
        public Builder refModel(String v) { this.refModel = v; return this; }
        public Builder refModelAdapters(String v) { this.refModelAdapters = v; return this; }
        public Builder rewardModel(String v) { this.rewardModel = v; return this; }
        public Builder rewardModelAdapters(String v) { this.rewardModelAdapters = v; return this; }
        public Builder rewardModelType(String v) { this.rewardModelType = v; return this; }
        public Builder dpoLabelSmoothing(double v) { this.dpoLabelSmoothing = v; return this; }
        public Builder simpoGamma(double v) { this.simpoGamma = v; return this; }
        public FinetuningArgs build() { return new FinetuningArgs(this); }
    }
}
