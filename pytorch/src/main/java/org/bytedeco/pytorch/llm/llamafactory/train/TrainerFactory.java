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

import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FinetuningArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.Stage;
import org.bytedeco.pytorch.llm.llamafactory.hparams.TrainingArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.peft.PeftModel;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.DPOTrainer;
import org.bytedeco.pytorch.llm.trl.GRPOTrainer;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.ORPOTrainer;
import org.bytedeco.pytorch.llm.trl.PPOTrainer;
import org.bytedeco.pytorch.llm.trl.RewardTrainer;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.DPOConfig;
import org.bytedeco.pytorch.llm.trl.config.GRPOConfig;
import org.bytedeco.pytorch.llm.trl.config.ORPOConfig;
import org.bytedeco.pytorch.llm.trl.config.PPOConfig;
import org.bytedeco.pytorch.llm.trl.config.RewardConfig;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;
import org.bytedeco.pytorch.optim.AdamW;
import org.bytedeco.pytorch.optim.options.AdamWOptions;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Objects;
import java.util.logging.Logger;

/**
 * Builds stage-specific TRL trainers from {@link FactoryArgs} + {@link LoadedModel}.
 *
 * <p>Composition only — loss math stays in {@code llm.trl.*}. Prefer PEFT trainable
 * params when an adapter is attached.
 */
public final class TrainerFactory {

    private static final Logger LOG = Logger.getLogger(TrainerFactory.class.getName());

    private TrainerFactory() {}

    /**
     * Create a trainer for {@code args.finetuning().stage()}.
     *
     * @param maxSteps resolved optimizer steps (from {@link TrainingArgs#effectiveMaxSteps})
     */
    public static BaseTrainer create(FactoryArgs args, LoadedModel loaded, int maxSteps) {
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(loaded, "loaded");
        Stage stage = args.finetuning().stage();
        Module module = loaded.module();
        LlmForward forward = causalForward(loaded);
        Optimizer optimizer = buildOptimizer(args, loaded);
        int steps = Math.max(1, maxSteps);

        return switch (stage) {
            case PT, SFT -> new SFTTrainer(module, forward, optimizer, sftConfig(args, steps));
            case DPO -> new DPOTrainer(module, forward, null, null, optimizer, dpoConfig(args, steps));
            case ORPO -> new ORPOTrainer(module, forward, optimizer, orpoConfig(args, steps));
            case RM -> new RewardTrainer(module, null, optimizer, rewardConfig(args, steps));
            case PPO -> new PPOTrainer(module, null, optimizer, ppoConfig(args, steps));
            case GRPO -> new GRPOTrainer(module, forward, null, null, optimizer, grpoConfig(args, steps));
            case KTO -> {
                // KTO trainer lives in trl when present; fall back to SFT bridge with log note.
                LOG.warning("KTOTrainer not on classpath — using SFTTrainer bridge for KTO stage "
                        + "(install trl KTO when available; collator already emits kto tags)");
                yield new SFTTrainer(module, forward, optimizer, sftConfig(args, steps));
            }
        };
    }

    /** Causal-LM logits forward used by all token-level trainers. */
    public static LlmForward causalForward(LoadedModel loaded) {
        CausalLM causal = loaded.causalLM();
        if (causal != null) {
            return (ids, mask) -> causal.forward(ids);
        }
        Module m = loaded.module();
        return (ids, mask) -> m.forward(ids);
    }

    public static Optimizer buildOptimizer(FactoryArgs args, LoadedModel loaded) {
        TrainingArgs t = args.training();
        double lr = t.learningRate() > 0 ? t.learningRate() : 1e-5;
        double wd = Math.max(0.0, t.weightDecay());
        TensorVector params = trainableParams(loaded);

        // Prefer AdamW (LLaMA-Factory default); fall back to Adam if options path fails.
        try {
            AdamWOptions opt = new AdamWOptions();
            opt.lr().put(lr);
            if (wd > 0.0) {
                opt.weight_decay().put(wd);
            }
            return new AdamW(params, opt);
        } catch (Throwable e) {
            LOG.warning("AdamW unavailable (" + e.getMessage() + "); using Adam");
            return new Adam(params, new AdamOptions(lr));
        }
    }

    public static TensorVector trainableParams(LoadedModel loaded) {
        PeftModel peft = loaded.peft();
        if (peft != null) {
            try {
                TensorVector tv = peft.trainableParameters();
                if (tv != null && tv.size() > 0) {
                    return tv;
                }
            } catch (Throwable ignored) {
            }
        }
        return loaded.module().parameters();
    }

    // ---- config builders ----------------------------------------------------

    public static SFTConfig sftConfig(FactoryArgs args, int maxSteps) {
        TrainingArgs t = args.training();
        return SFTConfig.builder()
                .learningRate(t.learningRate())
                .maxSteps(maxSteps)
                .loggingSteps(Math.max(1, t.loggingSteps()))
                .gradientAccumulationSteps(Math.max(1, t.gradientAccumulationSteps()))
                .maxGradNorm(t.maxGradNorm() > 0 ? t.maxGradNorm() : 1.0)
                .fp16(t.fp16())
                .seed(t.seed())
                .maxSeqLength(Math.max(8, args.data().cutoffLen()))
                .ignoreIndex(-100L)
                .packing(args.data().packing())
                .build();
    }

    public static DPOConfig dpoConfig(FactoryArgs args, int maxSteps) {
        TrainingArgs t = args.training();
        FinetuningArgs f = args.finetuning();
        return DPOConfig.builder()
                .learningRate(t.learningRate())
                .maxSteps(maxSteps)
                .loggingSteps(Math.max(1, t.loggingSteps()))
                .gradientAccumulationSteps(Math.max(1, t.gradientAccumulationSteps()))
                .maxGradNorm(t.maxGradNorm() > 0 ? t.maxGradNorm() : 1.0)
                .fp16(t.fp16())
                .seed(t.seed())
                .beta(f.prefBeta() > 0 ? f.prefBeta() : 0.1)
                .lossType(f.prefLoss() == null || f.prefLoss().isBlank() ? "sigmoid" : f.prefLoss())
                .referenceFree(f.refModel() == null || f.refModel().isBlank())
                .labelSmoothing(Math.max(0.0, f.dpoLabelSmoothing()))
                .build();
    }

    public static ORPOConfig orpoConfig(FactoryArgs args, int maxSteps) {
        TrainingArgs t = args.training();
        FinetuningArgs f = args.finetuning();
        return ORPOConfig.builder()
                .learningRate(t.learningRate())
                .maxSteps(maxSteps)
                .loggingSteps(Math.max(1, t.loggingSteps()))
                .gradientAccumulationSteps(Math.max(1, t.gradientAccumulationSteps()))
                .maxGradNorm(t.maxGradNorm() > 0 ? t.maxGradNorm() : 1.0)
                .fp16(t.fp16())
                .seed(t.seed())
                .beta(f.prefBeta() > 0 ? f.prefBeta() : 0.1)
                .build();
    }

    public static RewardConfig rewardConfig(FactoryArgs args, int maxSteps) {
        TrainingArgs t = args.training();
        return RewardConfig.builder()
                .learningRate(t.learningRate())
                .maxSteps(maxSteps)
                .loggingSteps(Math.max(1, t.loggingSteps()))
                .gradientAccumulationSteps(Math.max(1, t.gradientAccumulationSteps()))
                .maxGradNorm(t.maxGradNorm() > 0 ? t.maxGradNorm() : 1.0)
                .fp16(t.fp16())
                .seed(t.seed())
                .build();
    }

    public static PPOConfig ppoConfig(FactoryArgs args, int maxSteps) {
        TrainingArgs t = args.training();
        return PPOConfig.builder()
                .learningRate(t.learningRate())
                .maxSteps(maxSteps)
                .loggingSteps(Math.max(1, t.loggingSteps()))
                .gradientAccumulationSteps(Math.max(1, t.gradientAccumulationSteps()))
                .maxGradNorm(t.maxGradNorm() > 0 ? t.maxGradNorm() : 1.0)
                .fp16(t.fp16())
                .seed(t.seed())
                .build();
    }

    public static GRPOConfig grpoConfig(FactoryArgs args, int maxSteps) {
        TrainingArgs t = args.training();
        return GRPOConfig.builder()
                .learningRate(t.learningRate())
                .maxSteps(maxSteps)
                .loggingSteps(Math.max(1, t.loggingSteps()))
                .gradientAccumulationSteps(Math.max(1, t.gradientAccumulationSteps()))
                .maxGradNorm(t.maxGradNorm() > 0 ? t.maxGradNorm() : 1.0)
                .fp16(t.fp16())
                .seed(t.seed())
                .build();
    }
}
