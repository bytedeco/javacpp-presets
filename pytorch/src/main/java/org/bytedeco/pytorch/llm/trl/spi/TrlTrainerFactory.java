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
package org.bytedeco.pytorch.llm.trl.spi;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.DPOTrainer;
import org.bytedeco.pytorch.llm.trl.GRPOTrainer;
import org.bytedeco.pytorch.llm.trl.LlmForward;
import org.bytedeco.pytorch.llm.trl.ORPOTrainer;
import org.bytedeco.pytorch.llm.trl.PPOTrainer;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.DPOConfig;
import org.bytedeco.pytorch.llm.trl.config.GRPOConfig;
import org.bytedeco.pytorch.llm.trl.config.ORPOConfig;
import org.bytedeco.pytorch.llm.trl.config.PPOConfig;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Compile-time stable factory for TRL trainers.
 * Studio / host meshes must depend on this SPI — not {@code Class.forName}.
 */
public final class TrlTrainerFactory {

    private TrlTrainerFactory() {}

    public static SFTConfig defaultSftConfig() { return SFTConfig.builder().build(); }
    public static DPOConfig defaultDpoConfig() { return DPOConfig.builder().build(); }
    public static GRPOConfig defaultGrpoConfig() { return GRPOConfig.builder().build(); }
    public static PPOConfig defaultPpoConfig() { return PPOConfig.builder().build(); }
    public static ORPOConfig defaultOrpoConfig() { return ORPOConfig.builder().build(); }

    public static TrainerHandle sft(Module policy, LlmForward forward, Optimizer optim, SFTConfig config) {
        SFTTrainer t = new SFTTrainer(policy, forward, optim,
                config != null ? config : defaultSftConfig());
        return wrap("sft", t);
    }

    public static TrainerHandle dpo(Module policy, LlmForward policyForward,
                                    Module reference, LlmForward referenceForward,
                                    Optimizer optim, DPOConfig config) {
        DPOConfig cfg = config != null ? config : defaultDpoConfig();
        DPOTrainer t = (reference == null && referenceForward == null)
                ? new DPOTrainer(policy, policyForward, optim, cfg)
                : new DPOTrainer(policy, policyForward, reference, referenceForward, optim, cfg);
        return wrap("dpo", t);
    }

    public static TrainerHandle grpo(Module policy, LlmForward policyForward,
                                     Module reference, LlmForward referenceForward,
                                     Optimizer optim, GRPOConfig config) {
        GRPOConfig cfg = config != null ? config : defaultGrpoConfig();
        GRPOTrainer t = (reference == null && referenceForward == null)
                ? new GRPOTrainer(policy, policyForward, optim, cfg)
                : new GRPOTrainer(policy, policyForward, reference, referenceForward, optim, cfg);
        return wrap("grpo", t);
    }

    /**
     * PPO uses {@link PPOTrainer.PolicyValueForward}. Pass {@code null} pvForward for
     * precomputed-advantage batches only.
     */
    public static TrainerHandle ppo(Module policy,
                                    PPOTrainer.PolicyValueForward pvForward,
                                    Optimizer optim, PPOConfig config) {
        PPOConfig cfg = config != null ? config : defaultPpoConfig();
        PPOTrainer t = pvForward == null
                ? new PPOTrainer(policy, optim, cfg)
                : new PPOTrainer(policy, pvForward, optim, cfg);
        return wrap("ppo", t);
    }

    public static TrainerHandle ppoPrecomputed(Module policy, Optimizer optim, PPOConfig config) {
        return ppo(policy, null, optim, config);
    }

    public static TrainerHandle orpo(Module policy, LlmForward policyForward,
                                     Optimizer optim, ORPOConfig config) {
        ORPOTrainer t = new ORPOTrainer(policy, policyForward, optim,
                config != null ? config : defaultOrpoConfig());
        return wrap("orpo", t);
    }

    /**
     * Dispatch by algorithm name: {@code sft|dpo|grpo|ppo|orpo}.
     * PPO uses precomputed-only constructor (no value head required at open time).
     */
    public static TrainerHandle create(String algorithm,
                                       Module policy, LlmForward policyForward,
                                       Module reference, LlmForward referenceForward,
                                       Optimizer optim) {
        String algo = algorithm == null ? "sft" : algorithm.toLowerCase(Locale.ROOT).trim();
        return switch (algo) {
            case "dpo" -> dpo(policy, policyForward, reference, referenceForward, optim, null);
            case "grpo", "rl", "reinforcement", "reinforcement_learning" ->
                    grpo(policy, policyForward, reference, referenceForward, optim, null);
            case "ppo" -> ppoPrecomputed(policy, optim, null);
            case "orpo" -> orpo(policy, policyForward, optim, null);
            default -> sft(policy, policyForward, optim, null);
        };
    }

    private static TrainerHandle wrap(String algo, BaseTrainer trainer) {
        Objects.requireNonNull(trainer, "trainer");
        return new TrainerHandle() {
            @Override public String algorithm() { return algo; }
            @Override public BaseTrainer trainer() { return trainer; }
            @Override public double trainingStep(Map<String, Tensor> batch) {
                return trainer.trainingStep(batch);
            }
            @Override public int globalStep() { return trainer.globalStep(); }
            @Override public void save(Path dir) {
                try {
                    if (dir != null) java.nio.file.Files.createDirectories(dir);
                } catch (Exception ignored) {}
            }
            @Override public void close() { trainer.close(); }
        };
    }
}
