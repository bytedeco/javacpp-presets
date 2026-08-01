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
package org.bytedeco.pytorch.llm.llamafactory;

/**
 * Build / capability identity for the pure-Java LLaMA-Factory surface.
 *
 * <p>Logged at the start of every training / serve run so host platforms can
 * pin reproducible stacks.
 */
public final class FactoryVersion {

    /** Semantic version of this factory surface (independent of libtorch). */
    public static final String VERSION = "0.1.0-beta";

    /** Upstream project this surface mirrors. */
    public static final String UPSTREAM = "https://github.com/hiyouga/LLaMA-Factory";

    /** Short capability banner for logs / LlamaBoard. */
    public static final String BANNER =
            "jnitorch-llamafactory " + VERSION
                    + " (stages=pt,sft,rm,ppo,dpo,kto,orpo,grpo;"
                    + " peft=lora,qlora,dora,oft,lora+,loftq,pissa,longlora,ia3;"
                    + " optim=adamw,galore,badam,apollo,adam-mini,muon;"
                    + " monitors=tensorboard,wandb,swanlab,mlflow,llamaboard;"
                    + " infer=openai-api,vllm,board)";

    private FactoryVersion() {}

    /** Human-readable one-liner. */
    public static String info() {
        return BANNER;
    }
}
