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

package org.bytedeco.pytorch.llm.unsloth.studio.hardware;

/**
 * Rule-of-thumb VRAM estimator for LoRA / QLoRA / full FT planning.
 * Numbers are conservative engineering estimates, not Unsloth marketing claims.
 */
public final class VramEstimator {

    private VramEstimator() {}

    /**
     * @param paramBillions model parameter count in billions (e.g. 8.0 for Llama-3-8B)
     * @param seqLen max sequence length
     * @param batchSize micro-batch size
     * @param loadIn4bit base weights in 4-bit
     * @param lora whether LoRA adapters are trained (not full FT)
     * @param gradCheckpoint gradient checkpointing enabled
     * @return estimated peak VRAM in MiB
     */
    public static long estimateTrainingMb(double paramBillions, int seqLen, int batchSize,
                                          boolean loadIn4bit, boolean lora, boolean gradCheckpoint) {
        if (paramBillions <= 0) paramBillions = 0.1;
        double bytesPerParam = loadIn4bit ? 0.5 : 2.0; // 4-bit ~0.5B, fp16 ~2B
        double weightMb = paramBillions * 1e9 * bytesPerParam / (1024.0 * 1024.0);
        double optimMb = lora
                ? paramBillions * 1e9 * 0.02 * 8.0 / (1024.0 * 1024.0) // ~2% trainable, adam states
                : paramBillions * 1e9 * 8.0 / (1024.0 * 1024.0);       // full adam fp32 states rough
        double actScale = gradCheckpoint ? 0.35 : 1.0;
        double activationMb = paramBillions * seqLen * batchSize * 0.15 * actScale;
        double kvMb = seqLen * batchSize * 0.05 * paramBillions;
        long total = Math.round(weightMb + optimMb + activationMb + kvMb + 512); // +512 overhead
        return Math.max(256, total);
    }

    public static long estimateInferenceMb(double paramBillions, int seqLen, boolean loadIn4bit, int nGpuLayers) {
        double bytesPerParam = loadIn4bit ? 0.5 : 2.0;
        double weightMb = paramBillions * 1e9 * bytesPerParam / (1024.0 * 1024.0);
        if (nGpuLayers >= 0 && nGpuLayers < 32) {
            weightMb *= Math.max(0.1, nGpuLayers / 32.0);
        }
        double kvMb = seqLen * 0.08 * paramBillions;
        return Math.max(128, Math.round(weightMb + kvMb + 256));
    }
}
