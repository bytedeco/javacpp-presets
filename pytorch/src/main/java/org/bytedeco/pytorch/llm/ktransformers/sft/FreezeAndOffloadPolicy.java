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

import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.model.KtMiniMoECausalLM;
import org.bytedeco.pytorch.llm.ktransformers.moe.ExpertDevice;
import org.bytedeco.pytorch.llm.ktransformers.moe.ExpertPool;
import org.bytedeco.pytorch.llm.ktransformers.moe.RoutedMoE;
import org.bytedeco.pytorch.nn.Module;

/**
 * Training-time policy: keep frozen / cold experts on CPU residency to free GPU.
 *
 * <p>Aligns with upstream fine-tune offload knobs. Control-plane only on the
 * reference path (see {@link ExpertPool}).
 */
public final class FreezeAndOffloadPolicy {

    private final boolean offloadFrozen;
    private final int keepGpuExperts;

    public FreezeAndOffloadPolicy(boolean offloadFrozen, int keepGpuExperts) {
        this.offloadFrozen = offloadFrozen;
        this.keepGpuExperts = Math.max(0, keepGpuExperts);
    }

    public static FreezeAndOffloadPolicy from(KtConfig cfg) {
        boolean off = cfg.sft() != null && cfg.sft().offloadFrozen();
        int slots = cfg.moe() != null ? cfg.moe().gpuExpertSlots() : 0;
        return new FreezeAndOffloadPolicy(off, slots);
    }

    public boolean offloadFrozen() { return offloadFrozen; }
    public int keepGpuExperts() { return keepGpuExperts; }

    /** Apply residency hints on a RoutedMoE pool. */
    public void apply(RoutedMoE moe) {
        if (moe == null || !offloadFrozen) return;
        ExpertPool pool = moe.pool();
        for (int i = 0; i < pool.numExperts(); i++) {
            if (i >= keepGpuExperts) {
                pool.demoteToCpu(i);
            }
        }
    }

    public void applyModel(Module model) {
        if (!(model instanceof KtMiniMoECausalLM) || !offloadFrozen) return;
        KtMiniMoECausalLM m = (KtMiniMoECausalLM) model;
        for (KtMiniMoECausalLM.Layer layer : m.layers) {
            apply(layer.moe);
        }
    }

    public static long countGpuExperts(RoutedMoE moe) {
        if (moe == null) return 0;
        long c = 0;
        for (int i = 0; i < moe.pool().numExperts(); i++) {
            if (moe.pool().get(i).device() == ExpertDevice.GPU) c++;
        }
        return c;
    }
}
