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
package org.bytedeco.pytorch.llm.llamafactory.model.patch;

import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Applies optional model patches requested via {@link ModelArgs} flags
 * (Unsloth / Liger / KTransformers / FlashAttention / LongLoRA shift-attn).
 *
 * <p>Each patch is best-effort: when the corresponding fused op or package is
 * absent, the flag is recorded and the standard modules stay in place. This
 * matches the plan fidelity note — capability flags + thin adapters, not
 * bit-identical CUDA kernel parity.
 */
public final class ModelPatcher {

    private static final Logger LOG = Logger.getLogger(ModelPatcher.class.getName());

    private ModelPatcher() {}

    public static Map<String, Object> apply(Module model, ModelArgs args) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(args, "args");
        Map<String, Object> report = new LinkedHashMap<>();
        List<String> applied = new ArrayList<>();
        List<String> skipped = new ArrayList<>();

        if (args.flashAttn()) {
            if (FlashAttnPatch.apply(model)) {
                applied.add("flash_attn");
            } else {
                skipped.add("flash_attn");
            }
        }
        if (args.shiftAttn()) {
            if (LongLoraShiftPatch.apply(model)) {
                applied.add("shift_attn");
            } else {
                skipped.add("shift_attn");
            }
        }
        if (args.useUnsloth()) {
            if (UnslothPatch.apply(model, args)) {
                applied.add("unsloth");
            } else {
                skipped.add("unsloth");
            }
        }
        if (args.useLigerKernel()) {
            if (LigerKernelPatch.apply(model)) {
                applied.add("liger");
            } else {
                skipped.add("liger");
            }
        }
        if (args.useKtransformers()) {
            if (KTransformersPatch.apply(model)) {
                applied.add("ktransformers");
            } else {
                skipped.add("ktransformers");
            }
        }

        report.put("applied", applied);
        report.put("skipped", skipped);
        if (!applied.isEmpty() || !skipped.isEmpty()) {
            LOG.info("ModelPatcher applied=" + applied + " skipped=" + skipped);
        }
        return report;
    }
}
