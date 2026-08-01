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
package org.bytedeco.pytorch.llm.ktransformers.inject.families;

import org.bytedeco.pytorch.llm.ktransformers.config.KtModelFamily;
import org.bytedeco.pytorch.llm.ktransformers.config.KtQuantConfig;
import org.bytedeco.pytorch.llm.ktransformers.inject.LayerInjectPlan;

/** Generic fallback plan (used when family unknown). */
public final class GenericFamilyInject {

    private GenericFamilyInject() {}

    public static LayerInjectPlan plan() {
        return LayerInjectPlan.builder(KtModelFamily.GENERIC)
                .attentionKind(LayerInjectPlan.AttentionKind.STANDARD)
                .moe(false)
                .quantLinearGlobs("*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj",
                        "*.gate_proj", "*.up_proj", "*.down_proj")
                .recommendedQuant(KtQuantConfig.int8AmxLike())
                .note("layout", "user-supplied layer map / generic HF names")
                .note("upstream", "https://github.com/kvcache-ai/ktransformers")
                .build();
    }
}
