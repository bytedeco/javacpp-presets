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
package org.bytedeco.pytorch.llm.llamafactory.data.collator;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * KTO (Kahneman-Tversky Optimization) collator.
 *
 * <p>Expected keys:
 * <ul>
 *   <li>{@code input_ids} / {@code labels} / {@code attention_mask}</li>
 *   <li>{@code kto_tags} — 1 desirable / 0 undesirable (per example)</li>
 * </ul>
 * Also accepts {@code desirable} boolean.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class KtoCollator implements DataCollator {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final SupervisedCollator supervised;

    public KtoCollator(long padTokenId, long ignoreIndex, int cutoffLen) {
        this.supervised = new SupervisedCollator(padTokenId, ignoreIndex, cutoffLen, false);
    }

    public static KtoCollator defaults() {
        return new KtoCollator(0L, IGNORE_INDEX, 2048);
    }

    @Override
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        Map<String, Tensor> batch = supervised.collate(features);
        long[] tags = new long[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Map<String, Object> f = features.get(i);
            tags[i] = desirableOf(f) ? 1L : 0L;
        }
        batch.put("kto_tags", tensor(tags).to(org.bytedeco.pytorch.global.torch.ScalarType.Long));
        return batch;
    }

    private static boolean desirableOf(Map<String, Object> f) {
        if (f == null) return true;
        Object t = f.get("kto_tags");
        if (t == null) t = f.get("desirable");
        if (t == null) t = f.get("label");
        if (t instanceof Boolean b) return b;
        if (t instanceof Number n) return n.intValue() != 0;
        if (t instanceof String s) {
            String lower = s.toLowerCase();
            return !"false".equals(lower) && !"0".equals(lower)
                    && !"undesirable".equals(lower) && !"rejected".equals(lower);
        }
        return true;
    }
}
