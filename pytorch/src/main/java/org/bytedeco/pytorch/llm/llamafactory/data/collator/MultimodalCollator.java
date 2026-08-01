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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros;
import static org.bytedeco.pytorch.global.torch.cat;

/**
 * Multimodal collator for LLaVA / Qwen-VL style rows.
 *
 * <p>Wraps {@link SupervisedCollator} and optionally stacks {@code pixel_values}
 * tensors of shape {@code [C,H,W]} (or pre-batched) into {@code [B,C,H,W]}.
 * Missing pixels yield a zero placeholder of the configured default shape so
 * VL loaders can still run smoke paths offline.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class MultimodalCollator implements DataCollator {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final SupervisedCollator text;
    private final long[] defaultPixelShape; // C,H,W

    public MultimodalCollator(long padTokenId, long ignoreIndex, int cutoffLen, long c, long h, long w) {
        this.text = new SupervisedCollator(padTokenId, ignoreIndex, cutoffLen, false);
        this.defaultPixelShape = new long[]{c, h, w};
    }

    public static MultimodalCollator defaults() {
        return new MultimodalCollator(0L, IGNORE_INDEX, 2048, 3, 224, 224);
    }

    @Override
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        Map<String, Tensor> batch = text.collate(features);

        List<Tensor> pixels = new ArrayList<>(features.size());
        boolean any = false;
        for (Map<String, Object> f : features) {
            Object pv = f.get("pixel_values");
            if (pv == null) pv = f.get("images");
            if (pv instanceof Tensor t) {
                pixels.add(t.dim() == 3 ? t.unsqueeze(0) : t);
                any = true;
            } else {
                pixels.add(zeros(1, defaultPixelShape[0], defaultPixelShape[1], defaultPixelShape[2]));
            }
        }
        if (any || !features.isEmpty()) {
            // cat on batch dim; each element is [1,C,H,W]
            Tensor stacked = pixels.get(0);
            for (int i = 1; i < pixels.size(); i++) {
                stacked = cat(new org.bytedeco.pytorch.TensorVector(new Tensor[]{stacked, pixels.get(i)}), 0);
            }
            batch.put("pixel_values", stacked);
        }
        // image token count placeholder
        long[] nImages = new long[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Object n = features.get(i).get("num_images");
            nImages[i] = n instanceof Number num ? num.longValue() : (features.get(i).containsKey("pixel_values") ? 1L : 0L);
        }
        batch.put("num_images",
                org.bytedeco.pytorch.global.torch.tensor(nImages)
                        .to(org.bytedeco.pytorch.global.torch.ScalarType.Long));
        return batch;
    }
}
