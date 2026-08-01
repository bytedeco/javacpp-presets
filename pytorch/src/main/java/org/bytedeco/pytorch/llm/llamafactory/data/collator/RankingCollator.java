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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Ranking / multi-response collator for reward-model style lists.
 *
 * <p>Each example may carry {@code input_ids_list} (List of long[]) plus a
 * scalar {@code rank} / {@code score}. Falls back to pairwise chosen/rejected
 * when list form is absent.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RankingCollator implements DataCollator {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final long padTokenId;
    private final int cutoffLen;
    private final PairwiseCollator pairwise;

    public RankingCollator(long padTokenId, int cutoffLen) {
        this.padTokenId = padTokenId;
        this.cutoffLen = cutoffLen <= 0 ? 2048 : cutoffLen;
        this.pairwise = new PairwiseCollator(padTokenId, IGNORE_INDEX, cutoffLen);
    }

    public static RankingCollator defaults() {
        return new RankingCollator(0L, 2048);
    }

    @Override
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        if (features.isEmpty()) {
            throw new IllegalArgumentException("features must be non-empty");
        }
        // If ranking lists present, flatten first two as chosen/rejected for RM bridge
        List<Map<String, Object>> pairFeatures = new ArrayList<>(features.size());
        long[] scores = new long[features.size()];
        for (int i = 0; i < features.size(); i++) {
            Map<String, Object> f = features.get(i);
            Map<String, Object> pair = new LinkedHashMap<>();
            Object list = f.get("input_ids_list");
            if (list instanceof List<?> responses && responses.size() >= 2) {
                pair.put("chosen_input_ids", CollatorUtils.toLongArray(responses.get(0)));
                pair.put("rejected_input_ids", CollatorUtils.toLongArray(responses.get(1)));
            } else {
                pair.putAll(f);
            }
            pairFeatures.add(pair);
            Object sc = f.get("score");
            if (sc == null) sc = f.get("rank");
            scores[i] = sc instanceof Number n ? n.longValue() : 0L;
        }
        Map<String, Tensor> batch = pairwise.collate(pairFeatures);
        batch.put("scores", tensor(scores).to(org.bytedeco.pytorch.global.torch.ScalarType.Long));
        return batch;
    }
}
