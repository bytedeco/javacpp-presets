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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Pairwise preference collator for DPO / ORPO / RM stages.
 *
 * <p>Expected keys per example:
 * <ul>
 *   <li>{@code chosen_input_ids} / {@code chosen_attention_mask} / {@code chosen_labels}</li>
 *   <li>{@code rejected_input_ids} / {@code rejected_attention_mask} / {@code rejected_labels}</li>
 * </ul>
 * Aliases {@code chosen}/{@code rejected} as id lists are also accepted.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PairwiseCollator implements DataCollator {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final long padTokenId;
    private final long ignoreIndex;
    private final int cutoffLen;

    public PairwiseCollator(long padTokenId, long ignoreIndex, int cutoffLen) {
        this.padTokenId = padTokenId;
        this.ignoreIndex = ignoreIndex;
        this.cutoffLen = cutoffLen <= 0 ? 2048 : cutoffLen;
    }

    public PairwiseCollator(long padTokenId) {
        this(padTokenId, IGNORE_INDEX, 2048);
    }

    public static PairwiseCollator defaults() {
        return new PairwiseCollator(0L, IGNORE_INDEX, 2048);
    }

    @Override
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        if (features.isEmpty()) {
            throw new IllegalArgumentException("features must be non-empty");
        }

        List<long[]> chosen = extractAlias(features, "chosen_input_ids", "chosen");
        List<long[]> rejected = extractAlias(features, "rejected_input_ids", "rejected");
        List<long[]> chosenLab = CollatorUtils.extract(features, "chosen_labels");
        List<long[]> rejectedLab = CollatorUtils.extract(features, "rejected_labels");

        int maxC = CollatorUtils.maxLen(chosen, cutoffLen);
        int maxR = CollatorUtils.maxLen(rejected, cutoffLen);
        // pad each side independently (TRL DPO style) — use shared max for simplicity
        int max = Math.max(maxC, maxR);

        long[][] cIds = CollatorUtils.padRight(chosen, max, padTokenId);
        long[][] rIds = CollatorUtils.padRight(rejected, max, padTokenId);
        long[][] cLab = buildLabels(chosen, chosenLab, max);
        long[][] rLab = buildLabels(rejected, rejectedLab, max);
        long[][] cAttn = CollatorUtils.attentionFromPad(cIds, padTokenId);
        long[][] rAttn = CollatorUtils.attentionFromPad(rIds, padTokenId);

        Map<String, Tensor> batch = new LinkedHashMap<>();
        batch.put("chosen_input_ids", CollatorUtils.toLongTensor2d(cIds));
        batch.put("chosen_attention_mask", CollatorUtils.toLongTensor2d(cAttn));
        batch.put("chosen_labels", CollatorUtils.toLongTensor2d(cLab));
        batch.put("rejected_input_ids", CollatorUtils.toLongTensor2d(rIds));
        batch.put("rejected_attention_mask", CollatorUtils.toLongTensor2d(rAttn));
        batch.put("rejected_labels", CollatorUtils.toLongTensor2d(rLab));
        // Convenience aliases used by some TRL bridges
        batch.put("input_ids", batch.get("chosen_input_ids"));
        batch.put("attention_mask", batch.get("chosen_attention_mask"));
        return batch;
    }

    private long[][] buildLabels(List<long[]> ids, List<long[]> labs, int max) {
        long[][] out = new long[ids.size()][max];
        for (int i = 0; i < ids.size(); i++) {
            long[] id = ids.get(i);
            long[] lab = labs.get(i);
            int n = id == null ? 0 : Math.min(id.length, max);
            boolean hasLab = lab != null && lab.length > 0;
            for (int j = 0; j < max; j++) {
                if (j >= n) {
                    out[i][j] = ignoreIndex;
                } else if (hasLab) {
                    out[i][j] = j < lab.length ? lab[j] : ignoreIndex;
                } else {
                    out[i][j] = id[j];
                }
            }
        }
        return out;
    }

    private static List<long[]> extractAlias(
            List<Map<String, Object>> features, String primary, String alias) {
        List<long[]> primaryList = CollatorUtils.extract(features, primary);
        boolean any = false;
        for (long[] a : primaryList) {
            if (a != null && a.length > 0) {
                any = true;
                break;
            }
        }
        if (any) {
            return primaryList;
        }
        return CollatorUtils.extract(features, alias);
    }
}
