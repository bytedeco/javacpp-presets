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
 * SFT / continuous-pretrain collator (LLaMA-Factory SupervisedDataset collate).
 *
 * <p>Expected per-example keys (any may be pre-tokenized long[]):
 * <ul>
 *   <li>{@code input_ids}</li>
 *   <li>{@code labels} — optional; defaults to {@code input_ids} with optional
 *       prompt-span ignore via {@code prompt_len}</li>
 *   <li>{@code attention_mask} — optional; derived from pad if absent</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SupervisedCollator implements DataCollator {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final long padTokenId;
    private final long ignoreIndex;
    private final int cutoffLen;
    private final boolean trainOnPrompt;

    public SupervisedCollator(long padTokenId, long ignoreIndex, int cutoffLen, boolean trainOnPrompt) {
        this.padTokenId = padTokenId;
        this.ignoreIndex = ignoreIndex;
        this.cutoffLen = cutoffLen <= 0 ? 2048 : cutoffLen;
        this.trainOnPrompt = trainOnPrompt;
    }

    public SupervisedCollator(long padTokenId) {
        this(padTokenId, IGNORE_INDEX, 2048, false);
    }

    public static SupervisedCollator defaults() {
        return new SupervisedCollator(0L, IGNORE_INDEX, 2048, false);
    }

    public long padTokenId() { return padTokenId; }
    public long ignoreIndex() { return ignoreIndex; }
    public int cutoffLen() { return cutoffLen; }
    public boolean trainOnPrompt() { return trainOnPrompt; }

    @Override
    public Map<String, Tensor> collate(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        if (features.isEmpty()) {
            throw new IllegalArgumentException("features must be non-empty");
        }

        List<long[]> inputIds = CollatorUtils.extract(features, "input_ids");
        List<long[]> labelsIn = CollatorUtils.extract(features, "labels");
        boolean hasLabels = false;
        for (long[] l : labelsIn) {
            if (l != null && l.length > 0) {
                hasLabels = true;
                break;
            }
        }

        int max = CollatorUtils.maxLen(inputIds, cutoffLen);
        // also respect labels length
        if (hasLabels) {
            max = Math.max(max, Math.min(cutoffLen, CollatorUtils.maxLen(labelsIn, cutoffLen)));
        }

        long[][] padIds = CollatorUtils.padRight(inputIds, max, padTokenId);
        long[][] padLabels = new long[features.size()][max];

        for (int i = 0; i < features.size(); i++) {
            long[] ids = inputIds.get(i);
            long[] lab = hasLabels ? labelsIn.get(i) : null;
            int promptLen = promptLenOf(features.get(i));
            int n = ids == null ? 0 : Math.min(ids.length, max);
            for (int j = 0; j < max; j++) {
                if (j >= n) {
                    padLabels[i][j] = ignoreIndex;
                } else if (lab != null && lab.length > 0) {
                    padLabels[i][j] = j < lab.length ? lab[j] : ignoreIndex;
                } else if (!trainOnPrompt && promptLen > 0 && j < promptLen) {
                    padLabels[i][j] = ignoreIndex;
                } else {
                    padLabels[i][j] = ids[j];
                }
            }
            // pad positions always ignored
            for (int j = n; j < max; j++) {
                padLabels[i][j] = ignoreIndex;
            }
        }

        long[][] attn;
        List<long[]> attnIn = CollatorUtils.extract(features, "attention_mask");
        boolean hasAttn = false;
        for (long[] a : attnIn) {
            if (a != null && a.length > 0) {
                hasAttn = true;
                break;
            }
        }
        if (hasAttn) {
            attn = CollatorUtils.padRight(attnIn, max, 0L);
        } else {
            attn = CollatorUtils.attentionFromPad(padIds, padTokenId);
        }

        Map<String, Tensor> batch = new LinkedHashMap<>();
        batch.put("input_ids", CollatorUtils.toLongTensor2d(padIds));
        batch.put("labels", CollatorUtils.toLongTensor2d(padLabels));
        batch.put("attention_mask", CollatorUtils.toLongTensor2d(attn));
        return batch;
    }

    private static int promptLenOf(Map<String, Object> f) {
        if (f == null) return 0;
        Object v = f.get("prompt_len");
        if (v == null) v = f.get("source_len");
        if (v instanceof Number n) return Math.max(0, n.intValue());
        return 0;
    }
}
