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
 * Shared pad / stack helpers for factory collators.
 *
 * <p>Accepts {@code long[]}, {@code int[]}, {@code List<? extends Number>}, or
 * 1-D {@link Tensor} feature values and produces left-or-right padded 2-D
 * Long tensors.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class CollatorUtils {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private CollatorUtils() {}

    public static long[] toLongArray(Object value) {
        if (value == null) {
            return new long[0];
        }
        if (value instanceof long[] arr) {
            return arr;
        }
        if (value instanceof int[] arr) {
            long[] out = new long[arr.length];
            for (int i = 0; i < arr.length; i++) {
                out[i] = arr[i];
            }
            return out;
        }
        if (value instanceof List<?> list) {
            long[] out = new long[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object o = list.get(i);
                out[i] = o == null ? 0L : ((Number) o).longValue();
            }
            return out;
        }
        if (value instanceof Tensor t) {
            long n = t.numel();
            long[] out = new long[(int) n];
            Tensor flat = t.contiguous().view(n);
            for (int i = 0; i < out.length; i++) {
                out[i] = flat.get( i).item_long();
            }
            return out;
        }
        if (value instanceof Number n) {
            return new long[]{n.longValue()};
        }
        throw new IllegalArgumentException(
                "Unsupported feature type: " + value.getClass().getName());
    }

    public static int maxLen(List<long[]> sequences, int cutoff) {
        int max = 0;
        for (long[] s : sequences) {
            max = Math.max(max, s == null ? 0 : s.length);
        }
        if (cutoff > 0) {
            max = Math.min(max, cutoff);
        }
        return Math.max(1, max);
    }

    /**
     * Right-pad sequences to {@code maxLen} with {@code padId}. Truncates from
     * the right when longer than {@code maxLen}.
     */
    public static long[][] padRight(List<long[]> sequences, int maxLen, long padId) {
        long[][] out = new long[sequences.size()][maxLen];
        for (int i = 0; i < sequences.size(); i++) {
            long[] s = sequences.get(i);
            int n = s == null ? 0 : Math.min(s.length, maxLen);
            for (int j = 0; j < n; j++) {
                out[i][j] = s[j];
            }
            for (int j = n; j < maxLen; j++) {
                out[i][j] = padId;
            }
        }
        return out;
    }

    /** Attention mask: 1 for real tokens, 0 for pad. */
    public static long[][] attentionFromPad(long[][] padded, long padId) {
        int b = padded.length;
        int t = b == 0 ? 0 : padded[0].length;
        long[][] mask = new long[b][t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                mask[i][j] = padded[i][j] == padId ? 0L : 1L;
            }
        }
        return mask;
    }

    public static Tensor toLongTensor2d(long[][] data) {
        Objects.requireNonNull(data, "data");
        int b = data.length;
        int t = b == 0 ? 0 : data[0].length;
        long[] flat = new long[b * t];
        for (int i = 0; i < b; i++) {
            System.arraycopy(data[i], 0, flat, i * t, t);
        }
        return tensor(flat).view(b, t).to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
    }

    public static Tensor toLongTensor1d(long[] data) {
        return tensor(data).to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
    }

    /**
     * Extract key as list of long[] from feature maps. Missing keys yield empty arrays.
     */
    public static List<long[]> extract(List<Map<String, Object>> features, String key) {
        List<long[]> out = new ArrayList<>(features.size());
        for (Map<String, Object> f : features) {
            out.add(toLongArray(f == null ? null : f.get(key)));
        }
        return out;
    }

    public static Map<String, Tensor> singleKeyBatch(
            List<Map<String, Object>> features,
            String key,
            long padId,
            int cutoff) {
        List<long[]> seqs = extract(features, key);
        int max = maxLen(seqs, cutoff);
        long[][] padded = padRight(seqs, max, padId);
        Map<String, Tensor> batch = new LinkedHashMap<>();
        batch.put(key, toLongTensor2d(padded));
        return batch;
    }
}
