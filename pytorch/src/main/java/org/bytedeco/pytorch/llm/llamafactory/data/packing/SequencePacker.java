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
package org.bytedeco.pytorch.llm.llamafactory.data.packing;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Greedy sequence packing for SFT / PT (LLaMA-Factory neat-packing subset).
 *
 * <p>Concatenates tokenized examples until {@code cutoffLen} is reached,
 * inserting optional separator eos between documents. Labels of pad / sep
 * positions keep ignore index.
 */
public final class SequencePacker {

    private final int cutoffLen;
    private final long sepId;
    private final long ignoreIndex;
    private final boolean neat; // respect sample boundaries with sep

    public SequencePacker(int cutoffLen, long sepId, long ignoreIndex, boolean neat) {
        this.cutoffLen = cutoffLen <= 0 ? 2048 : cutoffLen;
        this.sepId = sepId;
        this.ignoreIndex = ignoreIndex;
        this.neat = neat;
    }

    public static SequencePacker defaults() {
        return new SequencePacker(2048, 1L, -100L, true);
    }

    public int cutoffLen() { return cutoffLen; }

    /**
     * Pack a list of already-tokenized feature maps ({@code input_ids}/{@code labels}).
     *
     * @return packed feature maps (usually fewer than input)
     */
    public List<Map<String, Object>> pack(List<Map<String, Object>> features) {
        Objects.requireNonNull(features, "features");
        List<Map<String, Object>> out = new ArrayList<>();
        List<Long> curIds = new ArrayList<>();
        List<Long> curLab = new ArrayList<>();

        for (Map<String, Object> f : features) {
            long[] ids = toLongs(f.get("input_ids"));
            long[] labs = toLongs(f.get("labels"));
            if (labs.length == 0) labs = ids.clone();
            if (ids.length == 0) continue;

            int need = ids.length + (neat && !curIds.isEmpty() ? 1 : 0);
            if (!curIds.isEmpty() && curIds.size() + need > cutoffLen) {
                out.add(flush(curIds, curLab));
                curIds = new ArrayList<>();
                curLab = new ArrayList<>();
            }
            // if single example longer than cutoff — truncate
            if (ids.length > cutoffLen) {
                if (!curIds.isEmpty()) {
                    out.add(flush(curIds, curLab));
                    curIds = new ArrayList<>();
                    curLab = new ArrayList<>();
                }
                long[] tIds = new long[cutoffLen];
                long[] tLab = new long[cutoffLen];
                System.arraycopy(ids, 0, tIds, 0, cutoffLen);
                System.arraycopy(labs, 0, tLab, 0, Math.min(labs.length, cutoffLen));
                for (int i = labs.length; i < cutoffLen; i++) tLab[i] = ignoreIndex;
                Map<String, Object> one = new LinkedHashMap<>();
                one.put("input_ids", tIds);
                one.put("labels", tLab);
                one.put("attention_mask", ones(cutoffLen));
                one.put("prompt_len", 0);
                out.add(one);
                continue;
            }
            if (neat && !curIds.isEmpty()) {
                curIds.add(sepId);
                curLab.add(ignoreIndex);
            }
            for (int i = 0; i < ids.length; i++) {
                curIds.add(ids[i]);
                curLab.add(i < labs.length ? labs[i] : ignoreIndex);
            }
        }
        if (!curIds.isEmpty()) {
            out.add(flush(curIds, curLab));
        }
        return out;
    }

    private Map<String, Object> flush(List<Long> ids, List<Long> labs) {
        int n = ids.size();
        long[] a = new long[n];
        long[] b = new long[n];
        for (int i = 0; i < n; i++) {
            a[i] = ids.get(i);
            b[i] = labs.get(i);
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("input_ids", a);
        m.put("labels", b);
        m.put("attention_mask", ones(n));
        m.put("prompt_len", 0);
        return m;
    }

    private static long[] ones(int n) {
        long[] a = new long[n];
        for (int i = 0; i < n; i++) a[i] = 1L;
        return a;
    }

    private static long[] toLongs(Object v) {
        if (v == null) return new long[0];
        if (v instanceof long[] a) return a;
        if (v instanceof int[] a) {
            long[] o = new long[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (v instanceof List<?> list) {
            long[] o = new long[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object x = list.get(i);
                o[i] = x instanceof Number n ? n.longValue() : 0L;
            }
            return o;
        }
        return new long[0];
    }
}
