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
package org.bytedeco.pytorch.utils.tokenizers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code tokenizers}-style encoding result.
 *
 * <pre>{@code
 * Encoding enc = tok.encode("Hello world", true);
 * int[] ids = enc.ids();
 * int[] mask = enc.attentionMask();
 * }</pre>
 */
public final class Encoding {

    private final int[] ids;
    private final int[] typeIds;
    private final int[] attentionMask;
    private final int[] specialTokensMask;
    private final List<String> tokens;
    private final List<Integer> offsetsStart;
    private final List<Integer> offsetsEnd;
    private final Integer overflowingOf; // index into parent batch, or null

    private Encoding(Builder b) {
        this.ids = b.ids == null ? new int[0] : Arrays.copyOf(b.ids, b.ids.length);
        this.typeIds = b.typeIds == null ? new int[ids.length] : Arrays.copyOf(b.typeIds, b.typeIds.length);
        this.attentionMask = b.attentionMask == null
                ? ones(ids.length) : Arrays.copyOf(b.attentionMask, b.attentionMask.length);
        this.specialTokensMask = b.specialTokensMask == null
                ? new int[ids.length] : Arrays.copyOf(b.specialTokensMask, b.specialTokensMask.length);
        this.tokens = b.tokens == null ? List.of() : Collections.unmodifiableList(new ArrayList<>(b.tokens));
        this.offsetsStart = b.offsetsStart == null ? List.of()
                : Collections.unmodifiableList(new ArrayList<>(b.offsetsStart));
        this.offsetsEnd = b.offsetsEnd == null ? List.of()
                : Collections.unmodifiableList(new ArrayList<>(b.offsetsEnd));
        this.overflowingOf = b.overflowingOf;
    }

    private static int[] ones(int n) {
        int[] a = new int[n];
        Arrays.fill(a, 1);
        return a;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int[] ids() {
        return Arrays.copyOf(ids, ids.length);
    }

    public int[] typeIds() {
        return Arrays.copyOf(typeIds, typeIds.length);
    }

    public int[] attentionMask() {
        return Arrays.copyOf(attentionMask, attentionMask.length);
    }

    public int[] specialTokensMask() {
        return Arrays.copyOf(specialTokensMask, specialTokensMask.length);
    }

    public List<String> tokens() {
        return tokens;
    }

    public int size() {
        return ids.length;
    }

    public int length() {
        return ids.length;
    }

    /** Char offsets start (parallel to tokens), empty if not tracked. */
    public List<Integer> offsetsStart() {
        return offsetsStart;
    }

    public List<Integer> offsetsEnd() {
        return offsetsEnd;
    }

    public Integer overflowingOf() {
        return overflowingOf;
    }

    /** Right-pad to {@code maxLen} with {@code padId} (backward-compatible). */
    public Encoding padTo(int maxLen, int padId, int padTypeId) {
        return padTo(maxLen, padId, padTypeId, "right");
    }

    /**
     * Pad to {@code maxLen}.
     *
     * @param direction {@code "right"} (default) or {@code "left"}
     */
    public Encoding padTo(int maxLen, int padId, int padTypeId, String direction) {
        if (maxLen < 0) {
            throw new IllegalArgumentException("maxLen must be >= 0");
        }
        if (ids.length == maxLen) {
            return this;
        }
        if (ids.length > maxLen) {
            return truncate(maxLen, direction);
        }
        boolean left = direction != null && direction.equalsIgnoreCase("left");
        int[] nIds = new int[maxLen];
        int[] nType = new int[maxLen];
        int[] nMask = new int[maxLen];
        int[] nSpec = new int[maxLen];
        Arrays.fill(nIds, padId);
        Arrays.fill(nType, padTypeId);
        // attention / special default 0 for pad slots
        int copy = ids.length;
        int dest = left ? (maxLen - copy) : 0;
        System.arraycopy(ids, 0, nIds, dest, copy);
        System.arraycopy(typeIds, 0, nType, dest, Math.min(typeIds.length, copy));
        System.arraycopy(attentionMask, 0, nMask, dest, Math.min(attentionMask.length, copy));
        System.arraycopy(specialTokensMask, 0, nSpec, dest, Math.min(specialTokensMask.length, copy));
        List<String> nTok = new ArrayList<>(maxLen);
        List<Integer> nOffS = new ArrayList<>(maxLen);
        List<Integer> nOffE = new ArrayList<>(maxLen);
        if (left) {
            for (int i = 0; i < maxLen - copy; i++) {
                nTok.add("[PAD]");
                nOffS.add(0);
                nOffE.add(0);
            }
        }
        for (int i = 0; i < copy; i++) {
            nTok.add(i < tokens.size() ? tokens.get(i) : "");
            nOffS.add(i < offsetsStart.size() ? offsetsStart.get(i) : 0);
            nOffE.add(i < offsetsEnd.size() ? offsetsEnd.get(i) : 0);
        }
        if (!left) {
            while (nTok.size() < maxLen) {
                nTok.add("[PAD]");
                nOffS.add(0);
                nOffE.add(0);
            }
        }
        return builder()
                .ids(nIds)
                .typeIds(nType)
                .attentionMask(nMask)
                .specialTokensMask(nSpec)
                .tokens(nTok)
                .offsetsStart(nOffS)
                .offsetsEnd(nOffE)
                .build();
    }

    /** Right-truncate to {@code maxLen} (backward-compatible). */
    public Encoding truncate(int maxLen) {
        return truncate(maxLen, "right");
    }

    /**
     * Truncate to {@code maxLen}.
     *
     * @param direction {@code "right"} keeps a prefix; {@code "left"} keeps a suffix
     */
    public Encoding truncate(int maxLen, String direction) {
        if (maxLen < 0) {
            throw new IllegalArgumentException("maxLen must be >= 0");
        }
        if (ids.length <= maxLen) {
            return this;
        }
        boolean left = direction != null && direction.equalsIgnoreCase("left");
        int from = left ? (ids.length - maxLen) : 0;
        int[] nIds = Arrays.copyOfRange(ids, from, from + maxLen);
        int[] nType = typeIds.length >= from + maxLen
                ? Arrays.copyOfRange(typeIds, from, from + maxLen)
                : Arrays.copyOf(typeIds, maxLen);
        int[] nMask = attentionMask.length >= from + maxLen
                ? Arrays.copyOfRange(attentionMask, from, from + maxLen)
                : Arrays.copyOf(attentionMask, maxLen);
        int[] nSpec = specialTokensMask.length >= from + maxLen
                ? Arrays.copyOfRange(specialTokensMask, from, from + maxLen)
                : Arrays.copyOf(specialTokensMask, maxLen);
        List<String> nTok = new ArrayList<>(maxLen);
        List<Integer> nOffS = new ArrayList<>(maxLen);
        List<Integer> nOffE = new ArrayList<>(maxLen);
        for (int i = from; i < from + maxLen; i++) {
            nTok.add(i < tokens.size() ? tokens.get(i) : "");
            nOffS.add(i < offsetsStart.size() ? offsetsStart.get(i) : 0);
            nOffE.add(i < offsetsEnd.size() ? offsetsEnd.get(i) : 0);
        }
        return builder()
                .ids(nIds)
                .typeIds(nType)
                .attentionMask(nMask)
                .specialTokensMask(nSpec)
                .tokens(nTok)
                .offsetsStart(nOffS)
                .offsetsEnd(nOffE)
                .build();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("ids", ids());
        m.put("type_ids", typeIds());
        m.put("attention_mask", attentionMask());
        m.put("special_tokens_mask", specialTokensMask());
        m.put("tokens", new ArrayList<>(tokens));
        return m;
    }

    @Override
    public String toString() {
        return "Encoding{len=" + ids.length + ", tokens=" + tokens + ", ids=" + Arrays.toString(ids) + "}";
    }

    public static final class Builder {
        private int[] ids;
        private int[] typeIds;
        private int[] attentionMask;
        private int[] specialTokensMask;
        private List<String> tokens;
        private List<Integer> offsetsStart;
        private List<Integer> offsetsEnd;
        private Integer overflowingOf;

        public Builder ids(int[] ids) {
            this.ids = ids;
            return this;
        }

        public Builder typeIds(int[] typeIds) {
            this.typeIds = typeIds;
            return this;
        }

        public Builder attentionMask(int[] attentionMask) {
            this.attentionMask = attentionMask;
            return this;
        }

        public Builder specialTokensMask(int[] specialTokensMask) {
            this.specialTokensMask = specialTokensMask;
            return this;
        }

        public Builder tokens(List<String> tokens) {
            this.tokens = tokens;
            return this;
        }

        public Builder offsetsStart(List<Integer> offsetsStart) {
            this.offsetsStart = offsetsStart;
            return this;
        }

        public Builder offsetsEnd(List<Integer> offsetsEnd) {
            this.offsetsEnd = offsetsEnd;
            return this;
        }

        public Builder overflowingOf(Integer overflowingOf) {
            this.overflowingOf = overflowingOf;
            return this;
        }

        public Encoding build() {
            return new Encoding(this);
        }
    }
}
