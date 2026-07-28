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
package org.bytedeco.pytorch.llm.tokenizers;

import org.bytedeco.pytorch.llm.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreToken;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * HuggingFace {@code AddedVocabulary}: greedy longest-match split for added / special tokens.
 *
 * <p>Pieces that match an added token become atomic {@link PreToken}s with {@code added=true}
 * and bypass the model BPE/Unigram. Non-matched spans are ordinary text for the normalizer
 * + pretokenizer + model pipeline.
 */
public final class AddedVocabulary {

    private final List<AddedToken> tokens;                 // length-desc sorted
    private final Map<String, AddedToken> byContent;
    private final Map<Integer, AddedToken> byId;
    private final Set<Integer> specialIds;
    private final Set<String> specialContents;

    public AddedVocabulary(List<AddedToken> tokens) {
        List<AddedToken> copy = new ArrayList<>(tokens == null ? List.of() : tokens);
        // Longest content first for greedy match; tie-break by id
        copy.sort(Comparator
                .comparingInt((AddedToken t) -> t.content().length()).reversed()
                .thenComparingInt(AddedToken::id));
        this.tokens = List.copyOf(copy);
        this.byContent = new HashMap<>();
        this.byId = new HashMap<>();
        this.specialIds = new HashSet<>();
        this.specialContents = new HashSet<>();
        for (AddedToken t : this.tokens) {
            byContent.put(t.content(), t);
            byId.put(t.id(), t);
            if (t.special()) {
                specialIds.add(t.id());
                specialContents.add(t.content());
            }
        }
    }

    public static AddedVocabulary empty() {
        return new AddedVocabulary(List.of());
    }

    public static AddedVocabulary fromJsonList(List<Object> raw) {
        if (raw == null || raw.isEmpty()) return empty();
        List<AddedToken> list = new ArrayList<>();
        for (Object o : raw) {
            Map<String, Object> m = JsonMaps.asMap(o);
            if (m != null) list.add(AddedToken.fromJson(m));
        }
        return new AddedVocabulary(list);
    }

    public List<AddedToken> tokens() { return tokens; }
    public boolean isSpecialId(int id) { return specialIds.contains(id); }
    public boolean isSpecialContent(String s) { return s != null && specialContents.contains(s); }
    public AddedToken getById(int id) { return byId.get(id); }
    public AddedToken getByContent(String c) { return byContent.get(c); }
    public Map<String, AddedToken> byContent() { return Collections.unmodifiableMap(byContent); }
    public Map<Integer, AddedToken> byId() { return Collections.unmodifiableMap(byId); }

    /**
     * Split {@code text} into ordinary spans and added-token spans.
     *
     * @param normalized whether this pass matches tokens with {@code normalized=true}
     *                   (true) or {@code normalized=false} (false, raw text)
     */
    public List<PreToken> split(String text, boolean normalized) {
        if (text == null || text.isEmpty()) return List.of();
        if (tokens.isEmpty()) {
            return List.of(new PreToken(text, 0, text.length()));
        }

        // Candidates that participate in this pass
        List<AddedToken> candidates = new ArrayList<>();
        for (AddedToken t : tokens) {
            if (t.normalized() == normalized) candidates.add(t);
        }
        if (candidates.isEmpty()) {
            return List.of(new PreToken(text, 0, text.length()));
        }

        List<PreToken> out = new ArrayList<>();
        int i = 0;
        int n = text.length();
        while (i < n) {
            AddedToken match = null;
            for (AddedToken t : candidates) {
                String c = t.content();
                if (c.isEmpty()) continue;
                if (i + c.length() > n) continue;
                if (text.startsWith(c, i)) {
                    if (t.singleWord()) {
                        // Require non-word char (or boundary) on both sides
                        if (!isWordBoundary(text, i, i + c.length())) continue;
                    }
                    match = t;
                    break; // candidates already longest-first
                }
            }
            if (match != null) {
                int start = i;
                int end = i + match.content().length();
                // lstrip: absorb spaces before into the token's left (HF: strip from left of match
                // meaning spaces immediately before are dropped from ordinary text — implemented
                // by not including them in previous ordinary span; they're skipped here)
                if (match.lstrip()) {
                    // spaces already in previous ordinary piece — HF lstrip removes whitespace
                    // to the left of the token from the *match context* when encoding; for split
                    // we simply don't extend. Common case lstrip=false.
                }
                if (match.rstrip()) {
                    while (end < n && text.charAt(end) == ' ') end++;
                }
                // Emit ordinary text before match
                // (handled by scanning)
                out.add(PreToken.added(match.content(), start, i + match.content().length(), match.id()));
                i = end;
            } else {
                // Accumulate ordinary run until next possible match
                int s = i;
                i++;
                while (i < n) {
                    boolean canMatch = false;
                    for (AddedToken t : candidates) {
                        String c = t.content();
                        if (!c.isEmpty() && i + c.length() <= n && text.startsWith(c, i)) {
                            if (!t.singleWord() || isWordBoundary(text, i, i + c.length())) {
                                canMatch = true;
                                break;
                            }
                        }
                    }
                    if (canMatch) break;
                    i++;
                }
                out.add(new PreToken(text.substring(s, i), s, i));
            }
        }
        return out;
    }

    /**
     * Full encode-time split used by the pipeline:
     * 1) split raw text on non-normalized added tokens
     * 2) for ordinary spans: optionally normalize, then split on normalized added tokens
     */
    public List<Segment> splitForEncode(String text, Normalizer normalizer) {
        List<Segment> out = new ArrayList<>();
        List<PreToken> rawParts = split(text == null ? "" : text, false);
        for (PreToken p : rawParts) {
            if (p.added()) {
                out.add(Segment.added(p));
                continue;
            }
            String ordinary = p.value();
            String norm = normalizer == null ? ordinary : normalizer.normalize(ordinary);
            // Offsets on normalized text may diverge; keep pretok-relative for model.
            List<PreToken> normParts = split(norm, true);
            if (normParts.isEmpty()) {
                // nothing
            } else if (normParts.size() == 1 && !normParts.get(0).added()) {
                out.add(Segment.ordinary(normParts.get(0).value(), p.start(), p.end()));
            } else {
                for (PreToken np : normParts) {
                    if (np.added()) out.add(Segment.added(np));
                    else out.add(Segment.ordinary(np.value(), p.start(), p.end()));
                }
            }
        }
        return out;
    }

    private static boolean isWordBoundary(String text, int start, int end) {
        if (start > 0 && isWordChar(text.charAt(start - 1)) && isWordChar(text.charAt(start))) {
            return false;
        }
        if (end < text.length() && isWordChar(text.charAt(end - 1)) && isWordChar(text.charAt(end))) {
            return false;
        }
        return true;
    }

    private static boolean isWordChar(char c) {
        return Character.isLetterOrDigit(c) || c == '_';
    }

    /** A segment after added-token splitting: either ordinary text or an added token. */
    public static final class Segment {
        public final String value;
        public final int start;
        public final int end;
        public final boolean added;
        public final int addedId;

        private Segment(String value, int start, int end, boolean added, int addedId) {
            this.value = value;
            this.start = start;
            this.end = end;
            this.added = added;
            this.addedId = addedId;
        }

        public static Segment ordinary(String value, int start, int end) {
            return new Segment(value, start, end, false, -1);
        }

        public static Segment added(PreToken p) {
            return new Segment(p.value(), p.start(), p.end(), true, p.addedId());
        }

        public PreToken toPreToken() {
            return added
                    ? PreToken.added(value, start, end, addedId)
                    : new PreToken(value, start, end);
        }
    }
}
