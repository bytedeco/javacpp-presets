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

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HuggingFace {@code Split} pretokenizer behaviors over a regex pattern.
 */
public final class RegexSplit {

    public enum Behavior {
        REMOVED,
        ISOLATED,
        MERGED_WITH_PREVIOUS,
        MERGED_WITH_NEXT,
        CONTIGUOUS;

        public static Behavior fromString(String s) {
            if (s == null) return ISOLATED;
            return switch (s) {
                case "Removed" -> REMOVED;
                case "Isolated" -> ISOLATED;
                case "MergedWithPrevious" -> MERGED_WITH_PREVIOUS;
                case "MergedWithNext" -> MERGED_WITH_NEXT;
                case "Contiguous" -> CONTIGUOUS;
                default -> ISOLATED;
            };
        }
    }

    private RegexSplit() {}

    /**
     * Split {@code text} according to HF Split semantics.
     * Offsets are relative to {@code text}.
     */
    public static List<Span> split(String text, Pattern pattern, Behavior behavior, boolean invert) {
        if (text == null || text.isEmpty()) return List.of();
        if (pattern == null) {
            return List.of(new Span(text, 0, text.length()));
        }
        List<Span> matches = new ArrayList<>();
        List<Span> gaps = new ArrayList<>();
        Matcher m = pattern.matcher(text);
        int prev = 0;
        while (m.find()) {
            int s = m.start();
            int e = m.end();
            if (s > prev) {
                gaps.add(new Span(text.substring(prev, s), prev, s));
            }
            if (e > s) {
                matches.add(new Span(text.substring(s, e), s, e));
            }
            prev = e;
        }
        if (prev < text.length()) {
            gaps.add(new Span(text.substring(prev), prev, text.length()));
        }

        List<Span> primary = invert ? gaps : matches;
        List<Span> secondary = invert ? matches : gaps;

        return switch (behavior) {
            case REMOVED -> filterEmpty(primary);
            case ISOLATED -> {
                // All pieces in order: interleave gaps and matches as they appear in text
                List<Span> all = new ArrayList<>();
                int i = 0, j = 0;
                // Rebuild in offset order from both lists
                List<Span> ordered = new ArrayList<>(matches.size() + gaps.size());
                ordered.addAll(matches);
                ordered.addAll(gaps);
                ordered.sort((a, b) -> Integer.compare(a.start, b.start));
                // For Isolated + invert=false: keep matches only? No — HF Isolated keeps matches
                // as isolated tokens; non-matches (gaps) are also kept as separate tokens.
                // Actually in tokenizers-rs Split:
                // - Isolated: each match is its own token; parts between matches are also tokens
                // So both matches and gaps are kept, sorted by offset.
                for (Span sp : ordered) {
                    if (!sp.value.isEmpty()) all.add(sp);
                }
                yield all;
            }
            case MERGED_WITH_PREVIOUS -> mergeWithPrevious(text, matches, gaps, invert);
            case MERGED_WITH_NEXT -> mergeWithNext(text, matches, gaps, invert);
            case CONTIGUOUS -> contiguous(primary);
        };
    }

    private static List<Span> filterEmpty(List<Span> in) {
        List<Span> out = new ArrayList<>(in.size());
        for (Span s : in) if (!s.value.isEmpty()) out.add(s);
        return out;
    }

    private static List<Span> contiguous(List<Span> primary) {
        // Merge consecutive primary spans
        List<Span> filtered = filterEmpty(primary);
        if (filtered.isEmpty()) return filtered;
        List<Span> out = new ArrayList<>();
        Span cur = filtered.get(0);
        for (int i = 1; i < filtered.size(); i++) {
            Span n = filtered.get(i);
            if (n.start == cur.end) {
                cur = new Span(cur.value + n.value, cur.start, n.end);
            } else {
                out.add(cur);
                cur = n;
            }
        }
        out.add(cur);
        return out;
    }

    private static List<Span> mergeWithPrevious(String text, List<Span> matches, List<Span> gaps, boolean invert) {
        // Simplified: produce ordered pieces; attach each match to previous gap
        List<Span> ordered = new ArrayList<>();
        ordered.addAll(matches);
        ordered.addAll(gaps);
        ordered.sort((a, b) -> Integer.compare(a.start, b.start));
        if (ordered.isEmpty()) return ordered;
        List<Span> out = new ArrayList<>();
        StringBuilder buf = new StringBuilder();
        int start = ordered.get(0).start;
        int end = ordered.get(0).end;
        buf.append(ordered.get(0).value);
        boolean lastWasMatch = matchesContains(matches, ordered.get(0));
        for (int i = 1; i < ordered.size(); i++) {
            Span sp = ordered.get(i);
            boolean isMatch = matchesContains(matches, sp);
            // When invert, roles flip conceptually — keep practical: always merge match into prev
            if (isMatch != invert) {
                // match → merge into previous buffer
                buf.append(sp.value);
                end = sp.end;
            } else {
                if (buf.length() > 0) out.add(new Span(buf.toString(), start, end));
                buf.setLength(0);
                buf.append(sp.value);
                start = sp.start;
                end = sp.end;
            }
            lastWasMatch = isMatch;
        }
        if (buf.length() > 0) out.add(new Span(buf.toString(), start, end));
        return out;
    }

    private static List<Span> mergeWithNext(String text, List<Span> matches, List<Span> gaps, boolean invert) {
        List<Span> ordered = new ArrayList<>();
        ordered.addAll(matches);
        ordered.addAll(gaps);
        ordered.sort((a, b) -> Integer.compare(a.start, b.start));
        if (ordered.isEmpty()) return ordered;
        List<Span> out = new ArrayList<>();
        int i = 0;
        while (i < ordered.size()) {
            Span sp = ordered.get(i);
            boolean isMatch = matchesContains(matches, sp);
            if (isMatch != invert && i + 1 < ordered.size()) {
                // merge match with next
                Span next = ordered.get(i + 1);
                out.add(new Span(sp.value + next.value, sp.start, next.end));
                i += 2;
            } else {
                if (!sp.value.isEmpty()) out.add(sp);
                i++;
            }
        }
        return out;
    }

    private static boolean matchesContains(List<Span> matches, Span sp) {
        for (Span m : matches) {
            if (m.start == sp.start && m.end == sp.end) return true;
        }
        return false;
    }

    public static final class Span {
        public final String value;
        public final int start;
        public final int end;

        public Span(String value, int start, int end) {
            this.value = value;
            this.start = start;
            this.end = end;
        }
    }
}
