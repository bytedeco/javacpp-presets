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
package org.bytedeco.pytorch.llm.spacy.pipeline;

import org.bytedeco.pytorch.llm.spacy.Doc;
import org.bytedeco.pytorch.llm.spacy.PipelineComponent;
import org.bytedeco.pytorch.llm.spacy.Span;
import org.bytedeco.pytorch.llm.spacy.Token;
import org.bytedeco.pytorch.llm.spacy.impl.SpanImpl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Lite Matcher: register named patterns that are either
 * <ul>
 *   <li>token sequence patterns (list of token texts / {@code "ORTH"} maps), or</li>
 *   <li>substring / regex contains patterns over the Doc text</li>
 * </ul>
 * Matches are stored as entity spans on the Doc when used as a pipeline component.
 */
public final class Matcher implements PipelineComponent {

    public static final class Match {
        public final String matchId;
        public final int start;
        public final int end;
        public final Span span;

        public Match(String matchId, int start, int end, Span span) {
            this.matchId = matchId;
            this.start = start;
            this.end = end;
            this.span = span;
        }

        @Override
        public String toString() {
            return "Match(" + matchId + ", " + start + ", " + end + ", '" + span.getText() + "')";
        }
    }

    private final Map<String, List<Object>> patterns = new LinkedHashMap<>();
    private final boolean addAsEnts;

    public Matcher() {
        this(true);
    }

    public Matcher(boolean addAsEnts) {
        this.addAsEnts = addAsEnts;
    }

    /**
     * Add a token-sequence pattern.
     * Each element is a String (exact orth, case-insensitive) or a Map with keys
     * {@code ORTH}/{@code LOWER}/{@code TEXT}/{@code REGEX}/{@code POS}/{@code IS_PUNCT}/etc.
     */
    public Matcher add(String matchId, List<?> tokenPattern) {
        Objects.requireNonNull(matchId, "matchId");
        patterns.computeIfAbsent(matchId, k -> new ArrayList<>()).add(new ArrayList<>(tokenPattern));
        return this;
    }

    /** Add a simple substring contains pattern over doc text (case-insensitive). */
    public Matcher addContains(String matchId, String substring) {
        Objects.requireNonNull(matchId, "matchId");
        patterns.computeIfAbsent(matchId, k -> new ArrayList<>()).add("contains:" + substring);
        return this;
    }

    /** Add a regex pattern over doc text. */
    public Matcher addRegex(String matchId, String regex) {
        Objects.requireNonNull(matchId, "matchId");
        patterns.computeIfAbsent(matchId, k -> new ArrayList<>()).add(Pattern.compile(regex));
        return this;
    }

    public Matcher remove(String matchId) {
        patterns.remove(matchId);
        return this;
    }

    public List<Match> match(Doc doc) {
        List<Match> out = new ArrayList<>();
        if (doc == null) {
            return out;
        }
        for (Map.Entry<String, List<Object>> e : patterns.entrySet()) {
            String id = e.getKey();
            for (Object pat : e.getValue()) {
                if (pat instanceof Pattern p) {
                    var m = p.matcher(doc.getText());
                    while (m.find()) {
                        Span span = doc.charSpan(m.start(), m.end(), id);
                        out.add(new Match(id, span.getStart(), span.getEnd(), span));
                    }
                } else if (pat instanceof String s && s.startsWith("contains:")) {
                    String sub = s.substring("contains:".length());
                    String text = doc.getText();
                    String lower = text.toLowerCase(Locale.ROOT);
                    String needle = sub.toLowerCase(Locale.ROOT);
                    int from = 0;
                    while (from < lower.length()) {
                        int idx = lower.indexOf(needle, from);
                        if (idx < 0) {
                            break;
                        }
                        Span span = doc.charSpan(idx, idx + sub.length(), id);
                        out.add(new Match(id, span.getStart(), span.getEnd(), span));
                        from = idx + Math.max(1, sub.length());
                    }
                } else if (pat instanceof List<?> tokenPat) {
                    out.addAll(matchTokenPattern(doc, id, tokenPat));
                }
            }
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private List<Match> matchTokenPattern(Doc doc, String id, List<?> pattern) {
        List<Match> out = new ArrayList<>();
        int plen = pattern.size();
        if (plen == 0 || doc.length() < plen) {
            return out;
        }
        for (int i = 0; i + plen <= doc.length(); i++) {
            boolean ok = true;
            for (int j = 0; j < plen; j++) {
                if (!tokenMatches(doc.getToken(i + j), pattern.get(j))) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                Span span = new SpanImpl(doc, i, i + plen, id);
                out.add(new Match(id, i, i + plen, span));
            }
        }
        return out;
    }

    private boolean tokenMatches(Token token, Object spec) {
        if (spec == null) {
            return true;
        }
        if (spec instanceof String s) {
            return token.getText().equalsIgnoreCase(s);
        }
        if (spec instanceof Map<?, ?> map) {
            for (Map.Entry<?, ?> e : map.entrySet()) {
                String key = String.valueOf(e.getKey()).toUpperCase(Locale.ROOT);
                Object val = e.getValue();
                String vs = val == null ? "" : String.valueOf(val);
                switch (key) {
                    case "ORTH", "TEXT" -> {
                        if (!token.getText().equals(vs)) {
                            return false;
                        }
                    }
                    case "LOWER" -> {
                        if (!token.lower().equals(vs.toLowerCase(Locale.ROOT))) {
                            return false;
                        }
                    }
                    case "POS" -> {
                        if (!token.getPos().equalsIgnoreCase(vs)) {
                            return false;
                        }
                    }
                    case "TAG" -> {
                        if (!token.getTag().equalsIgnoreCase(vs)) {
                            return false;
                        }
                    }
                    case "LEMMA" -> {
                        if (!token.getLemma().equalsIgnoreCase(vs)) {
                            return false;
                        }
                    }
                    case "IS_PUNCT" -> {
                        boolean want = Boolean.parseBoolean(vs);
                        if (token.isPunct() != want) {
                            return false;
                        }
                    }
                    case "IS_DIGIT" -> {
                        boolean want = Boolean.parseBoolean(vs);
                        if (token.isDigit() != want) {
                            return false;
                        }
                    }
                    case "IS_ALPHA" -> {
                        boolean want = Boolean.parseBoolean(vs);
                        if (token.isAlpha() != want) {
                            return false;
                        }
                    }
                    case "LIKE_NUM" -> {
                        boolean want = Boolean.parseBoolean(vs);
                        if (token.likeNum() != want) {
                            return false;
                        }
                    }
                    case "REGEX" -> {
                        if (!Pattern.compile(vs).matcher(token.getText()).find()) {
                            return false;
                        }
                    }
                    default -> {
                        // ignore unknown keys
                    }
                }
            }
            return true;
        }
        return false;
    }

    @Override
    public Doc apply(Doc doc) {
        List<Match> matches = match(doc);
        if (addAsEnts && !matches.isEmpty()) {
            List<Span> ents = new ArrayList<>(doc.getEnts());
            for (Match m : matches) {
                ents.add(m.span);
                // set IOB on tokens
                for (int i = m.start; i < m.end && i < doc.length(); i++) {
                    Token t = doc.getToken(i);
                    t.setEntType(m.matchId);
                    t.setEntIob(i == m.start ? "B" : "I");
                }
            }
            doc.setEnts(ents);
        }
        return doc;
    }

    public Map<String, List<Object>> patterns() {
        return Collections.unmodifiableMap(patterns);
    }

    @Override
    public String name() {
        return "matcher";
    }
}
