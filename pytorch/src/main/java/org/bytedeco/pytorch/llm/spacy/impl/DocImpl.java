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
package org.bytedeco.pytorch.llm.spacy.impl;

import org.bytedeco.pytorch.llm.spacy.Doc;
import org.bytedeco.pytorch.llm.spacy.Language;
import org.bytedeco.pytorch.llm.spacy.Retokenizer;
import org.bytedeco.pytorch.llm.spacy.Span;
import org.bytedeco.pytorch.llm.spacy.Token;
import org.bytedeco.pytorch.llm.spacy.vocab.Vocab;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

public final class DocImpl implements Doc {

    private final String text;
    private final List<Token> tokens;
    private final List<Span> sents = new ArrayList<>();
    private final List<Span> ents = new ArrayList<>();
    private final Vocab vocab;
    private final Language language;
    private boolean tagged;
    private boolean parsed;
    private boolean nered;

    public DocImpl(String text, List<? extends Token> tokens, Vocab vocab, Language language) {
        this.text = text == null ? "" : text;
        this.tokens = Collections.unmodifiableList(new ArrayList<>(tokens == null ? List.of() : tokens));
        this.vocab = vocab == null ? new Vocab() : vocab;
        this.language = language;
    }

    public DocImpl(String text, List<? extends Token> tokens) {
        this(text, tokens, new Vocab(), null);
    }

    @Override
    public String getText() {
        return text;
    }

    @Override
    public int length() {
        return tokens.size();
    }

    @Override
    public int charLength() {
        return text.length();
    }

    @Override
    public Token getToken(int i) {
        return tokens.get(i);
    }

    @Override
    public Span getSlice(int start, int end) {
        return new SpanImpl(this, start, end);
    }

    @Override
    public List<Token> getTokens() {
        return tokens;
    }

    @Override
    public List<Span> getSents() {
        return Collections.unmodifiableList(sents);
    }

    @Override
    public List<Span> getEnts() {
        return Collections.unmodifiableList(ents);
    }

    @Override
    public void setEnts(List<Span> ents) {
        this.ents.clear();
        if (ents != null) {
            this.ents.addAll(ents);
        }
        this.nered = !this.ents.isEmpty();
    }

    @Override
    public void setSents(List<Span> sents) {
        this.sents.clear();
        if (sents != null) {
            this.sents.addAll(sents);
        }
    }

    /** Mutable access for pipeline components. */
    public List<Span> sentsMutable() {
        return sents;
    }

    public List<Span> entsMutable() {
        return ents;
    }

    public void setTagged(boolean tagged) {
        this.tagged = tagged;
    }

    public void setParsed(boolean parsed) {
        this.parsed = parsed;
    }

    public void setNered(boolean nered) {
        this.nered = nered;
    }

    @Override
    public Iterable<Span> nounChunks() {
        return List.of();
    }

    @Override
    public boolean hasAnnotation(String attr) {
        if (attr == null) {
            return false;
        }
        return switch (attr.toLowerCase()) {
            case "sent", "sents", "sentence" -> !sents.isEmpty();
            case "ent", "ents", "ner" -> !ents.isEmpty();
            case "pos", "tag" -> tagged;
            case "dep", "parse" -> parsed;
            default -> false;
        };
    }

    @Override
    public Map<Integer, Integer> countBy(int attr) {
        return Map.of();
    }

    @Override
    public Map<String, Object> toJson() {
        Map<String, Object> m = new HashMap<>();
        m.put("text", text);
        List<Map<String, Object>> toks = new ArrayList<>();
        for (Token t : tokens) {
            Map<String, Object> tm = new HashMap<>();
            tm.put("text", t.getText());
            tm.put("i", t.getI());
            tm.put("idx", t.getIdx());
            tm.put("lemma", t.getLemma());
            tm.put("pos", t.getPos());
            tm.put("tag", t.getTag());
            tm.put("dep", t.getDep());
            tm.put("ent_type", t.entType());
            tm.put("ent_iob", t.entIob());
            toks.add(tm);
        }
        m.put("tokens", toks);
        List<Map<String, Object>> sentList = new ArrayList<>();
        for (Span s : sents) {
            Map<String, Object> sm = new HashMap<>();
            sm.put("start", s.getStart());
            sm.put("end", s.getEnd());
            sm.put("text", s.getText());
            sentList.add(sm);
        }
        m.put("sents", sentList);
        List<Map<String, Object>> entList = new ArrayList<>();
        for (Span e : ents) {
            Map<String, Object> em = new HashMap<>();
            em.put("start", e.getStart());
            em.put("end", e.getEnd());
            em.put("label", e.label());
            em.put("text", e.getText());
            entList.add(em);
        }
        m.put("ents", entList);
        return m;
    }

    @Override
    public void toDisk(Path path) throws Exception {
        Files.writeString(path, text);
    }

    @Override
    public Object toArray(int[] attrs) {
        return null;
    }

    @Override
    public Span charSpan(int startChar, int endChar) {
        return charSpan(startChar, endChar, "");
    }

    @Override
    public Span charSpan(int startChar, int endChar, String label) {
        int tokStart = -1;
        int tokEnd = -1;
        for (int i = 0; i < tokens.size(); i++) {
            Token t = tokens.get(i);
            int tStart = t.getIdx();
            int tEnd = tStart + t.getText().length();
            if (tokStart < 0 && tEnd > startChar) {
                tokStart = i;
            }
            if (tStart < endChar) {
                tokEnd = i + 1;
            }
        }
        if (tokStart < 0) {
            tokStart = 0;
            tokEnd = 0;
        }
        return new SpanImpl(this, tokStart, Math.max(tokStart, tokEnd), label);
    }

    @Override
    public double similarity(Doc other) {
        if (other == null || length() == 0 || other.length() == 0) {
            return 0;
        }
        double sum = 0;
        int n = Math.min(length(), other.length());
        for (int i = 0; i < n; i++) {
            sum += getToken(i).similarity(other.getToken(i));
        }
        return sum / n;
    }

    @Override
    public Token merge(Span span) {
        // shell: return first token of span
        if (span == null || span.length() == 0) {
            return null;
        }
        return span.getToken(0);
    }

    @Override
    public Retokenizer retokenize() {
        return new Retokenizer() {
            @Override
            public void merge(Span span) {
                // no-op shell (Doc tokens are immutable list)
            }

            @Override
            public void split(Token token, String[] orths) {
                // no-op shell
            }

            @Override
            public void close() {
            }
        };
    }

    @Override
    public boolean isTagged() {
        return tagged;
    }

    @Override
    public boolean isParsed() {
        return parsed;
    }

    @Override
    public boolean isNered() {
        return nered;
    }

    @Override
    public Vocab vocab() {
        return vocab;
    }

    @Override
    public Language language() {
        return language;
    }

    @Override
    public Iterator<Token> iterator() {
        return new Iterator<>() {
            int cur = 0;

            @Override
            public boolean hasNext() {
                return cur < tokens.size();
            }

            @Override
            public Token next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return tokens.get(cur++);
            }
        };
    }

    @Override
    public String toString() {
        return text;
    }
}
