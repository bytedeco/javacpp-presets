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
import org.bytedeco.pytorch.llm.spacy.Span;
import org.bytedeco.pytorch.llm.spacy.Token;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

public final class SpanImpl implements Span {

    private final Doc doc;
    private final int start; // token index
    private final int end;   // token index exclusive
    private final int startChar;
    private final int endChar;
    private String label;

    public SpanImpl(Doc doc, int start, int end, String label) {
        this.doc = doc;
        this.start = Math.max(0, start);
        this.end = Math.max(this.start, end);
        this.label = label == null ? "" : label;
        if (doc != null && doc.length() > 0 && this.start < doc.length()) {
            this.startChar = doc.getToken(this.start).getIdx();
            if (this.end > 0 && this.end <= doc.length()) {
                Token last = doc.getToken(this.end - 1);
                this.endChar = last.getIdx() + last.getText().length();
            } else if (this.end == 0) {
                this.endChar = this.startChar;
            } else {
                this.endChar = doc.charLength();
            }
        } else if (doc != null) {
            this.startChar = 0;
            this.endChar = 0;
        } else {
            this.startChar = 0;
            this.endChar = 0;
        }
    }

    /** Char-span constructor (token indices resolved later if needed). */
    public SpanImpl(String text, int startChar, int endChar) {
        this.doc = null;
        this.start = -1;
        this.end = -1;
        this.startChar = startChar;
        this.endChar = endChar;
        this.label = "";
        this.textOverride = text == null ? "" : text.substring(
                Math.max(0, startChar), Math.min(text.length(), endChar));
    }

    private String textOverride;

    public SpanImpl(Doc doc, int start, int end) {
        this(doc, start, end, "");
    }

    @Override
    public String getText() {
        if (textOverride != null) {
            return textOverride;
        }
        if (doc == null) {
            return "";
        }
        if (start < 0 || start >= end || start >= doc.length()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < end && i < doc.length(); i++) {
            Token t = doc.getToken(i);
            sb.append(t.getText());
            if (i + 1 < end) {
                sb.append(t.whitespace());
            }
        }
        return sb.toString();
    }

    @Override
    public int getStart() {
        return start;
    }

    @Override
    public int getEnd() {
        return end;
    }

    @Override
    public int getStartChar() {
        return startChar;
    }

    @Override
    public int getEndChar() {
        return endChar;
    }

    @Override
    public String label() {
        return label;
    }

    @Override
    public void setLabel(String label) {
        this.label = label == null ? "" : label;
    }

    @Override
    public Doc doc() {
        return doc;
    }

    @Override
    public int length() {
        return Math.max(0, end - start);
    }

    @Override
    public Token getToken(int i) {
        if (doc == null) {
            throw new IndexOutOfBoundsException("no doc");
        }
        return doc.getToken(start + i);
    }

    @Override
    public List<Token> getTokens() {
        if (doc == null) {
            return List.of();
        }
        List<Token> list = new ArrayList<>(length());
        for (int i = start; i < end && i < doc.length(); i++) {
            list.add(doc.getToken(i));
        }
        return Collections.unmodifiableList(list);
    }

    @Override
    public double similarity(Span other) {
        if (other == null || length() == 0 || other.length() == 0) {
            return 0;
        }
        // average token similarity (O(n*m) lite)
        double sum = 0;
        int n = 0;
        for (Token a : this) {
            for (Token b : other) {
                sum += a.similarity(b);
                n++;
            }
        }
        return n == 0 ? 0 : sum / n;
    }

    @Override
    public Iterator<Token> iterator() {
        return new Iterator<>() {
            int cur = start;

            @Override
            public boolean hasNext() {
                return doc != null && cur < end && cur < doc.length();
            }

            @Override
            public Token next() {
                if (!hasNext()) {
                    throw new NoSuchElementException();
                }
                return doc.getToken(cur++);
            }
        };
    }

    @Override
    public String toString() {
        String t = getText();
        return label == null || label.isEmpty() ? t : t + " (" + label + ")";
    }
}
