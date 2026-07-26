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
package org.bytedeco.pytorch.nn.modules;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.nn.*;

import org.bytedeco.pytorch.*;

import org.bytedeco.pytorch.utils.spacy.Doc;
import org.bytedeco.pytorch.utils.spacy.Token;
import org.bytedeco.pytorch.utils.spacy.vocab.Lexeme;
import org.bytedeco.pytorch.utils.spacy.vocab.Vocab;

import java.util.Locale;
import java.util.regex.Pattern;

public final class TokenImpl implements Token {

    private static final Pattern EMAIL = Pattern.compile("^[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}$");
    private static final Pattern URL = Pattern.compile("(?i)^(https?://|www\\.).+");
    private static final Pattern NUM = Pattern.compile("^[+-]?\\d+(?:[.,]\\d+)*%?$");

    private final String text;
    private final int idx;
    private final int i;
    private final Vocab vocab;
    private Doc doc;
    private String whitespace = "";
    private String lemma;
    private String pos = "";
    private String tag = "";
    private String dep = "";
    private String entType = "";
    private String entIob = "O";
    private boolean stop;

    public TokenImpl(String text, int idx, int i, Vocab vocab) {
        this.text = text == null ? "" : text;
        this.idx = idx;
        this.i = i;
        this.vocab = vocab == null ? new Vocab() : vocab;
        this.lemma = this.text.toLowerCase(Locale.ROOT);
        this.stop = this.vocab.isStop(this.text);
        this.vocab.get(this.text); // register lexeme
        this.vocab.strings().add(this.text);
    }

    public void setDoc(Doc doc) {
        this.doc = doc;
    }

    public void setWhitespace(String ws) {
        this.whitespace = ws == null ? "" : ws;
    }

    @Override
    public String getText() {
        return text;
    }

    @Override
    public String getLemma() {
        return lemma;
    }

    @Override
    public String getPos() {
        return pos;
    }

    @Override
    public String getTag() {
        return tag;
    }

    @Override
    public String getDep() {
        return dep;
    }

    @Override
    public boolean isStop() {
        return stop;
    }

    @Override
    public int getIdx() {
        return idx;
    }

    @Override
    public int getI() {
        return i;
    }

    @Override
    public String lower() {
        return text.toLowerCase(Locale.ROOT);
    }

    @Override
    public String shape() {
        Lexeme lex = vocab.get(text);
        return lex.shape();
    }

    @Override
    public boolean isAlpha() {
        if (text.isEmpty()) {
            return false;
        }
        return text.chars().allMatch(Character::isLetter);
    }

    @Override
    public boolean isPunct() {
        if (text.isEmpty()) {
            return false;
        }
        return text.chars().allMatch(c -> !Character.isLetterOrDigit(c) && !Character.isWhitespace(c));
    }

    @Override
    public boolean isDigit() {
        if (text.isEmpty()) {
            return false;
        }
        return text.chars().allMatch(Character::isDigit);
    }

    @Override
    public boolean isSpace() {
        if (text.isEmpty()) {
            return false;
        }
        return text.chars().allMatch(Character::isWhitespace);
    }

    @Override
    public boolean likeNum() {
        return NUM.matcher(text).matches();
    }

    @Override
    public boolean likeEmail() {
        return EMAIL.matcher(text).matches();
    }

    @Override
    public boolean likeUrl() {
        return URL.matcher(text).matches();
    }

    @Override
    public String entType() {
        return entType;
    }

    @Override
    public String entIob() {
        return entIob;
    }

    @Override
    public double[] vector() {
        Lexeme lex = vocab.get(text);
        double[] v = lex.getVector();
        return v == null ? new double[0] : v.clone();
    }

    @Override
    public double similarity(Token other) {
        if (other == null) {
            return 0;
        }
        double[] a = vector();
        double[] b = other.vector();
        if (a.length == 0 || b.length == 0 || a.length != b.length) {
            // fallback: orth equality
            return text.equalsIgnoreCase(other.getText()) ? 1.0 : 0.0;
        }
        double dot = 0, na = 0, nb = 0;
        for (int k = 0; k < a.length; k++) {
            dot += a[k] * b[k];
            na += a[k] * a[k];
            nb += b[k] * b[k];
        }
        if (na == 0 || nb == 0) {
            return 0;
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    @Override
    public String whitespace() {
        return whitespace;
    }

    @Override
    public Doc doc() {
        return doc;
    }

    @Override
    public void setPos(String pos) {
        this.pos = pos == null ? "" : pos;
    }

    @Override
    public void setTag(String tag) {
        this.tag = tag == null ? "" : tag;
    }

    @Override
    public void setLemma(String lemma) {
        this.lemma = lemma == null ? "" : lemma;
    }

    @Override
    public void setDep(String dep) {
        this.dep = dep == null ? "" : dep;
    }

    @Override
    public void setEntType(String entType) {
        this.entType = entType == null ? "" : entType;
    }

    @Override
    public void setEntIob(String entIob) {
        this.entIob = entIob == null ? "O" : entIob;
    }

    @Override
    public void setStop(boolean stop) {
        this.stop = stop;
    }

    @Override
    public String toString() {
        return text;
    }
}
