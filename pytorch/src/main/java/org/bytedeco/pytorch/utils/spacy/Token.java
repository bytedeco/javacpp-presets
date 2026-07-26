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
package org.bytedeco.pytorch.utils.spacy;

/**
 * A single token in a {@link Doc}.
 */
public interface Token {

    String getText();

    /** Alias of {@link #getText()}. */
    default String text() {
        return getText();
    }

    String getLemma();

    default String lemma_() {
        return getLemma();
    }

    String getPos();

    default String pos_() {
        return getPos();
    }

    String getTag();

    default String tag_() {
        return getTag();
    }

    String getDep();

    default String dep_() {
        return getDep();
    }

    boolean isStop();

    /** Character offset into the parent Doc text. */
    int getIdx();

    default int idx() {
        return getIdx();
    }

    /** Token index within the Doc. */
    int getI();

    default int i() {
        return getI();
    }

    String lower();

    default String lower_() {
        return lower();
    }

    String shape();

    default String shape_() {
        return shape();
    }

    boolean isAlpha();

    boolean isPunct();

    boolean isDigit();

    boolean isSpace();

    boolean likeNum();

    boolean likeEmail();

    boolean likeUrl();

    String entType();

    default String entType_() {
        return entType();
    }

    String entIob();

    default String entIob_() {
        return entIob();
    }

    double[] vector();

    double similarity(Token other);

    /** Whitespace following this token (including empty). */
    String whitespace();

    Doc doc();

    void setPos(String pos);

    void setTag(String tag);

    void setLemma(String lemma);

    void setDep(String dep);

    void setEntType(String entType);

    void setEntIob(String entIob);

    void setStop(boolean stop);
}
