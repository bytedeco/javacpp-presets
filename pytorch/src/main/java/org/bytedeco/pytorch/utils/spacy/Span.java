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
import org.bytedeco.pytorch.jit.*;

import java.util.Iterator;
import java.util.List;

/**
 * A slice of a {@link Doc} (sentence, entity, arbitrary span).
 * Iterable over its {@link Token}s.
 */
public interface Span extends Iterable<Token> {

    String getText();

    default String text() {
        return getText();
    }

    /** Token start index (inclusive). */
    int getStart();

    default int start() {
        return getStart();
    }

    /** Token end index (exclusive). */
    int getEnd();

    default int end() {
        return getEnd();
    }

    int getStartChar();

    default int startChar() {
        return getStartChar();
    }

    int getEndChar();

    default int endChar() {
        return getEndChar();
    }

    String label();

    default String label_() {
        return label();
    }

    void setLabel(String label);

    Doc doc();

    int length();

    Token getToken(int i);

    List<Token> getTokens();

    double similarity(Span other);

    @Override
    Iterator<Token> iterator();
}
