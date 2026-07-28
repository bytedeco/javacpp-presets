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
package org.bytedeco.pytorch.llm.spacy;

import org.bytedeco.pytorch.llm.spacy.vocab.Vocab;

import java.nio.file.Path;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * A processed document: container of tokens, sentences, entities.
 * Iterable over {@link Token}s.
 */
public interface Doc extends Iterable<Token> {

    String getText();

    /** Number of tokens (not characters). */
    int length();

    /** Character length of the underlying text. */
    int charLength();

    Token getToken(int i);

    /** Token index access alias. */
    default Token get(int i) {
        return getToken(i);
    }

    Span getSlice(int start, int end);

    List<Token> getTokens();

    List<Span> getSents();

    List<Span> getEnts();

    void setEnts(List<Span> ents);

    void setSents(List<Span> sents);

    Iterable<Span> nounChunks();

    boolean hasAnnotation(String attr);

    Map<Integer, Integer> countBy(int attr);

    Map<String, Object> toJson();

    void toDisk(Path path) throws Exception;

    Object toArray(int[] attrs);

    Span charSpan(int startChar, int endChar);

    Span charSpan(int startChar, int endChar, String label);

    double similarity(Doc other);

    Token merge(Span span);

    Retokenizer retokenize();

    boolean isTagged();

    boolean isParsed();

    boolean isNered();

    Vocab vocab();

    Language language();

    @Override
    Iterator<Token> iterator();
}
