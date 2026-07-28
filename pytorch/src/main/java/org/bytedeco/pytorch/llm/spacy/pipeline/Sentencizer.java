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
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Rule-based sentence segmenter: splits on {@code . ! ?} (and fullwidth variants)
 * when followed by whitespace / end / capital letter.
 */
public final class Sentencizer implements PipelineComponent {

    private final Set<String> punct;

    public Sentencizer() {
        this(Set.of(".", "!", "?", "。", "！", "？", "…"));
    }

    public Sentencizer(Set<String> sentenceEndPunct) {
        this.punct = new HashSet<>(sentenceEndPunct == null ? Set.of() : sentenceEndPunct);
    }

    @Override
    public Doc apply(Doc doc) {
        if (doc == null || doc.length() == 0) {
            return doc;
        }
        List<Span> sents = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < doc.length(); i++) {
            Token t = doc.getToken(i);
            boolean isEndPunct = punct.contains(t.getText());
            boolean endOfDoc = i == doc.length() - 1;
            boolean nextIsBoundary = false;
            if (isEndPunct && !endOfDoc) {
                Token next = doc.getToken(i + 1);
                String ws = t.whitespace();
                // boundary if whitespace after punct, or next token starts uppercase
                nextIsBoundary = (ws != null && !ws.isEmpty())
                        || (!next.getText().isEmpty() && Character.isUpperCase(next.getText().charAt(0)))
                        || next.isPunct();
            }
            if ((isEndPunct && (endOfDoc || nextIsBoundary)) || endOfDoc) {
                int end = i + 1;
                if (end > start) {
                    sents.add(new SpanImpl(doc, start, end, "SENT"));
                }
                start = end;
            }
        }
        if (start < doc.length()) {
            sents.add(new SpanImpl(doc, start, doc.length(), "SENT"));
        }
        doc.setSents(sents);
        return doc;
    }

    @Override
    public String name() {
        return "sentencizer";
    }
}
