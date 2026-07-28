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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Training example shell (predicted / reference docs), spaCy {@code Example}-like.
 */
public final class Example {

    private final Doc predicted;
    private final Doc reference;
    private final Map<String, Object> annotations;

    public Example(Doc predicted, Doc reference) {
        this(predicted, reference, Map.of());
    }

    public Example(Doc predicted, Doc reference, Map<String, Object> annotations) {
        this.predicted = predicted;
        this.reference = reference;
        this.annotations = annotations == null ? new HashMap<>() : new HashMap<>(annotations);
    }

    /**
     * Build an example from a Doc and annotation dict.
     * Supported keys: {@code entities} as List of {@code [startChar, endChar, label]}
     * or Map with start/end/label.
     */
    @SuppressWarnings("unchecked")
    public static Example fromDict(Doc doc, Map<String, Object> annots) {
        if (doc == null) {
            throw new IllegalArgumentException("doc is null");
        }
        Map<String, Object> a = annots == null ? Map.of() : annots;
        Doc reference = doc;
        Object entsObj = a.get("entities");
        if (entsObj instanceof List<?> rawEnts) {
            List<Span> ents = new ArrayList<>();
            for (Object item : rawEnts) {
                if (item instanceof List<?> triple && triple.size() >= 2) {
                    int start = ((Number) triple.get(0)).intValue();
                    int end = ((Number) triple.get(1)).intValue();
                    String label = triple.size() >= 3 ? String.valueOf(triple.get(2)) : "";
                    ents.add(doc.charSpan(start, end, label));
                } else if (item instanceof Map<?, ?> m) {
                    int start = ((Number) m.get("start")).intValue();
                    int end = ((Number) m.get("end")).intValue();
                    String label = m.get("label") == null ? "" : String.valueOf(m.get("label"));
                    ents.add(doc.charSpan(start, end, label));
                }
            }
            doc.setEnts(ents);
        }
        return new Example(doc, reference, a);
    }

    public static Example fromText(Language nlp, String text, Map<String, Object> annots) {
        Doc doc = nlp.call(text);
        return fromDict(doc, annots);
    }

    public Doc getPredicted() {
        return predicted;
    }

    public Doc getReference() {
        return reference;
    }

    public Doc predicted() {
        return predicted;
    }

    public Doc reference() {
        return reference;
    }

    public Map<String, Object> annotations() {
        return annotations;
    }

    public String text() {
        return predicted == null ? (reference == null ? "" : reference.getText()) : predicted.getText();
    }

    @Override
    public String toString() {
        return "Example(text='" + text() + "')";
    }
}
