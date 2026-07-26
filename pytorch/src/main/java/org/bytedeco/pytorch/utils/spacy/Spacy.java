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
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.utils.spacy.impl.LanguageImpl;

/**
 * Factory utilities similar to spaCy's {@code spacy.load} / {@code blank} / {@code empty}.
 *
 * <pre>{@code
 * Language nlp = Spacy.blank("en");
 * Doc doc = nlp.apply("Hello world!");
 * for (Token t : doc) {
 *     System.out.println(t.getText() + " " + t.i());
 * }
 * }</pre>
 */
public final class Spacy {

    public static final String VERSION = "0.2.0";

    private Spacy() {}

    /**
     * Load a named pipeline. Currently returns a blank model for the language code
     * (no pretrained weights shipped); name is stored as metadata.
     */
    public static Language load(String name) {
        String lang = name == null ? "en" : name;
        // strip model suffixes like en_core_web_sm → en
        int us = lang.indexOf('_');
        if (us > 0) {
            lang = lang.substring(0, us);
        }
        LanguageImpl nlp = new LanguageImpl(lang);
        nlp.setMeta("name", name);
        nlp.addPipe("sentencizer", new org.bytedeco.pytorch.utils.spacy.pipeline.Sentencizer());
        return nlp;
    }

    /** Create a blank Language for the given ISO code (e.g. {@code "en"}). */
    public static Language blank(String lang) {
        return new LanguageImpl(lang == null ? "en" : lang);
    }

    /** Create an empty Language with default English settings. */
    public static Language empty() {
        return new LanguageImpl("xx");
    }

    public static String info() {
        return "org.bytedeco.pytorch.utils.spacy v" + VERSION
                + " (spaCy-like pure Java NLP, JavaCPP PyTorch utils)";
    }

    public static String version() {
        return VERSION;
    }
}
