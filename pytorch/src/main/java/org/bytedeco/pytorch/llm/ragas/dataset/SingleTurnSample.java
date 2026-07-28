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
package org.bytedeco.pytorch.llm.ragas.dataset;

import java.util.List;
import java.util.Objects;

/** Single-turn RAG evaluation sample. */
public final class SingleTurnSample {
    private final String userInput;
    private final String response;
    private final String reference;
    private final List<String> retrievedContexts;

    public SingleTurnSample(String userInput, String response, String reference,
                           List<String> retrievedContexts) {
        this.userInput = Objects.requireNonNull(userInput, "userInput");
        this.response = response;
        this.reference = reference;
        this.retrievedContexts = retrievedContexts == null ? List.of() : List.copyOf(retrievedContexts);
    }

    public String userInput() { return userInput; }
    public String response() { return response; }
    public String reference() { return reference; }
    public List<String> retrievedContexts() { return retrievedContexts; }

    public static SingleTurnSample of(String userInput, String response) {
        return new SingleTurnSample(userInput, response, null, null);
    }

    public static SingleTurnSample of(String userInput, String response, String reference) {
        return new SingleTurnSample(userInput, response, reference, null);
    }

    public static SingleTurnSample of(String userInput, String response, String reference,
                                      List<String> retrievedContexts) {
        return new SingleTurnSample(userInput, response, reference, retrievedContexts);
    }
}
