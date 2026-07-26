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
package org.bytedeco.pytorch.utils.ragas.dataset;

import java.util.List;
import java.util.Objects;

/** Multi-turn RAG evaluation sample (list of user/assistant turns). */
public final class MultiTurnSample {
    public record Turn(String role, String content) {}
    private final List<Turn> messages;

    public MultiTurnSample(List<Turn> messages) {
        this.messages = Objects.requireNonNull(messages, "messages");
    }

    public List<Turn> messages() { return List.copyOf(messages); }
    public static MultiTurnSample of(List<Turn> messages) { return new MultiTurnSample(messages); }
}
