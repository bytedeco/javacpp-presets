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
package org.bytedeco.pytorch.llm.tokenizers.pretokenizers;

/**
 * A pre-tokenized span with optional character offsets into the original (or normalized) text.
 */
public final class PreToken {

    private final String value;
    private final int start;
    private final int end;
    /** When true this pretok is an added/special token and must bypass the model. */
    private final boolean added;
    private final int addedId;

    public PreToken(String value, int start, int end) {
        this(value, start, end, false, -1);
    }

    public PreToken(String value, int start, int end, boolean added, int addedId) {
        this.value = value == null ? "" : value;
        this.start = start;
        this.end = end;
        this.added = added;
        this.addedId = addedId;
    }

    public static PreToken of(String value) {
        return new PreToken(value, 0, value == null ? 0 : value.length());
    }

    public static PreToken added(String value, int start, int end, int id) {
        return new PreToken(value, start, end, true, id);
    }

    public String value() { return value; }
    public int start() { return start; }
    public int end() { return end; }
    public boolean added() { return added; }
    public int addedId() { return addedId; }

    public PreToken withValue(String newValue) {
        return new PreToken(newValue, start, end, added, addedId);
    }

    @Override
    public String toString() {
        return "PreToken{'" + value + "', " + start + ":" + end + (added ? ", added=" + addedId : "") + '}';
    }
}
