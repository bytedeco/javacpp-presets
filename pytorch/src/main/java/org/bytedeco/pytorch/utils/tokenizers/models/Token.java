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
package org.bytedeco.pytorch.utils.tokenizers.models;

/**
 * A model token (id + piece string + optional offsets + special flag).
 */
public final class Token {

    private final int id;
    private final String value;
    private final int start;
    private final int end;
    private final boolean special;

    public Token(int id, String value, int start, int end, boolean special) {
        this.id = id;
        this.value = value == null ? "" : value;
        this.start = start;
        this.end = end;
        this.special = special;
    }

    public Token(int id, String value, int start, int end) {
        this(id, value, start, end, false);
    }

    public static Token of(int id, String value) {
        return new Token(id, value, 0, 0, false);
    }

    public static Token special(int id, String value) {
        return new Token(id, value, 0, 0, true);
    }

    public int id() { return id; }
    public String value() { return value; }
    public int start() { return start; }
    public int end() { return end; }
    public boolean special() { return special; }

    public Token withOffsets(int s, int e) {
        return new Token(id, value, s, e, special);
    }

    public Token asSpecial() {
        return special ? this : new Token(id, value, start, end, true);
    }

    @Override
    public String toString() {
        return "Token{id=" + id + ", '" + value + "'" + (special ? ", special" : "") + '}';
    }
}
