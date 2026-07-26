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
package org.bytedeco.pytorch.utils.tokenizers;

import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code AddedToken} — special / extra tokens beyond the base model vocab.
 */
public final class AddedToken {

    private final int id;
    private final String content;
    private final boolean singleWord;
    private final boolean lstrip;
    private final boolean rstrip;
    private final boolean normalized;
    private final boolean special;

    public AddedToken(int id, String content, boolean singleWord, boolean lstrip,
                      boolean rstrip, boolean normalized, boolean special) {
        this.id = id;
        this.content = Objects.requireNonNull(content, "content");
        this.singleWord = singleWord;
        this.lstrip = lstrip;
        this.rstrip = rstrip;
        this.normalized = normalized;
        this.special = special;
    }

    public static AddedToken of(int id, String content, boolean special) {
        return new AddedToken(id, content, false, false, false, false, special);
    }

    public static AddedToken fromJson(Map<String, Object> m) {
        Objects.requireNonNull(m, "added_token");
        Integer id = JsonMaps.asInt(m.get("id"));
        String content = JsonMaps.asString(m.get("content"));
        if (id == null || content == null) {
            throw new IllegalArgumentException("added_token requires id and content: " + m);
        }
        return new AddedToken(
                id,
                content,
                JsonMaps.asBoolean(m, "single_word", false),
                JsonMaps.asBoolean(m, "lstrip", false),
                JsonMaps.asBoolean(m, "rstrip", false),
                JsonMaps.asBoolean(m, "normalized", true),
                JsonMaps.asBoolean(m, "special", false)
        );
    }

    public int id() { return id; }
    public String content() { return content; }
    public boolean singleWord() { return singleWord; }
    public boolean lstrip() { return lstrip; }
    public boolean rstrip() { return rstrip; }
    public boolean normalized() { return normalized; }
    public boolean special() { return special; }

    @Override
    public String toString() {
        return "AddedToken{id=" + id + ", content='" + content + "', special=" + special + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof AddedToken that)) return false;
        return id == that.id && content.equals(that.content);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, content);
    }
}
