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

package org.bytedeco.pytorch.llm.unsloth.studio.rag;

import java.util.ArrayList;
import java.util.List;

public final class Chunker {
    private final int chunkSize;
    private final int overlap;

    public Chunker(int chunkSize, int overlap) {
        this.chunkSize = Math.max(64, chunkSize);
        this.overlap = Math.max(0, Math.min(overlap, this.chunkSize / 2));
    }

    public static Chunker defaults() { return new Chunker(800, 100); }

    public List<String> chunk(String text) {
        List<String> out = new ArrayList<>();
        if (text == null || text.isBlank()) return out;
        String t = text.trim();
        int i = 0;
        while (i < t.length()) {
            int end = Math.min(t.length(), i + chunkSize);
            out.add(t.substring(i, end));
            if (end >= t.length()) break;
            i = Math.max(i + 1, end - overlap);
        }
        return out;
    }
}
