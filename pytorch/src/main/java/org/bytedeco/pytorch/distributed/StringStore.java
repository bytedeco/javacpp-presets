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
package org.bytedeco.pytorch.distributed;

import org.bytedeco.pytorch.*;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Bidirectional string ↔ id store.
 */
public final class StringStore {

    private final Map<String, Long> strToId = new HashMap<>();
    private final Map<Long, String> idToStr = new HashMap<>();
    private final AtomicLong counter = new AtomicLong(1);

    public long add(String s) {
        if (s == null) {
            s = "";
        }
        String key = s;
        return strToId.computeIfAbsent(key, k -> {
            long id = counter.getAndIncrement();
            idToStr.put(id, k);
            return id;
        });
    }

    public Long getId(String s) {
        return strToId.get(s);
    }

    public String getString(long id) {
        return idToStr.get(id);
    }

    public boolean contains(String s) {
        return strToId.containsKey(s);
    }

    public int size() {
        return strToId.size();
    }

    public Map<String, Long> asMap() {
        return Collections.unmodifiableMap(strToId);
    }
}
