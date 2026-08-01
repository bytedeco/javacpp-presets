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

package org.bytedeco.pytorch.llm.unsloth.studio.model;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

public final class DatasetCard {
    private final String id;
    private final String format;
    private final Path localPath;
    private final long rowCount;
    private final boolean streaming;
    private final Map<String, Object> meta;

    public DatasetCard(String id, String format, Path localPath, long rowCount, boolean streaming, Map<String, Object> meta) {
        this.id = id;
        this.format = format;
        this.localPath = localPath;
        this.rowCount = rowCount;
        this.streaming = streaming;
        this.meta = meta != null ? Map.copyOf(meta) : Map.of();
    }

    public String id() { return id; }
    public String format() { return format; }
    public Optional<Path> localPath() { return Optional.ofNullable(localPath); }
    public long rowCount() { return rowCount; }
    public boolean streaming() { return streaming; }
    public Map<String, Object> meta() { return meta; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("format", format);
        if (localPath != null) m.put("local_path", localPath.toString());
        m.put("row_count", rowCount);
        m.put("streaming", streaming);
        return m;
    }
}
