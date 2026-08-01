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

package org.bytedeco.pytorch.llm.unsloth.studio.data;

import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class FormatDetector {

    public enum Format { ALPACA, SHAREGPT, OPENAI_MESSAGES, PREFERENCE, KTO, CSV, RAW, UNKNOWN }

    private FormatDetector() {}

    public static Format detect(Path path) throws Exception {
        if (path == null || !Files.exists(path)) return Format.UNKNOWN;
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".csv")) return Format.CSV;
        if (name.endsWith(".txt") || name.endsWith(".md")) return Format.RAW;
        String sample = Files.readString(path, StandardCharsets.UTF_8);
        if (sample.length() > 4000) sample = sample.substring(0, 4000);
        sample = sample.trim();
        if (sample.startsWith("[")) {
            try {
                List<Object> arr = JsonMaps.parseArray(sample.endsWith("]") ? sample : sample + "]");
                // broken sample ok
            } catch (Exception ignored) {}
        }
        if (sample.contains("\"instruction\"") && sample.contains("\"output\"")) return Format.ALPACA;
        if (sample.contains("\"conversations\"") || sample.contains("\"from\"") && sample.contains("\"value\"")) {
            return Format.SHAREGPT;
        }
        if (sample.contains("\"messages\"") && sample.contains("\"role\"")) return Format.OPENAI_MESSAGES;
        if (sample.contains("\"chosen\"") && sample.contains("\"rejected\"")) return Format.PREFERENCE;
        if (sample.contains("\"kto_tag\"") || sample.contains("\"desirable\"")) return Format.KTO;
        return Format.UNKNOWN;
    }

    public static Format detectRow(Map<String, Object> row) {
        if (row == null) return Format.UNKNOWN;
        if (row.containsKey("instruction") && row.containsKey("output")) return Format.ALPACA;
        if (row.containsKey("conversations")) return Format.SHAREGPT;
        if (row.containsKey("messages")) return Format.OPENAI_MESSAGES;
        if (row.containsKey("chosen") && row.containsKey("rejected")) return Format.PREFERENCE;
        if (row.containsKey("kto_tag")) return Format.KTO;
        return Format.RAW;
    }
}
