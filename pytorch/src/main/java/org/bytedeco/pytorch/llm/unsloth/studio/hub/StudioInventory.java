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

package org.bytedeco.pytorch.llm.unsloth.studio.hub;

import org.bytedeco.pytorch.llm.unsloth.studio.model.DatasetCard;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ModelCard;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Scans local models / datasets directories for board & MCP listing. */
public final class StudioInventory {

    private final Path modelsDir;
    private final Path datasetsDir;
    private final StudioModelRegistry registry;

    public StudioInventory(Path modelsDir, Path datasetsDir, StudioModelRegistry registry) {
        this.modelsDir = modelsDir;
        this.datasetsDir = datasetsDir;
        this.registry = registry;
    }

    public List<ModelCard> models() {
        return registry.listLocal();
    }

    public List<DatasetCard> datasets() {
        List<DatasetCard> out = new ArrayList<>();
        // built-in demo
        out.add(new DatasetCard("alpaca_demo", "alpaca", null, 100, false, java.util.Map.of("builtin", true)));
        out.add(new DatasetCard("sharegpt_demo", "sharegpt", null, 50, false, java.util.Map.of("builtin", true)));
        if (datasetsDir != null && Files.isDirectory(datasetsDir)) {
            try (var stream = Files.list(datasetsDir)) {
                stream.forEach(p -> {
                    String name = p.getFileName().toString();
                    String fmt = name.endsWith(".csv") ? "csv"
                            : name.endsWith(".jsonl") || name.endsWith(".json") ? "jsonl"
                            : name.endsWith(".parquet") ? "parquet" : "raw";
                    long rows = 0;
                    try {
                        if (Files.isRegularFile(p) && (fmt.equals("csv") || fmt.equals("jsonl"))) {
                            rows = Files.lines(p).count();
                        }
                    } catch (Exception ignored) {}
                    out.add(new DatasetCard(name, fmt, p, rows, false, java.util.Map.of()));
                });
            } catch (Exception ignored) {}
        }
        return out;
    }
}
