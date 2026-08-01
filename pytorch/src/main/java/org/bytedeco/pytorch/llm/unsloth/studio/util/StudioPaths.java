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
package org.bytedeco.pytorch.llm.unsloth.studio.util;

import org.bytedeco.pytorch.llm.unsloth.studio.StudioOptions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Ensures Studio data directories exist. */
public final class StudioPaths {

    private StudioPaths() {}

    public static void ensureLayout(StudioOptions options) throws IOException {
        mkdirs(options.dataRoot());
        mkdirs(options.cacheRoot());
        mkdirs(options.runsDir());
        mkdirs(options.modelsDir());
        mkdirs(options.datasetsDir());
        mkdirs(options.exportsDir());
        mkdirs(options.recipesDir());
        if (options.tensorBoardSink()) {
            mkdirs(options.tensorBoardLogDir());
        }
    }

    public static Path runDir(StudioOptions options, String runId) {
        return options.runsDir().resolve(runId);
    }

    public static void mkdirs(Path p) throws IOException {
        if (p != null && !Files.exists(p)) {
            Files.createDirectories(p);
        }
    }

    public static Path resolveUnder(Path root, String relative) {
        Path r = root.toAbsolutePath().normalize();
        Path child = r.resolve(relative).normalize();
        if (!child.startsWith(r)) {
            throw new StudioValidationException("path escapes root: " + relative);
        }
        return child;
    }
}
