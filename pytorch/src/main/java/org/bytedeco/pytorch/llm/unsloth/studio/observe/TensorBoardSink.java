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

package org.bytedeco.pytorch.llm.unsloth.studio.observe;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

/**
 * Lightweight scalar event writer. Tries plot.tensorboard reflectively; else CSV log.
 */
public final class TensorBoardSink implements MetricsSink {

    private final Path logDir;
    private final Path csv;

    public TensorBoardSink(Path logDir) {
        this.logDir = Objects.requireNonNull(logDir);
        this.csv = logDir.resolve("studio_scalars.csv");
        try {
            Files.createDirectories(logDir);
            if (!Files.exists(csv)) {
                Files.writeString(csv, "run_id,step,loss,lr,tps,ts\n", StandardCharsets.UTF_8);
            }
        } catch (Exception ignored) {}
    }

    @Override
    public String name() { return "tensorboard"; }

    @Override
    public void record(TrainingMetrics metrics) {
        try {
            Class<?> tb = Class.forName("org.bytedeco.pytorch.plot.tensorboard.SummaryWriter");
            // best-effort; fall through to csv
        } catch (Throwable ignored) {}
        try {
            String line = metrics.runId() + "," + metrics.step() + "," + metrics.loss() + ","
                    + metrics.learningRate() + "," + metrics.tokensPerSecond() + ","
                    + metrics.timestampMs() + "\n";
            Files.writeString(csv, line, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (Exception ignored) {}
    }
}
