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

package org.bytedeco.pytorch.llm.unsloth.studio;

import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ExportRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingStartRequest;

import java.nio.file.Path;
import java.util.Objects;

/** Default {@link StudioAdapter} wrapping a single {@link UnslothStudio}. */
public final class DefaultStudioAdapter implements StudioAdapter {

    private final UnslothStudio studio;

    public DefaultStudioAdapter(UnslothStudio studio) {
        this.studio = Objects.requireNonNull(studio);
    }

    @Override
    public StudioOptions options() { return studio.options(); }

    @Override
    public UnslothStudio studio() { return studio; }

    @Override
    public String startTrain(TrainingStartRequest req) {
        return studio.train().start(req);
    }

    @Override
    public void stopTrain(String runId) {
        studio.train().stop(runId);
    }

    @Override
    public Path export(ExportRequest req) throws Exception {
        return studio.export().export(req);
    }

    @Override
    public ChatCompletionResponse chat(ChatCompletionRequest req) throws Exception {
        return studio.inference().chatCompletions(req);
    }

    @Override
    public void close() {
        studio.close();
    }
}
