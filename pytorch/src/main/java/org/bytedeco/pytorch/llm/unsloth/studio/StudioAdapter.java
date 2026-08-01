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
import org.bytedeco.pytorch.llm.unsloth.studio.webui.BoardState;

import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;

/**
 * SPI for host LLM platforms (ByteDance / Taobao / Tencent style meshes).
 *
 * <p>Depend on this interface — not on peft/trl/unsloth internals — so Studio
 * can evolve without breaking outer systems. Mirrors {@code factory.FinetuneAdapter}
 * style for dual-stack hosts.
 */
public interface StudioAdapter extends AutoCloseable {

    StudioOptions options();

    UnslothStudio studio();

    String startTrain(TrainingStartRequest req);

    void stopTrain(String runId);

    default void awaitTrain(String runId) throws Exception {
        studio().train().await(runId);
    }

    Path export(ExportRequest req) throws Exception;

    ChatCompletionResponse chat(ChatCompletionRequest req) throws Exception;

    default Map<String, Double> lastMetrics(String runId) {
        return studio().train().run(runId)
                .map(r -> r.lastMetrics())
                .orElse(Collections.emptyMap());
    }

    default BoardState board() {
        return studio().board().state();
    }

    @Override
    void close();
}
