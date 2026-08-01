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
package org.bytedeco.pytorch.llm.llamafactory.chat;

import org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter;

/**
 * Minimal chat surface for {@link FinetuneAdapter}.
 *
 * <p>Host meshes (ByteDance / Taobao / Tencent style) call {@link #chat(String)}
 * after train/export. Implementations may wrap a tokenizer + generate engine.
 */
public interface ChatEngine extends AutoCloseable {

    /** Single-turn chat; returns model text (or token-id dump for mini models). */
    String chat(String userMessage);

    /** Multi-turn helper; default concatenates turns naively. */
    default String chat(String system, String userMessage) {
        if (system == null || system.isBlank()) {
            return chat(userMessage);
        }
        return chat(system + "\n" + userMessage);
    }

    @Override
    default void close() {}
}
