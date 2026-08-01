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

package org.bytedeco.pytorch.llm.unsloth.studio.webui;

import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingProgressEvent;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/** Fan-out training events as SSE frames. */
public final class BoardSseHub {

    private final CopyOnWriteArrayList<OutputStream> clients = new CopyOnWriteArrayList<>();

    public void add(OutputStream out) {
        clients.add(out);
    }

    public void remove(OutputStream out) {
        clients.remove(out);
    }

    public void publish(TrainingProgressEvent event) {
        if (event == null) return;
        String data = "event: progress\ndata: " + JsonMaps.stringify(event.toMap()) + "\n\n";
        byte[] bytes = data.getBytes(StandardCharsets.UTF_8);
        for (OutputStream out : List.copyOf(clients)) {
            try {
                out.write(bytes);
                out.flush();
            } catch (IOException e) {
                clients.remove(out);
            }
        }
    }
}
