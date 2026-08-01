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

import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

/** Run / request id helpers. */
public final class IdGen {

    private static final AtomicLong SEQ = new AtomicLong();

    private IdGen() {}

    public static String uuid() {
        return UUID.randomUUID().toString().replace("-", "");
    }

    public static String runId() {
        return "run_" + Long.toHexString(System.currentTimeMillis()) + "_" + SEQ.incrementAndGet();
    }

    public static String requestId() {
        return "req_" + uuid().substring(0, 16);
    }

    public static String exportId() {
        return "exp_" + uuid().substring(0, 12);
    }

    public static String recipeJobId() {
        return "recipe_" + uuid().substring(0, 12);
    }
}
