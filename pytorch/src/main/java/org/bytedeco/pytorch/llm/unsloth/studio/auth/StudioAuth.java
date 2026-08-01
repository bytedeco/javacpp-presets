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

package org.bytedeco.pytorch.llm.unsloth.studio.auth;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** Optional local token gate (disabled by default in StudioOptions). */
public final class StudioAuth {
    private final Map<String, String> tokens = new ConcurrentHashMap<>(); // token -> user
    private volatile boolean enabled;

    public StudioAuth(boolean enabled) { this.enabled = enabled; }

    public boolean enabled() { return enabled; }
    public void setEnabled(boolean enabled) { this.enabled = enabled; }

    public String issue(String user) {
        String t = UUID.randomUUID().toString().replace("-", "");
        tokens.put(t, user != null ? user : "local");
        return t;
    }

    public Optional<String> authenticate(String token) {
        if (!enabled) return Optional.of("anonymous");
        if (token == null) return Optional.empty();
        return Optional.ofNullable(tokens.get(token));
    }

    public boolean revoke(String token) {
        return tokens.remove(token) != null;
    }
}
