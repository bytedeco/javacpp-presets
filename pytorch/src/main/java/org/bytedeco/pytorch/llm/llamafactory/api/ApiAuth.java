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
package org.bytedeco.pytorch.llm.llamafactory.api;

import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

/**
 * Optional Bearer / raw API-key gate for OpenAI-compatible routes.
 *
 * <p>When {@code apiKey} is null/blank, all requests are allowed (local demos).
 * Otherwise accepts {@code Authorization: Bearer <key>} or {@code X-Api-Key: <key>}.
 */
public final class ApiAuth {

    private final String apiKey;

    public ApiAuth(String apiKey) {
        this.apiKey = apiKey == null || apiKey.isBlank() ? null : apiKey.trim();
    }

    public static ApiAuth disabled() {
        return new ApiAuth(null);
    }

    public boolean enabled() {
        return apiKey != null;
    }

    public boolean allow(String authorizationHeader, String xApiKeyHeader) {
        if (apiKey == null) {
            return true;
        }
        if (xApiKeyHeader != null && apiKey.equals(xApiKeyHeader.trim())) {
            return true;
        }
        if (authorizationHeader == null || authorizationHeader.isBlank()) {
            return false;
        }
        String h = authorizationHeader.trim();
        String lower = h.toLowerCase(Locale.ROOT);
        if (lower.startsWith("bearer ")) {
            String token = h.substring(7).trim();
            return apiKey.equals(token);
        }
        // raw key in Authorization
        return apiKey.equals(h);
    }

    public Optional<String> challenge() {
        if (apiKey == null) {
            return Optional.empty();
        }
        return Optional.of("Bearer");
    }

    /** Constant-time-ish compare to avoid trivial timing leaks on short keys. */
    public static boolean slowEquals(String a, String b) {
        if (a == null || b == null) return Objects.equals(a, b);
        byte[] x = a.getBytes(StandardCharsets.UTF_8);
        byte[] y = b.getBytes(StandardCharsets.UTF_8);
        if (x.length != y.length) {
            // still walk to reduce length oracle slightly
            int r = 0;
            for (int i = 0; i < x.length; i++) {
                r |= x[i] ^ (i < y.length ? y[i] : 0);
            }
            return false;
        }
        int r = 0;
        for (int i = 0; i < x.length; i++) {
            r |= x[i] ^ y[i];
        }
        return r == 0;
    }
}
