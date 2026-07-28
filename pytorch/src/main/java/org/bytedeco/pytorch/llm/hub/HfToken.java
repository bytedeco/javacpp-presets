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
package org.bytedeco.pytorch.llm.hub;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Resolve a Hugging Face Hub access token the same way {@code huggingface_hub}
 * does, without requiring the caller to pass it every time.
 *
 * <p>Lookup order:
 * <ol>
 *   <li>explicit argument (if non-blank)</li>
 *   <li>{@code HF_TOKEN} env</li>
 *   <li>{@code HUGGING_FACE_HUB_TOKEN} env</li>
 *   <li>{@code HF_TOKEN_PATH} file (entire contents trimmed)</li>
 *   <li>{@code $HF_HOME/token}</li>
 *   <li>{@code ~/.cache/huggingface/token}</li>
 *   <li>{@code ~/.huggingface/token}</li>
 *   <li>{@code ~/.config/huggingface/token}</li>
 * </ol>
 *
 * <p>Also resolves the preferred Hub endpoint (useful in regions where
 * {@code huggingface.co} is slow/blocked):
 * {@code HF_ENDPOINT} → {@code HF_MIRROR} → default
 * {@code https://huggingface.co}. Common mirror: {@code https://hf-mirror.com}.
 */
public final class HfToken {

    public static final String ENV_TOKEN = "HF_TOKEN";
    public static final String ENV_TOKEN_ALT = "HUGGING_FACE_HUB_TOKEN";
    public static final String ENV_TOKEN_PATH = "HF_TOKEN_PATH";
    public static final String ENV_ENDPOINT = "HF_ENDPOINT";
    public static final String ENV_MIRROR = "HF_MIRROR";
    public static final String DEFAULT_ENDPOINT = "https://huggingface.co";
    public static final String MIRROR_CN = "https://hf-mirror.com";

    private HfToken() {}

    /** Resolve a token, preferring {@code explicit} when non-blank. */
    public static String resolve(String explicit) {
        if (explicit != null && !explicit.isBlank()) {
            return explicit.trim();
        }
        String env = firstNonBlank(System.getenv(ENV_TOKEN), System.getenv(ENV_TOKEN_ALT));
        if (env != null) return env.trim();

        for (Path p : candidateTokenFiles()) {
            try {
                if (Files.isRegularFile(p)) {
                    String body = Files.readString(p, StandardCharsets.UTF_8).trim();
                    // strip optional "Bearer " prefix some users paste in
                    if (body.regionMatches(true, 0, "Bearer ", 0, 7)) {
                        body = body.substring(7).trim();
                    }
                    if (!body.isEmpty()) return body;
                }
            } catch (IOException ignored) {
                // try next candidate
            }
        }
        return null;
    }

    /** Resolve from env / token files only. */
    public static String resolve() {
        return resolve(null);
    }

    public static Optional<String> resolveOptional() {
        return Optional.ofNullable(resolve());
    }

    public static boolean hasToken() {
        return resolve() != null;
    }

    /**
     * Preferred Hub endpoint. Honours {@code HF_ENDPOINT} then {@code HF_MIRROR}.
     * Trailing slash is stripped.
     */
    public static String resolveEndpoint() {
        String ep = firstNonBlank(System.getenv(ENV_ENDPOINT), System.getenv(ENV_MIRROR));
        if (ep == null || ep.isBlank()) {
            // Heuristic: if huggingface.co is unreachable regions often set nothing —
            // still default to official; callers can override. Mirror is opt-in.
            return DEFAULT_ENDPOINT;
        }
        ep = ep.trim();
        while (ep.endsWith("/")) ep = ep.substring(0, ep.length() - 1);
        return ep;
    }

    /**
     * Endpoint with optional forced mirror. When {@code preferMirror} is true and
     * no {@code HF_ENDPOINT}/{@code HF_MIRROR} is set, returns {@link #MIRROR_CN}.
     */
    public static String resolveEndpoint(boolean preferMirror) {
        String env = firstNonBlank(System.getenv(ENV_ENDPOINT), System.getenv(ENV_MIRROR));
        if (env != null && !env.isBlank()) {
            String ep = env.trim();
            while (ep.endsWith("/")) ep = ep.substring(0, ep.length() - 1);
            return ep;
        }
        return preferMirror ? MIRROR_CN : DEFAULT_ENDPOINT;
    }

    /** Build an {@link HfHub} pre-wired with resolved token + endpoint + default cache. */
    public static HfHub defaultHub() {
        return defaultHub(false);
    }

    public static HfHub defaultHub(boolean preferMirror) {
        return HfHub.create()
                .token(resolve())
                .endpoint(resolveEndpoint(preferMirror))
                .logger(s -> {})
                .build();
    }

    public static HfHub defaultHub(String tokenOrNull, String endpointOrNull) {
        String tok = resolve(tokenOrNull);
        String ep = endpointOrNull;
        if (ep == null || ep.isBlank()) ep = resolveEndpoint();
        while (ep.endsWith("/")) ep = ep.substring(0, ep.length() - 1);
        return HfHub.create().token(tok).endpoint(ep).build();
    }

    // ---- internals ---------------------------------------------------------

    static List<Path> candidateTokenFiles() {
        List<Path> out = new ArrayList<>(6);
        String pathEnv = System.getenv(ENV_TOKEN_PATH);
        if (pathEnv != null && !pathEnv.isBlank()) {
            out.add(Path.of(pathEnv.trim()));
        }
        String hfHome = System.getenv(HfCache.DEFAULT_ENV);
        if (hfHome != null && !hfHome.isBlank()) {
            out.add(Path.of(hfHome, "token"));
        }
        String userHome = System.getProperty("user.home");
        if (userHome != null && !userHome.isBlank()) {
            out.add(Path.of(userHome, ".cache", "huggingface", "token"));
            out.add(Path.of(userHome, ".huggingface", "token"));
            out.add(Path.of(userHome, ".config", "huggingface", "token"));
        }
        return out;
    }

    private static String firstNonBlank(String a, String b) {
        if (a != null && !a.isBlank()) return a;
        if (b != null && !b.isBlank()) return b;
        return null;
    }

    /** Mask a token for logs: {@code hf_****abcd}. */
    public static String mask(String token) {
        if (token == null || token.isBlank()) return "(none)";
        String t = token.trim();
        if (t.length() <= 8) return "****";
        return t.substring(0, Math.min(4, t.length())) + "****" + t.substring(t.length() - 4);
    }

    @Override
    public String toString() {
        return "HfToken";
    }

    /** Guard for equality helpers that shouldn't depend on identity. */
    public static boolean equalsToken(String a, String b) {
        return Objects.equals(a == null ? null : a.trim(), b == null ? null : b.trim());
    }
}
