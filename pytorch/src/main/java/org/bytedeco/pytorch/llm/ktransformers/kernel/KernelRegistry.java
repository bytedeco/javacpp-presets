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
package org.bytedeco.pytorch.llm.ktransformers.kernel;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Process-wide registry of {@link KtKernelBackend} implementations.
 *
 * <p>Default backend is always {@link CpuRefKernelBackend}. Host platforms may
 * register additional backends (e.g. future native AMX wrappers) without changing
 * call sites that resolve by name or capability.
 */
public final class KernelRegistry {

    private static final ConcurrentHashMap<String, KtKernelBackend> BACKENDS = new ConcurrentHashMap<>();
    private static volatile String defaultName = CpuRefKernelBackend.NAME;

    static {
        register(new CpuRefKernelBackend());
    }

    private KernelRegistry() {}

    public static void register(KtKernelBackend backend) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(backend.name(), "backend.name");
        BACKENDS.put(backend.name(), backend);
    }

    public static void unregister(String name) {
        if (name == null) return;
        if (CpuRefKernelBackend.NAME.equals(name)) {
            throw new IllegalArgumentException("cannot unregister default cpu-ref backend");
        }
        BACKENDS.remove(name);
        if (name.equals(defaultName)) {
            defaultName = CpuRefKernelBackend.NAME;
        }
    }

    public static void setDefault(String name) {
        if (!BACKENDS.containsKey(name)) {
            throw new IllegalArgumentException("unknown backend: " + name);
        }
        defaultName = name;
    }

    public static KtKernelBackend defaultBackend() {
        KtKernelBackend b = BACKENDS.get(defaultName);
        if (b == null) {
            b = BACKENDS.get(CpuRefKernelBackend.NAME);
        }
        if (b == null) {
            b = new CpuRefKernelBackend();
            register(b);
        }
        return b;
    }

    public static Optional<KtKernelBackend> get(String name) {
        return Optional.ofNullable(BACKENDS.get(name));
    }

    public static KtKernelBackend require(String name) {
        KtKernelBackend b = BACKENDS.get(name);
        if (b == null) {
            throw new IllegalArgumentException("unknown kernel backend: " + name);
        }
        return b;
    }

    /** First registered backend that claims the capability, else default. */
    public static KtKernelBackend forCapability(KtKernelBackend.Capability cap) {
        for (KtKernelBackend b : BACKENDS.values()) {
            if (b.supports(cap)) {
                return b;
            }
        }
        return defaultBackend();
    }

    public static Map<String, KtKernelBackend> all() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(BACKENDS));
    }

    public static String defaultName() {
        return defaultName;
    }
}
