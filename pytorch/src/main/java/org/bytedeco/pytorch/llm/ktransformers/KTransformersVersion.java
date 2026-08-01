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
package org.bytedeco.pytorch.llm.ktransformers;

/**
 * Version constants for the pure-Java KTransformers module.
 *
 * <p>Aligned conceptually with upstream kt-kernel releases (v0.6.x lineage) but
 * versioned independently for the JavaCPP port.
 */
public final class KTransformersVersion {

    /** Module semantic version. */
    public static final String VERSION = "0.6.1-java-01";

    /** Upstream project this port tracks. */
    public static final String UPSTREAM = "https://github.com/kvcache-ai/ktransformers";

    /** Capability tags exposed by this build. */
    public static final String[] CAPABILITIES = {
            "inference",
            "sft",
            "cpu-gpu-expert-schedule",
            "three-tier-prefix-cache",
            "int4-int8-fp8-ref",
            "amx-like-gemm-ref",
            "moe",
            "mla",
            "multi-concurrency",
            "visual-train",
            "finetune-adapter-spi",
            "vllm-hook",
            "llama-factory-bridge"
    };

    private KTransformersVersion() {}

    public static String banner() {
        return "KTransformers-Java " + VERSION + " (upstream: " + UPSTREAM + ")";
    }
}
