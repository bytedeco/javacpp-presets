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

import org.bytedeco.pytorch.llm.llamafactory.FinetuneAdapter;
import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.ktransformers.adapter.KTransformersFinetuneAdapter;
import org.bytedeco.pytorch.llm.ktransformers.adapter.KtFactoryHparamsExtension;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtInferenceEngine;
import org.bytedeco.pytorch.llm.ktransformers.serve.KtMetricsEndpoint;
import org.bytedeco.pytorch.llm.ktransformers.serve.KtOpenAiHandler;
import org.bytedeco.pytorch.llm.ktransformers.sft.KtSftSession;

import java.io.IOException;
import java.util.Map;
import java.util.Objects;

/**
 * Public facade for the pure-Java KTransformers module.
 *
 * <p>Mirrors upstream kt-kernel dual entry points:
 * <ul>
 *   <li>{@link #openInference(KtConfig)} — heterogeneous serving</li>
 *   <li>{@link #openSft(FactoryArgs)} — fine-tuning via {@link FinetuneAdapter}</li>
 * </ul>
 *
 * <pre>{@code
 * try (KtInferenceEngine eng = KTransformers.openInferenceMini()) {
 *     eng.generate(new int[]{1, 2, 3}, 8);
 * }
 * try (FinetuneAdapter job = KTransformers.openSftMini()) {
 *     job.train();
 * }
 * }</pre>
 */
public final class KTransformers {

    private KTransformers() {}

    public static String version() {
        return KTransformersVersion.VERSION;
    }

    public static String banner() {
        return KTransformersVersion.banner();
    }

    public static String[] capabilities() {
        return KTransformersVersion.CAPABILITIES.clone();
    }

    // ── Inference ──────────────────────────────────────────────────────────

    public static KtInferenceEngine openInference(KtConfig config) throws IOException {
        return KtInferenceEngine.open(Objects.requireNonNull(config, "config"));
    }

    public static KtInferenceEngine openInference(Map<String, ?> flat) throws IOException {
        return openInference(KtConfig.fromMap(flat));
    }

    public static KtInferenceEngine openInferenceMini() throws IOException {
        return KtInferenceEngine.openMini();
    }

    // ── SFT ────────────────────────────────────────────────────────────────

    /**
     * Open a host-mesh {@link FinetuneAdapter} from LLaMA-Factory style args
     * (including optional {@code kt_*} keys).
     */
    public static FinetuneAdapter openSft(FactoryArgs args) {
        return KTransformersFinetuneAdapter.open(Objects.requireNonNull(args, "args"));
    }

    public static FinetuneAdapter openSft(Map<String, ?> flat) {
        FactoryArgs args = FactoryArgs.parse(KtFactoryHparamsExtension.mergeIntoFactoryMap(flat));
        return openSft(args);
    }

    public static FinetuneAdapter openSft(KtConfig config) {
        return new KTransformersFinetuneAdapter(
                FactoryArgs.parse(KtFactoryHparamsExtension.toFactoryMap(config)),
                Objects.requireNonNull(config, "config"));
    }

    public static FinetuneAdapter openSftMini() {
        return KTransformersFinetuneAdapter.openMini();
    }

    /** Direct SFT session (not via FinetuneAdapter SPI). */
    public static KtSftSession openSftSession(KtConfig config) {
        return KtSftSession.open(Objects.requireNonNull(config, "config"));
    }

    public static KtSftSession openSftSessionMini() {
        return KtSftSession.openMini();
    }

    // ── Serving helpers ────────────────────────────────────────────────────

    public static KtOpenAiHandler openAiHandler(KtInferenceEngine engine) {
        return new KtOpenAiHandler(Objects.requireNonNull(engine, "engine"));
    }

    public static KtMetricsEndpoint metricsEndpoint(KtInferenceEngine engine) {
        return KtMetricsEndpoint.fromEngine(Objects.requireNonNull(engine, "engine"));
    }

    public static KtMetricsEndpoint metricsEndpoint(KtSftSession session) {
        return KtMetricsEndpoint.fromSession(Objects.requireNonNull(session, "session"));
    }
}
