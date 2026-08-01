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
package org.bytedeco.pytorch.llm.ktransformers.serve;

import org.bytedeco.pytorch.llm.ktransformers.KTransformersVersion;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtInferenceEngine;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtMetrics;
import org.bytedeco.pytorch.llm.ktransformers.sft.KtSftSession;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Prometheus-ish / JSON metrics endpoint surface (no HTTP stack required).
 *
 * <p>Host meshes expose {@link #snapshot()} as JSON or {@link #prometheusText()}
 * as text/plain. Pulls from inference engine and/or SFT session metrics.
 */
public final class KtMetricsEndpoint {

    private final KtMetrics metrics;
    private final String jobLabel;
    private final Map<String, String> extraLabels;

    public KtMetricsEndpoint(KtMetrics metrics, String jobLabel, Map<String, String> extraLabels) {
        this.metrics = metrics != null ? metrics : new KtMetrics();
        this.jobLabel = jobLabel != null ? jobLabel : "ktransformers";
        this.extraLabels = extraLabels != null
                ? Map.copyOf(extraLabels)
                : Map.of("version", KTransformersVersion.VERSION);
    }

    public static KtMetricsEndpoint fromEngine(KtInferenceEngine engine) {
        Objects.requireNonNull(engine, "engine");
        Map<String, String> labels = new LinkedHashMap<>();
        labels.put("version", KTransformersVersion.VERSION);
        labels.put("mode", "inference");
        String model = engine.config().modelNameOrPath();
        if (model != null) labels.put("model", model);
        return new KtMetricsEndpoint(engine.metrics(), "kt-infer", labels);
    }

    public static KtMetricsEndpoint fromSession(KtSftSession session) {
        Objects.requireNonNull(session, "session");
        Map<String, String> labels = new LinkedHashMap<>();
        labels.put("version", KTransformersVersion.VERSION);
        labels.put("mode", "sft");
        labels.put("stage", session.config().sft().stage().name());
        return new KtMetricsEndpoint(session.monitor().metrics(), "kt-sft", labels);
    }

    public KtMetrics metrics() { return metrics; }

    /** Flat double map for JSON handlers. */
    public Map<String, Double> snapshot() {
        Map<String, Double> m = new LinkedHashMap<>(metrics.snapshot());
        m.put("kt/meta/endpoint", 1.0);
        return m;
    }

    /** Prometheus exposition format (subset). */
    public String prometheusText() {
        StringBuilder sb = new StringBuilder();
        sb.append("# HELP kt_info KTransformers-Java build info\n");
        sb.append("# TYPE kt_info gauge\n");
        sb.append("kt_info{");
        sb.append(labelPairs());
        sb.append("} 1\n");

        Map<String, Double> snap = metrics.snapshot();
        for (Map.Entry<String, Double> e : snap.entrySet()) {
            String name = sanitize(e.getKey());
            sb.append("# TYPE ").append(name).append(" gauge\n");
            sb.append(name).append('{').append(labelPairs()).append("} ");
            sb.append(String.format(Locale.ROOT, "%.8g", e.getValue())).append('\n');
        }
        return sb.toString();
    }

    private String labelPairs() {
        StringBuilder sb = new StringBuilder();
        sb.append("job=\"").append(escape(jobLabel)).append('"');
        for (Map.Entry<String, String> e : extraLabels.entrySet()) {
            sb.append(',').append(e.getKey()).append("=\"").append(escape(e.getValue())).append('"');
        }
        return sb.toString();
    }

    private static String sanitize(String key) {
        // kt/train/loss → kt_train_loss
        String s = key.replace('/', '_').replace('.', '_').replace('-', '_');
        if (!s.startsWith("kt_")) s = "kt_" + s;
        return s;
    }

    private static String escape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }
}
