/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.ragas;

import org.bytedeco.pytorch.utils.ragas.dataset.EvaluationDataset;
import org.bytedeco.pytorch.utils.ragas.dataset.SingleTurnSample;
import org.bytedeco.pytorch.utils.ragas.llms.HeuristicJudge;
import org.bytedeco.pytorch.utils.ragas.llms.LlmJudge;
import org.bytedeco.pytorch.utils.ragas.metrics.AnswerCorrectness;
import org.bytedeco.pytorch.utils.ragas.metrics.AnswerRelevancy;
import org.bytedeco.pytorch.utils.ragas.metrics.AnswerSimilarity;
import org.bytedeco.pytorch.utils.ragas.metrics.ContextPrecision;
import org.bytedeco.pytorch.utils.ragas.metrics.ContextRecall;
import org.bytedeco.pytorch.utils.ragas.metrics.Faithfulness;
import org.bytedeco.pytorch.utils.ragas.metrics.Metric;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * RAG evaluation facade — mirrors Python {@code evaluate()} API.
 *
 * <p>All metrics default to heuristic {@link HeuristicJudge} so CI runs offline.
 * Supply a real {@link LlmJudge} via {@link Builder} for LLM-as-judge mode.
 *
 * <pre>{@code
 * EvaluationDataset ds = EvaluationDataset.of(List.of(
 *     SingleTurnSample.of("What is Java?", "Java is a language."),
 *     SingleTurnSample.of("What is PyTorch?", "PyTorch is a framework.")
 * ));
 * EvaluationResult r = Ragas.evaluate(ds, List.of(
 *     new Faithfulness(), new AnswerRelevancy(), new ContextPrecision()
 * ));
 * System.out.println(r);
 * }</pre>
 */
public final class Ragas {

    public static final String VERSION = "1.0";

    private final LlmJudge defaultJudge;
    private final boolean verbose;

    private Ragas(Builder b) {
        this.defaultJudge = b.defaultJudge != null ? b.defaultJudge : new HeuristicJudge();
        this.verbose = b.verbose;
    }

    public static Builder builder() { return new Builder(); }

    public static EvaluationResult evaluate(EvaluationDataset dataset, List<Metric> metrics) {
        return builder().build().doEvaluate(dataset, metrics);
    }

    public static EvaluationResult evaluate(EvaluationDataset dataset, List<Metric> metrics, LlmJudge judge) {
        return builder().defaultJudge(judge).build().doEvaluate(dataset, metrics);
    }

    public EvaluationResult doEvaluate(EvaluationDataset dataset, List<Metric> metrics) {
        Objects.requireNonNull(dataset, "dataset");
        Objects.requireNonNull(metrics, "metrics");
        Map<String, double[]> result = new LinkedHashMap<>();
        for (Metric m : metrics) {
            result.put(m.name(), computeScores(dataset, m));
        }
        return new EvaluationResult(result);
    }

    private double[] computeScores(EvaluationDataset ds, Metric metric) {
        List<Double> scores = new ArrayList<>();
        for (SingleTurnSample s : ds.samples()) {
            try {
                double sc = metric.score(s, defaultJudge);
                scores.add(Math.max(0, Math.min(1, sc)));
                if (verbose) {
                    System.out.printf("[%s] sample: %s -> %.4f%n",
                            metric.name(), s.userInput().substring(0, Math.min(30, s.userInput().length())), sc);
                }
            } catch (Exception e) {
                scores.add(0.0);
            }
        }
        double[] out = new double[scores.size()];
        for (int i = 0; i < out.length; i++) out[i] = scores.get(i);
        return out;
    }

    public static String version() { return VERSION; }

    public static List<Metric> defaults() {
        return List.of(
                new Faithfulness(),
                new AnswerRelevancy(),
                new ContextPrecision(),
                new ContextRecall(),
                new AnswerCorrectness(),
                new AnswerSimilarity()
        );
    }

    public static EvaluationResult evaluate(EvaluationDataset dataset) {
        return evaluate(dataset, defaults());
    }

    public static final class Builder {
        private LlmJudge defaultJudge;
        private boolean verbose;

        public Builder defaultJudge(LlmJudge j) { this.defaultJudge = j; return this; }
        public Builder verbose(boolean v) { this.verbose = v; return this; }
        public Builder verbose() { this.verbose = true; return this; }
        public Ragas build() { return new Ragas(this); }
    }
}
