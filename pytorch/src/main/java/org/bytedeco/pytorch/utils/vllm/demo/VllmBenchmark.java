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
package org.bytedeco.pytorch.utils.vllm.demo;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.utils.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.utils.vllm.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark harness for the vLLM-style engine.
 *
 * <p>Runs two modes:
 * <ul>
 *   <li><b>Tiny offline</b> (default): random-weight Qwen2/Llama model, no network.</li>
 *   <li><b>Real HF model</b>: specify --model Qwen/Qwen2-0.5B-Instruct.</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * # Tiny benchmark (always works offline)
 * java ...VllmBenchmark
 *
 * # Real HF model benchmark
 * java ...VllmBenchmark --model Qwen/Qwen2-0.5B-Instruct --hf-token YOUR_TOKEN
 *
 * # Custom tiny kind
 * java ...VllmBenchmark --kind llama --concurrent 8 --tokens 32
 * }</pre>
 */
public final class VllmBenchmark {

    public static void main(String[] args) throws Exception {
        String modelId = null;
        String hfToken = null;
        String kind = "qwen";
        int concurrent = 4;
        int maxTokens = 16;
        int warmup = 2;
        int rounds = 3;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--model"     -> modelId = args[++i];
                case "--hf-token"  -> hfToken = args[++i];
                case "--kind"      -> kind = args[++i];
                case "--concurrent"-> concurrent = Integer.parseInt(args[++i]);
                case "--tokens"    -> maxTokens = Integer.parseInt(args[++i]);
                case "--warmup"    -> warmup = Integer.parseInt(args[++i]);
                case "--rounds"    -> rounds = Integer.parseInt(args[++i]);
            }
        }

        System.out.println("=== vLLM-style Engine Benchmark ===");
        System.out.println("Model: " + (modelId != null ? modelId : "tiny." + kind));
        System.out.println("Concurrent requests: " + concurrent);
        System.out.println("Max tokens per request: " + maxTokens);
        System.out.println("Warmup rounds: " + warmup + " | Benchmark rounds: " + rounds);
        System.out.println();

        // Build prompts
        String[] promptTexts = {
                "Hello, how are you?",
                "What is 2+2?",
                "Explain gravity in one sentence.",
                "Write a haiku about code.",
                "What is the capital of France?",
                "Summarize transformer attention.",
                "Why is the sky blue?",
                "Tell me a joke."
        };
        List<String> prompts = new ArrayList<>();
        for (int i = 0; i < concurrent; i++) {
            prompts.add(promptTexts[i % promptTexts.length]);
        }

        SamplingParams params = SamplingParams.builder()
                .maxTokens(maxTokens)
                .temperature(0)
                .doSample(false)
                .build();

        EngineConfig ec = EngineConfig.builder()
                .maxNumSeqs(concurrent + 2)
                .maxNumBatchedTokens(concurrent * maxTokens + 256)
                .blockSize(32)
                .maxBlocks(256)
                .device("cpu")
                .build();

        LLM llm;
        if (modelId != null) {
            System.out.println("Loading real HF model: " + modelId + " ...");
            HfHub hub = HfHub.builder()
                    .token(hfToken)
                    .logger(System.out::println)
                    .build();
            try {
                llm = LLM.fromPretrained(modelId, hub, ec);
            } catch (IOException e) {
                System.out.println("WARNING: Could not load model '" + modelId
                        + "' (offline or auth error). Falling back to tiny." + kind);
                System.out.println("  → " + e.getMessage());
                llm = LLM.tiny(kind, ec);
            }
        } else {
            System.out.println("Loading tiny model: " + kind);
            llm = LLM.tiny(kind, ec);
        }

        System.out.println("Engine config: " + llm.config());
        System.out.println("Cache stats (init): " + llm.engine().cache().stats());
        System.out.println();

        // Warmup
        System.out.println("--- Warmup (" + warmup + " rounds) ---");
        for (int w = 0; w < warmup; w++) {
            List<RequestOutput> outs = llm.generate(prompts, params);
            for (RequestOutput o : outs) {
                System.out.println("  warmup[" + w + "] req=" + o.requestId
                        + " tokens=" + o.generatedTokens
                        + " ttft=" + String.format("%.1fms", o.ttftMs));
            }
        }
        System.out.println("Cache stats (after warmup): " + llm.engine().cache().stats());
        System.out.println();

        // Benchmark rounds
        System.out.println("--- Benchmark (" + rounds + " rounds) ---");
        List<Long> roundTimesMs = new ArrayList<>();
        List<Integer> totalTokens = new ArrayList<>();
        List<Double> ttfts = new ArrayList<>();

        for (int r = 0; r < rounds; r++) {
            long start = System.nanoTime();
            List<RequestOutput> outs = llm.generate(prompts, params);
            long elapsed = System.nanoTime() - start;

            int tokens = 0;
            double avgTtft = 0;
            for (RequestOutput o : outs) {
                tokens += o.generatedTokens;
                avgTtft += o.ttftMs;
            }
            avgTtft /= outs.size();

            double ms = elapsed / 1_000_000.0;
            double tps = tokens * 1000.0 / ms;
            System.out.printf("  round[%d] time=%.1fms tokens=%d tps=%.1f avgTTFT=%.1fms%n",
                    r, ms, tokens, tps, avgTtft);

            roundTimesMs.add((long) ms);
            totalTokens.add(tokens);
            ttfts.add(avgTtft);
        }

        // Summary
        long totalMs = roundTimesMs.stream().mapToLong(Long::longValue).sum();
        int totalTok = totalTokens.stream().mapToInt(Integer::intValue).sum();
        double avgTps = totalTok * 1000.0 / totalMs;
        double avgTtft = ttfts.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double p50Lat = roundTimesMs.stream().sorted().skip(roundTimesMs.size() / 2).findFirst().orElse(0L);

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.printf("Total tokens generated: %d%n", totalTok);
        System.out.printf("Overall throughput: %.1f tokens/sec%n", avgTps);
        System.out.printf("Avg TTFT: %.1f ms%n", avgTtft);
        System.out.printf("P50 latency: %.1f ms%n", p50Lat);
        System.out.printf("Cache stats (final): %s%n", llm.engine().cache().stats());
        System.out.printf("Engine metrics: %s%n", llm.metrics());

        // Sanity checks
        System.out.println();
        System.out.println("=== Sanity Checks ===");
        boolean allFinished = true;
        int minTokens = Integer.MAX_VALUE;
        for (int r = 0; r < rounds; r++) {
            if (totalTokens.get(r) == 0) {
                System.out.println("  [FAIL] round " + r + " generated 0 tokens");
                allFinished = false;
            }
            minTokens = Math.min(minTokens, totalTokens.get(r));
        }
        if (allFinished) System.out.println("  [PASS] All rounds produced tokens");
        if (minTokens >= maxTokens * concurrent * 0.5) {
            System.out.println("  [PASS] Token count reasonable (≥50% of max)");
        } else {
            System.out.println("  [WARN] Token count lower than expected");
        }

        // Test chat path
        System.out.println();
        System.out.println("--- Chat path test ---");
        try {
            List<java.util.Map<String, String>> msg = List.of(
                    java.util.Map.of("role", "user", "content", "Say hello in 3 words")
            );
            String reply = llm.chat(msg, SamplingParams.greedy(8));
            System.out.println("  chat reply: " + reply);
            System.out.println("  [PASS] Chat path works");
        } catch (Exception e) {
            System.out.println("  [FAIL] Chat path: " + e.getMessage());
        }

        llm.close();
        System.out.println();
        System.out.println("Benchmark complete.");
    }
}
