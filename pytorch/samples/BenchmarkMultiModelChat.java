package samples;

import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.pipeline.TextGenerationPipeline;
import org.bytedeco.pytorch.llm.vllm.EngineConfig;
import org.bytedeco.pytorch.llm.vllm.LLM;
import org.bytedeco.pytorch.llm.vllm.RequestOutput;
import org.bytedeco.pytorch.llm.vllm.SamplingParams;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Multi-model dialogue + stress benchmark for vLLM-style engine and fine-tune stack.
 *
 * <p>Loads small HF snapshots (config + tokenizer + weights) for Qwen / DeepSeek /
 * Llama / GPT / GLM-edge and runs:
 * <ul>
 *   <li>Config + tokenizer smoke</li>
 *   <li>Single-turn + multi-turn chat (vLLM {@link LLM#chat} / pipeline fallback)</li>
 *   <li>Concurrent batch generate stress (TTFT / tokens/sec)</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>
 *   # Prefer local snapshots under ./models (downloaded via scripts/download_small_models.sh)
 *   javac -cp target/classes:$(cat target/cp.txt) -d target/samples-compile \
 *         samples/BenchmarkMultiModelChat.java
 *   java  -cp target/samples-compile:target/classes:$(cat target/cp.txt) \
 *         samples.BenchmarkMultiModelChat
 *
 *   # Or point at a custom models root / HF token for online pull
 *   java ... samples.BenchmarkMultiModelChat \
 *         --models-dir ./models --hf-token $HF_TOKEN \
 *         --concurrent 4 --tokens 32 --rounds 3
 * </pre>
 */
public final class BenchmarkMultiModelChat {

    static int passed = 0, failed = 0, skipped = 0;
    static final List<String> failures = new ArrayList<>();
    static final List<String> summary = new ArrayList<>();

    /** One candidate model under models/ or HF id. */
    static final class ModelSpec {
        final String name;       // short label
        final String family;     // qwen / deepseek / llama / gpt / glm
        final String localDir;   // models/<dir>
        final String hfId;       // org/name
        final boolean chatModel; // has chat template / instruct
        final boolean nativelySupported; // ModelRegistry has architecture

        ModelSpec(String name, String family, String localDir, String hfId,
                  boolean chatModel, boolean nativelySupported) {
            this.name = name;
            this.family = family;
            this.localDir = localDir;
            this.hfId = hfId;
            this.chatModel = chatModel;
            this.nativelySupported = nativelySupported;
        }
    }

    static final ModelSpec[] SPECS = {
            new ModelSpec("Qwen2.5-0.5B-Instruct", "qwen",
                    "Qwen__Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
                    true, true),
            new ModelSpec("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek",
                    "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    true, true), // Qwen2 architecture
            new ModelSpec("Llama-3.2-1B-Instruct", "llama",
                    "unsloth__Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct",
                    true, true),
            new ModelSpec("gpt2", "gpt",
                    "openai-community__gpt2", "openai-community/gpt2",
                    false, true),
            new ModelSpec("glm-edge-1.5b-chat", "glm",
                    "zai-org__glm-edge-1.5b-chat", "zai-org/glm-edge-1.5b-chat",
                    true, true), // GlmForCausalLM registry + fused SwiGLU
    };

    static final String[] PROMPTS = {
            "Hello, how are you?",
            "What is 2+2? Reply with only the digit.",
            "Explain gravity in one sentence.",
            "Write a haiku about code.",
            "What is the capital of France?",
            "Summarize transformer attention in one sentence.",
            "Why is the sky blue?",
            "Tell me a short joke."
    };

    static void section(String t) {
        System.out.println("\n=== " + t + " ===");
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("  PASS  " + name);
        } else {
            failed++;
            failures.add(name);
            System.out.println("  FAIL  " + name);
        }
    }

    static void skip(String name, String why) {
        skipped++;
        System.out.println("  SKIP  " + name + " (" + why + ")");
    }

    static boolean hasWeights(Path dir) {
        Path st = dir.resolve("model.safetensors");
        if (Files.isRegularFile(st)) {
            try {
                return Files.size(st) > 50_000_000L;
            } catch (Exception e) {
                return false;
            }
        }
        // sharded
        try {
            return Files.list(dir)
                    .anyMatch(p -> {
                        String n = p.getFileName().toString();
                        if (!n.endsWith(".safetensors") && !n.endsWith(".bin")) return false;
                        try {
                            return Files.size(p) > 50_000_000L;
                        } catch (Exception e) {
                            return false;
                        }
                    });
        } catch (Exception e) {
            return false;
        }
    }

    static boolean hasConfigAndTok(Path dir) {
        return Files.isRegularFile(dir.resolve("config.json"))
                && (Files.isRegularFile(dir.resolve("tokenizer.json"))
                || Files.isRegularFile(dir.resolve("tokenizer.model"))
                || Files.isRegularFile(dir.resolve("vocab.json")));
    }

    static Path resolveModelDir(Path modelsRoot, ModelSpec spec) {
        Path local = modelsRoot.resolve(spec.localDir);
        if (Files.isDirectory(local) && hasConfigAndTok(local)) {
            return local;
        }
        return null;
    }

    static LLM loadVllm(Path dir, EngineConfig ec) throws Exception {
        return LLM.fromDirectory(dir, ec);
    }

    static TextGenerationPipeline loadPipeline(Path dir) throws Exception {
        return TextGenerationPipeline.fromDirectory(dir);
    }

    static void runChatSuite(String label, LLM llm, int maxTokens) {
        section(label + " / chat dialogue");
        SamplingParams greedy = SamplingParams.greedy(maxTokens);
        try {
            // single-turn EN
            String en = llm.chat(List.of(
                    Map.of("role", "system", "content", "You are a helpful assistant."),
                    Map.of("role", "user", "content", "What is 2+2? Reply with only the digit.")
            ), greedy);
            System.out.println("  [EN] " + truncate(en, 120));
            check(label + " chat EN non-empty", en != null && !en.isBlank());

            // single-turn ZH
            String zh = llm.chat(List.of(
                    Map.of("role", "user", "content", "用一句话介绍杭州")
            ), greedy);
            System.out.println("  [ZH] " + truncate(zh, 120));
            check(label + " chat ZH non-empty", zh != null && !zh.isBlank());

            // multi-turn
            String mt = llm.chat(List.of(
                    Map.of("role", "user", "content", "My name is Ada."),
                    Map.of("role", "assistant", "content", "Nice to meet you, Ada!"),
                    Map.of("role", "user", "content", "What is my name? Reply with only the name.")
            ), greedy);
            System.out.println("  [MT] " + truncate(mt, 120));
            check(label + " chat multi-turn non-empty", mt != null && !mt.isBlank());
        } catch (Exception e) {
            check(label + " chat suite", false);
            System.out.println("    error: " + e.getMessage());
            e.printStackTrace(System.out);
        }
    }

    static void runPipelineChat(String label, TextGenerationPipeline pipe, int maxTokens) {
        section(label + " / pipeline chat");
        try {
            AutoModelForCausalLM.Bundle b = pipe.bundle();
            GenerationConfig gen = GenerationConfig.builder()
                    .doSample(false)
                    .maxNewTokens(maxTokens)
                    .eosTokenId(b.config().eosTokenId())
                    .build();
            String en = pipe.chat(List.of(
                    Map.of("role", "user", "content", "What is 2+2? Reply with only the digit.")
            ), gen);
            System.out.println("  [pipe EN] " + truncate(en, 120));
            check(label + " pipeline chat non-empty", en != null && !en.isBlank());
        } catch (Exception e) {
            check(label + " pipeline chat", false);
            System.out.println("    error: " + e.getMessage());
        }
    }

    static Map<String, Object> runStress(String label, LLM llm, int concurrent,
                                         int maxTokens, int warmup, int rounds) {
        section(label + " / concurrent stress c=" + concurrent + " tok=" + maxTokens);
        Map<String, Object> metrics = new LinkedHashMap<>();
        List<String> prompts = new ArrayList<>();
        for (int i = 0; i < concurrent; i++) {
            prompts.add(PROMPTS[i % PROMPTS.length]);
        }
        SamplingParams params = SamplingParams.builder()
                .maxTokens(maxTokens)
                .temperature(0)
                .doSample(false)
                .build();

        try {
            System.out.println("--- warmup (" + warmup + ") ---");
            for (int w = 0; w < warmup; w++) {
                List<RequestOutput> outs = llm.generate(prompts, params);
                int tok = 0;
                for (RequestOutput o : outs) tok += o.generatedTokens;
                System.out.printf(Locale.ROOT, "  warmup[%d] tokens=%d%n", w, tok);
            }

            System.out.println("--- rounds (" + rounds + ") ---");
            List<Long> roundMs = new ArrayList<>();
            List<Integer> roundTok = new ArrayList<>();
            List<Double> ttfts = new ArrayList<>();
            for (int r = 0; r < rounds; r++) {
                long t0 = System.nanoTime();
                List<RequestOutput> outs = llm.generate(prompts, params);
                long ms = (System.nanoTime() - t0) / 1_000_000L;
                int tokens = 0;
                double avgTtft = 0;
                for (RequestOutput o : outs) {
                    tokens += o.generatedTokens;
                    avgTtft += o.ttftMs;
                }
                avgTtft = outs.isEmpty() ? 0 : avgTtft / outs.size();
                double tps = ms <= 0 ? 0 : tokens * 1000.0 / ms;
                System.out.printf(Locale.ROOT,
                        "  round[%d] time=%dms tokens=%d tps=%.1f avgTTFT=%.1fms%n",
                        r, ms, tokens, tps, avgTtft);
                roundMs.add(ms);
                roundTok.add(tokens);
                ttfts.add(avgTtft);
            }

            long totalMs = roundMs.stream().mapToLong(Long::longValue).sum();
            int totalTok = roundTok.stream().mapToInt(Integer::intValue).sum();
            double avgTps = totalMs <= 0 ? 0 : totalTok * 1000.0 / totalMs;
            double avgTtft = ttfts.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            List<Long> sorted = new ArrayList<>(roundMs);
            sorted.sort(Long::compareTo);
            long p50 = sorted.isEmpty() ? 0 : sorted.get(sorted.size() / 2);

            metrics.put("total_tokens", totalTok);
            metrics.put("avg_tps", avgTps);
            metrics.put("avg_ttft_ms", avgTtft);
            metrics.put("p50_latency_ms", p50);
            metrics.put("cache", String.valueOf(llm.engine().cache().stats()));
            metrics.put("engine", String.valueOf(llm.metrics()));

            System.out.printf(Locale.ROOT,
                    "  SUMMARY tokens=%d tps=%.1f avgTTFT=%.1fms p50=%dms%n",
                    totalTok, avgTps, avgTtft, p50);
            check(label + " stress produced tokens", totalTok > 0);
            check(label + " stress tps>0", avgTps > 0);
            summary.add(String.format(Locale.ROOT,
                    "%-32s tokens=%5d  tps=%7.1f  ttft=%7.1fms  p50=%5dms",
                    label, totalTok, avgTps, avgTtft, p50));
        } catch (Exception e) {
            check(label + " stress", false);
            System.out.println("    error: " + e.getMessage());
            e.printStackTrace(System.out);
            summary.add(String.format(Locale.ROOT, "%-32s FAILED: %s", label, e.getMessage()));
        }
        return metrics;
    }

    static void runGptGenerate(String label, LLM llm, int maxTokens) {
        section(label + " / plain generate (no chat template)");
        try {
            List<RequestOutput> outs = llm.generate(
                    List.of("The meaning of life is", "Once upon a time"),
                    SamplingParams.greedy(maxTokens));
            for (RequestOutput o : outs) {
                String text = o.outputs.isEmpty() ? "" : o.outputs.get(0).text;
                System.out.println("  gen: " + truncate(text, 100)
                        + "  tokens=" + o.generatedTokens);
            }
            int tok = outs.stream().mapToInt(o -> o.generatedTokens).sum();
            check(label + " generate tokens>0", tok > 0);
        } catch (Exception e) {
            check(label + " generate", false);
            System.out.println("    error: " + e.getMessage());
        }
    }

    /** Pipeline plain generate for models without vLLM runner support (e.g. GPT-2). */
    static void runPipelineGenerate(String label, TextGenerationPipeline pipe, int maxTokens) {
        section(label + " / pipeline generate");
        try {
            AutoModelForCausalLM.Bundle b = pipe.bundle();
            GenerationConfig gen = GenerationConfig.builder()
                    .doSample(false)
                    .maxNewTokens(maxTokens)
                    .eosTokenId(b.config().eosTokenId())
                    .build();
            String[] prompts = {
                    "The meaning of life is",
                    "Once upon a time"
            };
            int nonEmpty = 0;
            for (String p : prompts) {
                String out = pipe.generate(p, gen);
                System.out.println("  gen: " + truncate(out, 100));
                if (out != null && !out.isBlank()) nonEmpty++;
            }
            check(label + " pipeline generate non-empty", nonEmpty > 0);
        } catch (Exception e) {
            check(label + " pipeline generate", false);
            System.out.println("    error: " + e.getMessage());
        }
    }

    /** Timed sequential generate via pipeline (fallback stress when vLLM runner N/A). */
    static void runPipelineStress(String label, TextGenerationPipeline pipe,
                                  int concurrent, int maxTokens, int warmup, int rounds) {
        section(label + " / pipeline stress c=" + concurrent + " tok=" + maxTokens);
        try {
            AutoModelForCausalLM.Bundle b = pipe.bundle();
            GenerationConfig gen = GenerationConfig.builder()
                    .doSample(false)
                    .maxNewTokens(maxTokens)
                    .eosTokenId(b.config().eosTokenId())
                    .build();
            List<String> prompts = new ArrayList<>();
            for (int i = 0; i < concurrent; i++) {
                prompts.add(PROMPTS[i % PROMPTS.length]);
            }

            System.out.println("--- warmup (" + warmup + ") ---");
            for (int w = 0; w < warmup; w++) {
                for (String p : prompts) pipe.generate(p, gen);
                System.out.printf(Locale.ROOT, "  warmup[%d] done%n", w);
            }

            System.out.println("--- rounds (" + rounds + ") ---");
            List<Long> roundMs = new ArrayList<>();
            int totalOutChars = 0;
            for (int r = 0; r < rounds; r++) {
                long t0 = System.nanoTime();
                int chars = 0;
                for (String p : prompts) {
                    String out = pipe.generate(p, gen);
                    if (out != null) chars += out.length();
                }
                long ms = (System.nanoTime() - t0) / 1_000_000L;
                // approximate tokens ≈ chars/4 for latin; report both
                int approxTok = Math.max(1, chars / 4);
                double tps = ms <= 0 ? 0 : approxTok * 1000.0 / ms;
                System.out.printf(Locale.ROOT,
                        "  round[%d] time=%dms approxTok≈%d tps≈%.1f chars=%d%n",
                        r, ms, approxTok, tps, chars);
                roundMs.add(ms);
                totalOutChars += chars;
            }
            long totalMs = roundMs.stream().mapToLong(Long::longValue).sum();
            int approxTok = Math.max(1, totalOutChars / 4);
            double avgTps = totalMs <= 0 ? 0 : approxTok * 1000.0 / totalMs;
            List<Long> sorted = new ArrayList<>(roundMs);
            sorted.sort(Long::compareTo);
            long p50 = sorted.isEmpty() ? 0 : sorted.get(sorted.size() / 2);
            System.out.printf(Locale.ROOT,
                    "  SUMMARY approxTok≈%d tps≈%.1f p50=%dms (pipeline sequential)%n",
                    approxTok, avgTps, p50);
            check(label + " pipeline stress produced output", totalOutChars > 0);
            summary.add(String.format(Locale.ROOT,
                    "%-32s approxTok≈%5d  tps≈%6.1f  p50=%5dms  (pipeline)",
                    label, approxTok, avgTps, p50));
        } catch (Exception e) {
            check(label + " pipeline stress", false);
            System.out.println("    error: " + e.getMessage());
            summary.add(String.format(Locale.ROOT, "%-32s pipeline FAILED: %s",
                    label, e.getMessage()));
        }
    }

    static String truncate(String s, int n) {
        if (s == null) return "null";
        String t = s.replace('\n', ' ').trim();
        return t.length() <= n ? t : t.substring(0, n) + "…";
    }

    public static void main(String[] args) throws Exception {
        Path modelsDir = Path.of("models");
        String hfToken = System.getenv("HF_TOKEN");
        int concurrent = 4;
        int maxTokens = 32;
        int warmup = 1;
        int rounds = 3;
        boolean online = false;
        String only = null;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--models-dir" -> modelsDir = Path.of(args[++i]);
                case "--hf-token" -> hfToken = args[++i];
                case "--concurrent" -> concurrent = Integer.parseInt(args[++i]);
                case "--tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--online" -> online = true;
                case "--only" -> only = args[++i];
                case "--help" -> {
                    System.out.println("BenchmarkMultiModelChat [--models-dir DIR] [--hf-token T] "
                            + "[--concurrent N] [--tokens N] [--rounds N] [--online] [--only name]");
                    return;
                }
            }
        }

        System.out.println("=== Multi-Model Chat + Stress Benchmark ===");
        System.out.println("models-dir : " + modelsDir.toAbsolutePath());
        System.out.println("concurrent : " + concurrent + "  tokens=" + maxTokens
                + "  warmup=" + warmup + "  rounds=" + rounds);
        System.out.println("online     : " + online);
        System.out.println();

        EngineConfig ec = EngineConfig.builder()
                .maxNumSeqs(concurrent + 2)
                .maxNumBatchedTokens(concurrent * maxTokens + 256)
                .blockSize(32)
                .maxBlocks(256)
                .device("cpu")
                .build();

        HfHub hub = null;
        if (online) {
            hub = HfHub.builder()
                    .token(hfToken)
                    .logger(System.out::println)
                    .build();
        }

        for (ModelSpec spec : SPECS) {
            if (only != null && !spec.name.toLowerCase(Locale.ROOT).contains(only.toLowerCase(Locale.ROOT))
                    && !spec.family.equalsIgnoreCase(only)) {
                continue;
            }

            section("MODEL " + spec.name + " (" + spec.family + ")");
            Path dir = resolveModelDir(modelsDir, spec);
            boolean weights = dir != null && hasWeights(dir);
            boolean cfgTok = dir != null && hasConfigAndTok(dir);

            System.out.println("  local : " + (dir == null ? "(missing)" : dir));
            System.out.println("  config+tok: " + cfgTok + "  weights: " + weights
                    + "  native: " + spec.nativelySupported);

            if (!cfgTok && !online) {
                skip(spec.name + " load", "no local snapshot; re-run download script or --online");
                continue;
            }
            if (!spec.nativelySupported) {
                // Still verify config/tokenizer present for future GLM support.
                check(spec.name + " config+tokenizer present", cfgTok);
                if (cfgTok) {
                    try {
                        String cfg = Files.readString(dir.resolve("config.json"));
                        System.out.println("  config head: "
                                + truncate(cfg.replace('\n', ' '), 160));
                        check(spec.name + " config has model_type",
                                cfg.contains("model_type") || cfg.contains("architectures"));
                    } catch (Exception e) {
                        check(spec.name + " read config", false);
                    }
                }
                skip(spec.name + " inference",
                        "architecture not in ModelRegistry yet (glm) — config/tok only");
                summary.add(String.format(Locale.ROOT,
                        "%-32s CONFIG/TOK only (no native runner)", spec.name));
                continue;
            }
            if (!weights && !online) {
                skip(spec.name + " inference",
                        "weights missing (config/tok ok) — finish download");
                check(spec.name + " config+tokenizer present", cfgTok);
                summary.add(String.format(Locale.ROOT,
                        "%-32s weights missing", spec.name));
                continue;
            }

            LLM llm = null;
            try {
                long tLoad0 = System.nanoTime();
                if (weights) {
                    System.out.println("  loading from directory ...");
                    llm = loadVllm(dir, ec);
                } else {
                    System.out.println("  loading from HF " + spec.hfId + " ...");
                    llm = LLM.fromPretrained(spec.hfId, hub, ec);
                }
                double loadSec = (System.nanoTime() - tLoad0) / 1e9;
                System.out.printf(Locale.ROOT, "  loaded in %.1fs  engine=%s%n",
                        loadSec, llm.config());
                check(spec.name + " load ok", true);

                // tokenizer smoke
                try {
                    int vocab = llm.tokenizer().vocabSize();
                    System.out.println("  tokenizer backend=" + llm.tokenizer().backend()
                            + " vocab≈" + vocab);
                    check(spec.name + " tokenizer vocab>0", vocab > 0);
                } catch (Exception e) {
                    check(spec.name + " tokenizer", false);
                }

                // vLLM CausalLmRunner currently supports Qwen2 + Llama only.
                // GPT-2 (and other CausalLM families) exercise the transformers
                // pipeline path + generate for dialogue/smoke; stress still tries
                // vLLM and falls back to pipeline timed generate on UOE.
                boolean vllmNative = "qwen".equals(spec.family)
                        || "deepseek".equals(spec.family)
                        || "llama".equals(spec.family)
                        || "glm".equals(spec.family);

                if (spec.chatModel && vllmNative) {
                    runChatSuite(spec.name, llm, maxTokens);
                    if (dir != null && weights) {
                        try {
                            TextGenerationPipeline pipe = loadPipeline(dir);
                            runPipelineChat(spec.name, pipe, Math.min(maxTokens, 24));
                        } catch (Exception e) {
                            System.out.println("  pipeline path: " + e.getMessage());
                        }
                    }
                    runStress(spec.name, llm, concurrent, maxTokens, warmup, rounds);
                } else if (dir != null && weights) {
                    // GPT-2 / non-vLLM-native: pipeline generate + timed stress
                    try {
                        TextGenerationPipeline pipe = loadPipeline(dir);
                        runPipelineGenerate(spec.name, pipe, maxTokens);
                        runPipelineStress(spec.name, pipe, concurrent, maxTokens, warmup, rounds);
                    } catch (Exception e) {
                        check(spec.name + " pipeline path", false);
                        System.out.println("    error: " + e.getMessage());
                        // last-ditch: try vLLM generate (may UOE)
                        runGptGenerate(spec.name, llm, maxTokens);
                    }
                } else {
                    runGptGenerate(spec.name, llm, maxTokens);
                    runStress(spec.name, llm, concurrent, maxTokens, warmup, rounds);
                }
            } catch (Exception e) {
                check(spec.name + " load/run", false);
                System.out.println("  ERROR: " + e.getMessage());
                e.printStackTrace(System.out);
                summary.add(String.format(Locale.ROOT, "%-32s ERROR: %s",
                        spec.name, e.getMessage()));
            } finally {
                if (llm != null) {
                    try {
                        llm.close();
                    } catch (Exception ignore) {
                    }
                }
            }
        }

        section("FINAL SUMMARY");
        for (String line : summary) {
            System.out.println("  " + line);
        }
        System.out.println();
        System.out.printf(Locale.ROOT, "passed=%d failed=%d skipped=%d%n",
                passed, failed, skipped);
        if (!failures.isEmpty()) {
            System.out.println("failures:");
            for (String f : failures) System.out.println("  - " + f);
        }
        if (failed > 0) System.exit(1);
    }
}
