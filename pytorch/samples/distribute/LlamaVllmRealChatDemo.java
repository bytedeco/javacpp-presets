package distribute;

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
 * Real Unsloth Llama-3.2-1B-Instruct dialogue + stress via our vLLM engine.
 *
 * <p>All replies are model-generated (not hardcoded). Loads the local HF snapshot
 * into {@link LLM} and exercises:
 * <ul>
 *   <li>EN / ZH / multi-turn chat ({@link LLM#chat})</li>
 *   <li>Concurrent batch generate stress (TTFT / tokens/sec)</li>
 * </ul>
 *
 * <pre>
 *   javac -cp target/classes:$(cat target/cp.txt) -d target/samples-compile \
 *         samples/LlamaVllmRealChatDemo.java
 *   java  --enable-native-access=ALL-UNNAMED \
 *         -cp target/samples-compile:target/classes:$(cat target/cp.txt) \
 *         distribute.LlamaVllmRealChatDemo \
 *         --dir models/unsloth__Llama-3.2-1B-Instruct \
 *         --tokens 48 --concurrent 2 --rounds 3
 * </pre>
 */
public final class LlamaVllmRealChatDemo {

    static final String[] STRESS_PROMPTS = {
            "Hello, how are you?",
            "What is 2+2? Reply with only the digit.",
            "Explain gravity in one sentence.",
            "Write a short haiku about code.",
            "What is the capital of France?",
            "Why is the sky blue? One sentence.",
            "Tell me a short joke.",
            "Summarize attention in transformers in one sentence."
    };

    static void section(String t) {
        System.out.println();
        System.out.println("======== " + t + " ========");
    }

    static void box(String title, String body) {
        System.out.println("--- " + title + " ---");
        System.out.println(body == null ? "(null)" : body);
        System.out.println("--- end ---");
    }

    static boolean nonEmpty(String s) {
        return s != null && !s.isBlank();
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of("models/unsloth__Llama-3.2-1B-Instruct");
        int maxTokens = 48;
        int concurrent = 2;
        int warmup = 1;
        int rounds = 3;
        String device = "cpu";

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--dir" -> dir = Path.of(args[++i]);
                case "--tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--concurrent" -> concurrent = Integer.parseInt(args[++i]);
                case "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--device" -> device = args[++i];
                case "--help" -> {
                    System.out.println("LlamaVllmRealChatDemo [--dir PATH] [--tokens N] "
                            + "[--concurrent N] [--rounds N] [--device cpu]");
                    return;
                }
            }
        }

        if (!Files.isDirectory(dir) || !Files.isRegularFile(dir.resolve("model.safetensors"))) {
            System.err.println("Missing Llama snapshot at " + dir.toAbsolutePath());
            System.err.println("Need config.json + tokenizer + model.safetensors");
            System.exit(2);
            return;
        }

        System.out.println("=== Unsloth Llama → vLLM Real Dialogue + Stress ===");
        System.out.println("dir        : " + dir.toAbsolutePath());
        System.out.println("tokens     : " + maxTokens);
        System.out.println("concurrent : " + concurrent + "  warmup=" + warmup + "  rounds=" + rounds);
        System.out.println("device     : " + device);
        System.out.println("(all chat replies below are model-generated, not hardcoded)");

        EngineConfig ec = EngineConfig.builder()
                .maxNumSeqs(concurrent + 2)
                .maxNumBatchedTokens(concurrent * maxTokens + 512)
                .blockSize(32)
                .maxBlocks(512)
                .device(device)
                .build();

        long tLoad0 = System.nanoTime();
        System.out.println();
        System.out.println("Loading into vLLM LLM.fromDirectory ...");
        LLM llm = LLM.fromDirectory(dir, ec);
        double loadSec = (System.nanoTime() - tLoad0) / 1e9;
        System.out.printf(Locale.ROOT, "loaded in %.1fs%n", loadSec);
        System.out.println("engine : " + llm.config());
        System.out.println("cache  : " + llm.engine().cache().stats());
        if (llm.bundle() != null && llm.bundle().loadReport() != null) {
            System.out.println("load   : " + llm.bundle().loadReport());
        }
        if (llm.bundle() != null && llm.bundle().config() != null) {
            System.out.println("config : " + llm.bundle().config());
        }
        System.out.println("tok    : backend=" + llm.tokenizer().backend()
                + " vocab≈" + llm.tokenizer().vocabSize());

        SamplingParams greedy = SamplingParams.greedy(maxTokens);
        int pass = 0, fail = 0;
        List<Map<String, String>> rows = new ArrayList<>();

        // ---- real chat dialogues ----
        section("CHAT EN factual");
        String enPrompt = "What is 2+2? Reply with only the digit.";
        box("user", enPrompt);
        long t0 = System.nanoTime();
        String en = llm.chat(List.of(
                Map.of("role", "system", "content", "You are a helpful assistant."),
                Map.of("role", "user", "content", enPrompt)
        ), greedy);
        double enMs = (System.nanoTime() - t0) / 1e6;
        box("assistant (model, " + String.format(Locale.ROOT, "%.0fms", enMs) + ")", en);
        if (nonEmpty(en)) pass++; else fail++;
        rows.add(row("EN", enPrompt, en));

        section("CHAT ZH");
        String zhPrompt = "用一句话介绍杭州";
        box("user", zhPrompt);
        t0 = System.nanoTime();
        String zh = llm.chat(List.of(
                Map.of("role", "user", "content", zhPrompt)
        ), greedy);
        double zhMs = (System.nanoTime() - t0) / 1e6;
        box("assistant (model, " + String.format(Locale.ROOT, "%.0fms", zhMs) + ")", zh);
        if (nonEmpty(zh)) pass++; else fail++;
        rows.add(row("ZH", zhPrompt, zh));

        section("CHAT multi-turn name memory");
        List<Map<String, String>> mt = List.of(
                Map.of("role", "user", "content", "My name is Ada."),
                Map.of("role", "assistant", "content", "Nice to meet you, Ada!"),
                Map.of("role", "user", "content", "What is my name? Reply with only the name.")
        );
        box("messages", mt.toString());
        t0 = System.nanoTime();
        String mtOut = llm.chat(mt, greedy);
        double mtMs = (System.nanoTime() - t0) / 1e6;
        box("assistant (model, " + String.format(Locale.ROOT, "%.0fms", mtMs) + ")", mtOut);
        if (nonEmpty(mtOut)) pass++; else fail++;
        rows.add(row("MT", "What is my name?", mtOut));

        section("CHAT free dialogue");
        String free = "Explain gravity in one simple sentence for a child.";
        box("user", free);
        t0 = System.nanoTime();
        String freeOut = llm.chat(List.of(
                Map.of("role", "user", "content", free)
        ), greedy);
        double freeMs = (System.nanoTime() - t0) / 1e6;
        box("assistant (model, " + String.format(Locale.ROOT, "%.0fms", freeMs) + ")", freeOut);
        if (nonEmpty(freeOut)) pass++; else fail++;
        rows.add(row("FREE", free, freeOut));

        section("CHAT code / creative");
        String code = "Write a 3-line Python hello world with a comment.";
        box("user", code);
        t0 = System.nanoTime();
        String codeOut = llm.chat(List.of(
                Map.of("role", "user", "content", code)
        ), SamplingParams.greedy(Math.max(maxTokens, 64)));
        double codeMs = (System.nanoTime() - t0) / 1e6;
        box("assistant (model, " + String.format(Locale.ROOT, "%.0fms", codeMs) + ")", codeOut);
        if (nonEmpty(codeOut)) pass++; else fail++;
        rows.add(row("CODE", code, codeOut));

        // ---- concurrent stress ----
        section("vLLM concurrent stress c=" + concurrent + " tok=" + maxTokens);
        List<String> prompts = new ArrayList<>();
        for (int i = 0; i < concurrent; i++) {
            prompts.add(STRESS_PROMPTS[i % STRESS_PROMPTS.length]);
        }
        SamplingParams stressParams = SamplingParams.builder()
                .maxTokens(maxTokens)
                .temperature(0)
                .doSample(false)
                .build();

        System.out.println("--- warmup (" + warmup + ") ---");
        for (int w = 0; w < warmup; w++) {
            List<RequestOutput> outs = llm.generate(prompts, stressParams);
            int tok = 0;
            for (RequestOutput o : outs) {
                tok += o.generatedTokens;
                System.out.println("  warmup req=" + o.requestId
                        + " tokens=" + o.generatedTokens
                        + " ttft=" + String.format(Locale.ROOT, "%.1fms", o.ttftMs)
                        + " text=" + clip(o.text(), 80));
            }
            System.out.println("  warmup[" + w + "] total_tokens=" + tok);
        }
        System.out.println("cache after warmup: " + llm.engine().cache().stats());

        System.out.println("--- rounds (" + rounds + ") ---");
        List<Long> roundMs = new ArrayList<>();
        List<Integer> roundTok = new ArrayList<>();
        List<Double> ttfts = new ArrayList<>();
        for (int r = 0; r < rounds; r++) {
            long start = System.nanoTime();
            List<RequestOutput> outs = llm.generate(prompts, stressParams);
            long ms = (System.nanoTime() - start) / 1_000_000L;
            int tokens = 0;
            double avgTtft = 0;
            System.out.println("  round[" + r + "] replies:");
            for (RequestOutput o : outs) {
                tokens += o.generatedTokens;
                avgTtft += o.ttftMs;
                System.out.println("    [" + o.requestId + "] tokens=" + o.generatedTokens
                        + " ttft=" + String.format(Locale.ROOT, "%.1fms", o.ttftMs)
                        + " | " + clip(o.text(), 100));
            }
            avgTtft = outs.isEmpty() ? 0 : avgTtft / outs.size();
            double tps = ms <= 0 ? 0 : tokens * 1000.0 / ms;
            System.out.printf(Locale.ROOT,
                    "  round[%d] SUMMARY time=%dms tokens=%d tps=%.1f avgTTFT=%.1fms%n",
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

        section("STRESS SUMMARY");
        System.out.printf(Locale.ROOT, "total_tokens=%d  avg_tps=%.1f  avg_ttft=%.1fms  p50=%dms%n",
                totalTok, avgTps, avgTtft, p50);
        System.out.println("cache final : " + llm.engine().cache().stats());
        System.out.println("engine metrics: " + llm.metrics());
        if (totalTok > 0) pass++; else fail++;

        section("DIALOGUE TABLE (model text)");
        System.out.printf(Locale.ROOT, "%-6s | %-40s | %s%n", "tag", "prompt", "reply");
        System.out.println("-".repeat(100));
        for (Map<String, String> r : rows) {
            System.out.printf(Locale.ROOT, "%-6s | %-40s | %s%n",
                    clip(r.get("tag"), 6),
                    clip(r.get("prompt"), 40),
                    clip(r.get("reply"), 50));
        }

        section("RESULT");
        System.out.printf(Locale.ROOT, "pass=%d fail=%d load=%.1fs tps=%.1f%n",
                pass, fail, loadSec, avgTps);
        llm.close();
        if (fail > 0 || totalTok == 0) System.exit(1);
    }

    static Map<String, String> row(String tag, String prompt, String reply) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("tag", tag);
        m.put("prompt", prompt);
        m.put("reply", reply == null ? "" : reply);
        return m;
    }

    static String clip(String s, int n) {
        if (s == null) return "";
        String t = s.replace('\n', ' ').trim();
        return t.length() <= n ? t : t.substring(0, n - 1) + "…";
    }
}
