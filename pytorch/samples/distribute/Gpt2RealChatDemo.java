package distribute;

import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.pipeline.TextGenerationPipeline;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Real GPT-2 dialogue / completion demo — no hardcoded answers.
 *
 * <p>Loads the local HF snapshot, runs several free-form prompts with
 * both greedy and sampled decoding, and prints the full model output so
 * you can inspect whether generation is actually meaningful.
 *
 * <pre>
 *   javac -cp target/classes:$(cat target/cp.txt) -d target/samples-compile \
 *         samples/Gpt2RealChatDemo.java
 *   java  -cp target/samples-compile:target/classes:$(cat target/cp.txt) \
 *         distribute.Gpt2RealChatDemo \
 *         --dir models/openai-community__gpt2 --tokens 48
 * </pre>
 */
public final class Gpt2RealChatDemo {

    /** Dialogue-style prompts (base GPT-2 has no chat template; we format as plain text). */
    static final String[][] SCENES = {
            {
                    "completion",
                    "The capital of France is"
            },
            {
                    "completion",
                    "Once upon a time in a small village,"
            },
            {
                    "qa-style",
                    "Q: What is 2 + 2?\nA:"
            },
            {
                    "dialogue",
                    "Human: Hello, how are you today?\nAssistant:"
            },
            {
                    "dialogue",
                    "Human: Explain gravity in one simple sentence.\nAssistant:"
            },
            {
                    "dialogue-zh",
                    "用户: 用一句话介绍杭州\n助手:"
            },
            {
                    "multi-turn",
                    "Human: My name is Ada.\nAssistant: Nice to meet you, Ada!\nHuman: What is my name?\nAssistant:"
            },
            {
                    "story",
                    "In the year 2040, robots and humans finally learned to"
            },
    };

    static void banner(String t) {
        System.out.println();
        System.out.println("======== " + t + " ========");
    }

    static void printBox(String title, String body) {
        System.out.println("--- " + title + " ---");
        System.out.println(body == null ? "(null)" : body);
        System.out.println("--- end ---");
    }

    static boolean looksDegenerate(String s) {
        if (s == null || s.isBlank()) return true;
        String t = s.trim();
        if (t.length() < 2) return true;
        // same token repeated many times
        String[] parts = t.split("\\s+");
        if (parts.length >= 6) {
            int same = 0;
            for (int i = 1; i < parts.length; i++) {
                if (parts[i].equals(parts[0])) same++;
            }
            if (same >= parts.length * 0.7) return true;
        }
        // single char / short loop
        if (t.length() >= 12) {
            String unit = t.substring(0, Math.min(8, t.length() / 2));
            int hits = 0;
            for (int i = 0; i + unit.length() <= t.length(); i += unit.length()) {
                if (t.regionMatches(i, unit, 0, unit.length())) hits++;
            }
            if (hits >= 4) return true;
        }
        return false;
    }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of("models/openai-community__gpt2");
        int maxTokens = 48;
        long seed = 42L;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--dir" -> dir = Path.of(args[++i]);
                case "--tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--seed" -> seed = Long.parseLong(args[++i]);
                case "--help" -> {
                    System.out.println("Gpt2RealChatDemo [--dir PATH] [--tokens N] [--seed S]");
                    return;
                }
            }
        }

        if (!Files.isDirectory(dir) || !Files.isRegularFile(dir.resolve("model.safetensors"))) {
            System.err.println("Missing model at " + dir.toAbsolutePath());
            System.err.println("Expected config.json + tokenizer + model.safetensors");
            System.exit(2);
            return;
        }

        System.out.println("=== GPT-2 Real Dialogue / Completion Demo ===");
        System.out.println("dir    : " + dir.toAbsolutePath());
        System.out.println("tokens : " + maxTokens + "  seed=" + seed);
        System.out.println("(outputs below are model-generated, not hardcoded)");

        long t0 = System.nanoTime();
        TextGenerationPipeline pipe = TextGenerationPipeline.fromDirectory(dir);
        AutoModelForCausalLM.Bundle b = pipe.bundle();
        double loadSec = (System.nanoTime() - t0) / 1e9;
        System.out.printf(Locale.ROOT, "loaded in %.1fs%n", loadSec);
        System.out.println("config : " + b.config());
        if (b.loadReport() != null) System.out.println("load   : " + b.loadReport());
        System.out.println("tok    : backend=" + b.tokenizer().backend()
                + " vocab≈" + b.tokenizer().vocabSize());

        GenerationConfig greedy = GenerationConfig.builder()
                .doSample(false)
                .temperature(1.0)
                .maxNewTokens(maxTokens)
                .eosTokenId(b.config().eosTokenId())
                .build();

        GenerationConfig sampled = GenerationConfig.builder()
                .doSample(true)
                .temperature(0.8)
                .topK(50)
                .topP(0.95)
                .maxNewTokens(maxTokens)
                .eosTokenId(b.config().eosTokenId())
                .build();

        int ok = 0, bad = 0, total = 0;
        List<Map<String, String>> rows = new ArrayList<>();

        for (String[] scene : SCENES) {
            String kind = scene[0];
            String prompt = scene[1];
            total++;

            banner(kind + "  #" + total);
            printBox("PROMPT (input to model)", prompt);

            // 1) greedy
            long g0 = System.nanoTime();
            String gOut = pipe.generate(prompt, greedy);
            double gMs = (System.nanoTime() - g0) / 1e6;
            boolean gDeg = looksDegenerate(gOut);
            printBox("GREEDY output  (" + String.format(Locale.ROOT, "%.0fms", gMs)
                    + (gDeg ? ", DEGENERATE?" : ", ok") + ")", gOut);

            // 2) sampled (more natural for base GPT-2)
            long s0 = System.nanoTime();
            String sOut = pipe.generate(prompt, sampled);
            double sMs = (System.nanoTime() - s0) / 1e6;
            boolean sDeg = looksDegenerate(sOut);
            printBox("SAMPLED output (T=0.8 top_p=0.95  "
                    + String.format(Locale.ROOT, "%.0fms", sMs)
                    + (sDeg ? ", DEGENERATE?" : ", ok") + ")", sOut);

            // prefer sampled for quality check; fall back to greedy
            String best = !sDeg ? sOut : gOut;
            boolean good = !looksDegenerate(best) && best != null && best.trim().length() >= 3;
            if (good) ok++;
            else bad++;

            Map<String, String> row = new LinkedHashMap<>();
            row.put("kind", kind);
            row.put("prompt", prompt.replace('\n', ' '));
            row.put("greedy", gOut == null ? "" : gOut.replace('\n', ' '));
            row.put("sampled", sOut == null ? "" : sOut.replace('\n', ' '));
            row.put("verdict", good ? "REAL_OK" : "WEAK");
            rows.add(row);

            System.out.println("verdict: " + (good ? "REAL_OK (non-empty, non-loop)" : "WEAK / degenerate"));
        }

        // multi-turn free chat simulation: feed previous assistant reply back in
        banner("live multi-turn (feed model output back)");
        String history = "Human: Tell me a short fun fact about the ocean.\nAssistant:";
        printBox("turn1 prompt", history);
        String a1 = pipe.generate(history, sampled);
        printBox("turn1 assistant (model)", a1);
        String history2 = history + (a1 == null ? "" : a1.trim())
                + "\nHuman: Why is that interesting?\nAssistant:";
        printBox("turn2 prompt (includes model reply)", history2);
        String a2 = pipe.generate(history2, sampled);
        printBox("turn2 assistant (model)", a2);

        banner("SUMMARY TABLE");
        System.out.printf(Locale.ROOT, "%-12s | %-40s | %-40s | %s%n",
                "kind", "prompt", "sampled (truncated)", "verdict");
        System.out.println("-".repeat(110));
        for (Map<String, String> r : rows) {
            System.out.printf(Locale.ROOT, "%-12s | %-40s | %-40s | %s%n",
                    clip(r.get("kind"), 12),
                    clip(r.get("prompt"), 40),
                    clip(r.get("sampled"), 40),
                    r.get("verdict"));
        }
        System.out.println();
        System.out.printf(Locale.ROOT,
                "real_ok=%d  weak=%d  total=%d%n", ok, bad, total);
        System.out.println("NOTE: base GPT-2 is not instruction-tuned; dialogue quality varies.");
        System.out.println("      Qwen/Llama Instruct models use chat templates for better QA.");
        if (ok == 0) System.exit(1);
    }

    static String clip(String s, int n) {
        if (s == null) return "";
        String t = s.replace('\n', ' ').trim();
        return t.length() <= n ? t : t.substring(0, n - 1) + "…";
    }
}
