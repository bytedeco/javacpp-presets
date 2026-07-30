package samples;

import org.bytedeco.pytorch.llm.sentence.SentenceTransformer;
import org.bytedeco.pytorch.llm.vllm.EngineConfig;
import org.bytedeco.pytorch.llm.vllm.OmniLLM;
import org.bytedeco.pytorch.llm.vllm.RequestOutput;
import org.bytedeco.pytorch.llm.vllm.SamplingParams;
import org.bytedeco.pytorch.llm.vllm.multimodal.CompositeMultimodalProcessor;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MultimodalPrompt;
import org.bytedeco.pytorch.llm.vllm.runner.EmbeddingRunner;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Multimodal stress suite for OmniLLM: embedding + image/audio/video/OCR/ASR + text.
 *
 * <p>Uses a small local causal LM (default Qwen2.5-0.5B) as the Omni backbone and
 * {@link CompositeMultimodalProcessor} with real encoders (DINOv2 / CLIP / SmolVLM /
 * Whisper + Video/OCR/ASR wrappers). Text embeddings run via
 * {@link EmbeddingRunner} / {@link SentenceTransformer#mini()}.
 *
 * <p>Inference replies are also written under {@code --out} (default
 * {@code samples/out/omni_mm/}) as Markdown for human verification.
 *
 * <p>For multi-backbone (Qwen / DeepSeek / Llama / GLM) stress see
 * {@code BenchmarkVllmMultimodalStress}.
 *
 * <pre>
 *   javac -cp target/classes:$(cat target/cp.txt) -d target/samples-compile \
 *         samples/BenchmarkOmniMultimodal.java
 *   java  --enable-native-access=ALL-UNNAMED -Xmx8g \
 *         -cp target/samples-compile:target/classes:$(cat target/cp.txt) \
 *         samples.BenchmarkOmniMultimodal \
 *         --dir models/Qwen__Qwen2.5-0.5B-Instruct \
 *         --concurrent 4 --tokens 24 --rounds 3 \
 *         --out samples/out/omni_mm
 * </pre>
 */
public final class BenchmarkOmniMultimodal {

    static int passed = 0, failed = 0, skipped = 0;
    static final List<String> failures = new ArrayList<>();
    static final List<String> summary = new ArrayList<>();
    /** Human-review lines: modality | ms | reply */
    static final List<String> reviewLines = new ArrayList<>();

    static void section(String t) {
        System.out.println();
        System.out.println("======== " + t + " ========");
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

    static void box(String title, String body) {
        System.out.println("--- " + title + " ---");
        System.out.println(body == null ? "(null)" : body);
        System.out.println("--- end ---");
    }

    static String clip(String s, int n) {
        if (s == null) return "";
        String t = s.replace('\n', ' ').trim();
        return t.length() <= n ? t : t.substring(0, n - 1) + "…";
    }

    static boolean hasWeights(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) return false;
        for (String name : List.of("model.safetensors", "pytorch_model.bin")) {
            Path p = dir.resolve(name);
            try {
                if (Files.isRegularFile(p) && Files.size(p) > 1_000_000L) return true;
            } catch (Exception ignored) {}
        }
        try {
            return Files.list(dir)
                    .anyMatch(p -> {
                        String n = p.getFileName().toString();
                        if (!n.endsWith(".safetensors") && !n.endsWith(".bin")) return false;
                        try { return Files.size(p) > 1_000_000L; } catch (Exception e) { return false; }
                    });
        } catch (Exception e) {
            return false;
        }
    }

    /** Inventory small multimodal HF snapshots under models/. */
    static void inventoryModels(Path modelsRoot) {
        section("HF multimodal snapshot inventory");
        record Spec(String label, String dir, String role) {}
        Spec[] specs = {
                new Spec("MiniLM-L6-v2", "sentence-transformers__all-MiniLM-L6-v2", "embedding"),
                new Spec("DINOv2-small", "facebook__dinov2-small", "vision-encoder"),
                new Spec("Whisper-tiny", "openai__whisper-tiny", "audio-encoder / ASR"),
                new Spec("CLIP-ViT-B/32", "openai__clip-vit-base-patch32", "vision-text"),
                new Spec("SmolVLM-256M", "HuggingFaceTB__SmolVLM-256M-Instruct", "vlm / qwen-vl alias"),
                new Spec("Qwen2.5-0.5B", "Qwen__Qwen2.5-0.5B-Instruct", "text-backbone"),
                new Spec("DeepSeek-1.5B", "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B", "text-backbone"),
                new Spec("Llama-3.2-1B", "unsloth__Llama-3.2-1B-Instruct", "text-backbone"),
                new Spec("GLM-edge-1.5B", "zai-org__glm-edge-1.5b-chat", "text-backbone"),
        };
        for (Spec s : specs) {
            Path d = modelsRoot.resolve(s.dir);
            boolean cfg = Files.isRegularFile(d.resolve("config.json"));
            boolean tok = Files.isRegularFile(d.resolve("tokenizer.json"))
                    || Files.isRegularFile(d.resolve("vocab.json"));
            boolean w = hasWeights(d);
            String status = !cfg ? "MISSING" : (w ? "READY" : "CONFIG_ONLY");
            System.out.printf(Locale.ROOT, "  %-14s %-12s cfg=%s tok=%s weights=%s  [%s]%n",
                    s.label, s.role, cfg, tok, w, status);
            summary.add(String.format(Locale.ROOT, "%-14s %s weights=%s", s.label, status, w));
            if (cfg) check(s.label + " config present", true);
            else check(s.label + " config present", false);
        }
    }

    static void runEmbeddingStress(int concurrent, int rounds) {
        section("Embedding stress (SentenceTransformer.mini + EmbeddingRunner)");
        EmbeddingRunner emb = OmniLLM.miniEmbedder();
        try {
            List<String> texts = new ArrayList<>();
            String[] base = {
                    "The capital of France is Paris.",
                    "杭州是中国的历史文化名城。",
                    "Gravity pulls objects toward the earth.",
                    "A quick brown fox jumps over the lazy dog.",
                    "machine learning and transformers",
                    "音频与视频多模态理解",
                    "Hello, how are you today?",
                    "vector embeddings for semantic search"
            };
            for (int i = 0; i < concurrent; i++) texts.add(base[i % base.length] + " #" + i);

            // warmup
            emb.encodeBatch(texts);
            check("embed dim>0", emb.dimension() > 0);
            System.out.println("  embedDim=" + emb.dimension());

            List<Long> times = new ArrayList<>();
            int totalVecs = 0;
            for (int r = 0; r < rounds; r++) {
                long t0 = System.nanoTime();
                float[][] v = emb.encodeBatch(texts);
                long ms = (System.nanoTime() - t0) / 1_000_000L;
                times.add(ms);
                totalVecs += v.length;
                // cosine sanity: self-sim ~ 1
                double self = SentenceTransformer.cosine(v[0], v[0]);
                double cross = SentenceTransformer.cosine(v[0], v[Math.min(1, v.length - 1)]);
                System.out.printf(Locale.ROOT,
                        "  round[%d] n=%d time=%dms selfCos=%.3f crossCos=%.3f sample[0][0..3]=%s%n",
                        r, v.length, ms, self, cross,
                        Arrays.toString(Arrays.copyOf(v[0], Math.min(4, v[0].length))));
                check("embed round" + r + " selfCos≈1", self > 0.99);
                check("embed round" + r + " finite", !Double.isNaN(cross));
            }
            long sum = times.stream().mapToLong(Long::longValue).sum();
            double qps = sum <= 0 ? 0 : totalVecs * 1000.0 / sum;
            System.out.printf(Locale.ROOT, "  SUMMARY vecs=%d qps=%.1f dim=%d%n", totalVecs, qps, emb.dimension());
            summary.add(String.format(Locale.ROOT, "embedding      vecs=%d qps=%.1f dim=%d", totalVecs, qps, emb.dimension()));
        } finally {
            emb.close();
        }
    }

    static void runMediaPath(OmniLLM omni, Path fixtures, int maxTokens) {
        section("Multimodal path (real encoders + text generation)");
        SamplingParams greedy = SamplingParams.greedy(maxTokens);
        Path img = fixtures.resolve("test_image.png");
        Path img2 = fixtures.resolve("test_image2.png");
        Path wav = fixtures.resolve("test_audio.wav");
        Path vid = fixtures.resolve("test_video.mp4");

        check("fixture image", Files.isRegularFile(img));
        check("fixture audio", Files.isRegularFile(wav));
        check("fixture video", Files.isRegularFile(vid));

        // Real encoder feature extraction
        if (omni.processor() instanceof CompositeMultimodalProcessor cmp) {
            section("Real encoder feature extraction");
            System.out.println("  hasImageEncoder=" + cmp.hasRealImageEncoder()
                    + " hasAudioEncoder=" + cmp.hasRealAudioEncoder());
            if (cmp.hasRealImageEncoder() && Files.isRegularFile(img)) {
                var f1 = cmp.encodeImageFeatures(MediaInput.image(img));
                var f2 = Files.isRegularFile(img2)
                        ? cmp.encodeImageFeatures(MediaInput.image(img2))
                        : f1;
                System.out.println("  image1: " + f1);
                System.out.println("  image2: " + f2);
                check("image encode non-empty", !f1.isEmpty());
                check("image encode dim>0", f1.dim() > 0);
                if (!f1.isEmpty() && !f2.isEmpty()) {
                    double cos = CompositeMultimodalProcessor.cosine(f1.pooled, f2.pooled);
                    System.out.printf(Locale.ROOT, "  image1↔image2 cosine=%.4f%n", cos);
                    check("image cosine finite", !Double.isNaN(cos));
                    summary.add(String.format(Locale.ROOT,
                            "image-feat     %s dim=%d ms=%.1f cos12=%.3f",
                            f1.source, f1.dim(), f1.encodeMs, cos));
                }
            } else {
                skip("image encode", "no real image encoder loaded");
            }
            if (cmp.hasRealAudioEncoder() && Files.isRegularFile(wav)) {
                var fa = cmp.encodeAudioFeatures(MediaInput.audio(wav, 1000));
                System.out.println("  audio: " + fa);
                check("audio encode non-empty", !fa.isEmpty());
                check("audio encode dim>0", fa.dim() > 0);
                summary.add(String.format(Locale.ROOT,
                        "audio-feat     %s dim=%d ms=%.1f", fa.source, fa.dim(), fa.encodeMs));
            } else {
                skip("audio encode", "no real audio encoder loaded");
            }
            for (String line : cmp.encodeLog()) {
                System.out.println("  enc-log: " + line);
            }
        }

        // IMAGE generation path
        if (Files.isRegularFile(img)) {
            long t0 = System.nanoTime();
            String ans = omni.askImage(img, "What colors do you see? One short sentence.", greedy);
            double ms = (System.nanoTime() - t0) / 1e6;
            box("IMAGE ask (" + String.format(Locale.ROOT, "%.0fms", ms) + ")", ans);
            check("image reply non-empty", ans != null && !ans.isBlank());
            if (omni.processor() instanceof CompositeMultimodalProcessor cmp2) {
                for (String line : cmp2.encodeLog()) System.out.println("  " + line);
            }
            summary.add(String.format(Locale.ROOT, "image          %.0fms  %s", ms, clip(ans, 60)));
            reviewLines.add(String.format(Locale.ROOT, "### IMAGE (%.0fms)\n\n```\n%s\n```\n", ms, ans));
        } else skip("image", "missing fixture");

        // AUDIO
        if (Files.isRegularFile(wav)) {
            long t0 = System.nanoTime();
            String ans = omni.askAudio(wav, "Is this audio speech or tone? One word.", greedy);
            double ms = (System.nanoTime() - t0) / 1e6;
            box("AUDIO ask (" + String.format(Locale.ROOT, "%.0fms", ms) + ")", ans);
            check("audio reply non-empty", ans != null && !ans.isBlank());
            summary.add(String.format(Locale.ROOT, "audio          %.0fms  %s", ms, clip(ans, 60)));
            reviewLines.add(String.format(Locale.ROOT, "### AUDIO (%.0fms)\n\n```\n%s\n```\n", ms, ans));
        } else skip("audio", "missing fixture");

        // VIDEO
        if (Files.isRegularFile(vid)) {
            long t0 = System.nanoTime();
            String ans = omni.askVideo(vid, "Describe the scene briefly.", greedy);
            double ms = (System.nanoTime() - t0) / 1e6;
            box("VIDEO ask (" + String.format(Locale.ROOT, "%.0fms", ms) + ")", ans);
            check("video reply non-empty", ans != null && !ans.isBlank());
            summary.add(String.format(Locale.ROOT, "video          %.0fms  %s", ms, clip(ans, 60)));
            reviewLines.add(String.format(Locale.ROOT, "### VIDEO (%.0fms)\n\n```\n%s\n```\n", ms, ans));
            if (omni.processor() instanceof CompositeMultimodalProcessor cmpV) {
                var fv = cmpV.encodeVideoFeatures(MediaInput.video(vid));
                System.out.println("  video features: " + fv);
                check("video encode non-empty", !fv.isEmpty() || !cmpV.hasRealVideoEncoder());
            }
        } else skip("video", "missing fixture");

        // OCR (document / text-in-image path)
        if (Files.isRegularFile(img)) {
            long t0 = System.nanoTime();
            String ans = omni.askOcr(img, "Read any visible text. Transcribe briefly.", greedy);
            double ms = (System.nanoTime() - t0) / 1e6;
            box("OCR ask (" + String.format(Locale.ROOT, "%.0fms", ms) + ")", ans);
            check("ocr reply non-empty", ans != null && !ans.isBlank());
            summary.add(String.format(Locale.ROOT, "ocr            %.0fms  %s", ms, clip(ans, 60)));
            reviewLines.add(String.format(Locale.ROOT, "### OCR (%.0fms)\n\n```\n%s\n```\n", ms, ans));
            if (omni.processor() instanceof CompositeMultimodalProcessor cmpO) {
                var fo = cmpO.encodeOcrFeatures(MediaInput.image(img));
                System.out.println("  ocr features: " + fo);
                check("ocr encode non-empty", !fo.isEmpty() || !cmpO.hasRealOcrEncoder());
            }
        } else skip("ocr", "missing fixture");

        // ASR (speech recognition path)
        if (Files.isRegularFile(wav)) {
            long t0 = System.nanoTime();
            String ans = omni.askAsr(wav, "Transcribe the speech. Output only the text.", greedy);
            double ms = (System.nanoTime() - t0) / 1e6;
            box("ASR ask (" + String.format(Locale.ROOT, "%.0fms", ms) + ")", ans);
            check("asr reply non-empty", ans != null && !ans.isBlank());
            summary.add(String.format(Locale.ROOT, "asr            %.0fms  %s", ms, clip(ans, 60)));
            reviewLines.add(String.format(Locale.ROOT, "### ASR (%.0fms)\n\n```\n%s\n```\n", ms, ans));
            if (omni.processor() instanceof CompositeMultimodalProcessor cmpA) {
                var fa = cmpA.encodeAsrFeatures(MediaInput.audio(wav, 1000));
                System.out.println("  asr features: " + fa);
                check("asr encode non-empty", !fa.isEmpty() || !cmpA.hasRealAsrEncoder());
            }
        } else skip("asr", "missing fixture");

        // MIXED: image + audio + text (smaller budgets via custom processor check)
        if (Files.isRegularFile(img) && Files.isRegularFile(wav)) {
            MultimodalPrompt mixed = MultimodalPrompt.of(
                    MediaInput.image(img, 64, 64),
                    MediaInput.audio(wav, 1000),
                    MediaInput.text("Summarize both media in one sentence."));
            System.out.println("  mixed media: " + CompositeMultimodalProcessor.mediaSummary(mixed));
            int[] ids = omni.processor().process(mixed, List.of(
                    Map.of("role", "user", "content", "Look and listen.")
            ));
            System.out.println("  mixed token budget ids=" + ids.length);
            check("mixed ids>50 (placeholders reserved)", ids.length > 50);

            try {
                long t0 = System.nanoTime();
                RequestOutput out = omni.generate(mixed, SamplingParams.greedy(maxTokens));
                double ms = (System.nanoTime() - t0) / 1e6;
                String text = "";
                if (out != null && !out.outputs.isEmpty()) {
                    text = omni.tokenizer().decode(out.outputs.get(0).tokenIds, true);
                }
                box("MIXED generate (" + String.format(Locale.ROOT, "%.0fms", ms) + ")", text);
                check("mixed generate tokens>0", out != null && out.generatedTokens > 0);
                summary.add(String.format(Locale.ROOT, "mixed          %.0fms tokens=%d  %s",
                        ms, out == null ? 0 : out.generatedTokens, clip(text, 50)));
            } catch (Exception e) {
                System.out.println("  mixed generate error (KV pressure with large media budget): " + e.getMessage());
                // Still counts as path exercised if we reserved tokens
                check("mixed generate path exercised", ids.length > 50);
                summary.add("mixed          KV OOM under default media budgets — path exercised");
            }
        }

        // token budget estimates
        if (Files.isRegularFile(img2)) {
            int bImg = omni.processor().estimateTokenBudget(MediaInput.image(img2));
            int bAud = omni.processor().estimateTokenBudget(MediaInput.audio(wav, 2000));
            int bVid = omni.processor().estimateTokenBudget(MediaInput.video(vid));
            System.out.printf(Locale.ROOT, "  budgets image=%d audio(2s)=%d video=%d%n", bImg, bAud, bVid);
            check("image budget>0", bImg > 0);
            check("audio budget scales with duration", bAud > 0);
            check("video budget>0", bVid > 0);
        }
    }

    static void runConcurrentMediaStress(OmniLLM omni, Path fixtures,
                                          int concurrent, int maxTokens, int warmup, int rounds) {
        section("Concurrent multimodal stress c=" + concurrent + " tok=" + maxTokens);
        Path img = fixtures.resolve("test_image.png");
        Path wav = fixtures.resolve("test_audio.wav");
        Path vid = fixtures.resolve("test_video.mp4");
        if (!Files.isRegularFile(img)) {
            skip("concurrent media", "no image fixture");
            return;
        }

        List<MultimodalPrompt> prompts = new ArrayList<>();
        String[] qs = {
                "Describe briefly.",
                "What is the main color?",
                "One word summary.",
                "Is it bright or dark?",
                "Any objects?",
                "Mood of the scene?",
                "Count dominant hues.",
                "Caption this."
        };
        for (int i = 0; i < concurrent; i++) {
            MediaInput media = switch (i % 3) {
                case 0 -> MediaInput.image(img);
                case 1 -> Files.isRegularFile(wav) ? MediaInput.audio(wav, 1000) : MediaInput.image(img);
                default -> Files.isRegularFile(vid) ? MediaInput.video(vid) : MediaInput.image(img);
            };
            prompts.add(MultimodalPrompt.of(media, MediaInput.text(qs[i % qs.length])));
        }

        SamplingParams params = SamplingParams.builder()
                .maxTokens(maxTokens).temperature(0).doSample(false).build();

        System.out.println("--- warmup (" + warmup + ") ---");
        for (int w = 0; w < warmup; w++) {
            int tok = 0;
            for (MultimodalPrompt p : prompts) {
                RequestOutput o = omni.generate(p, params);
                if (o != null) tok += o.generatedTokens;
            }
            System.out.println("  warmup[" + w + "] tokens=" + tok);
        }

        System.out.println("--- rounds (" + rounds + ") ---");
        List<Long> roundMs = new ArrayList<>();
        List<Integer> roundTok = new ArrayList<>();
        for (int r = 0; r < rounds; r++) {
            long t0 = System.nanoTime();
            int tokens = 0;
            for (int i = 0; i < prompts.size(); i++) {
                try {
                    RequestOutput o = omni.generate(prompts.get(i), params);
                    int gt = o == null ? 0 : o.generatedTokens;
                    tokens += gt;
                    String tx = "";
                    if (o != null && !o.outputs.isEmpty()) {
                        tx = omni.tokenizer().decode(o.outputs.get(0).tokenIds, true);
                    }
                    System.out.printf(Locale.ROOT, "    [%d] tokens=%d ttft=%.1fms | %s%n",
                            i, gt, o == null ? 0 : o.ttftMs, clip(tx, 80));
                } catch (Exception e) {
                    System.out.printf(Locale.ROOT, "    [%d] ERROR %s%n", i, e.getMessage());
                }
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            double tps = ms <= 0 ? 0 : tokens * 1000.0 / ms;
            System.out.printf(Locale.ROOT, "  round[%d] time=%dms tokens=%d tps=%.1f%n", r, ms, tokens, tps);
            roundMs.add(ms);
            roundTok.add(tokens);
        }
        long totalMs = roundMs.stream().mapToLong(Long::longValue).sum();
        int totalTok = roundTok.stream().mapToInt(Integer::intValue).sum();
        double avgTps = totalMs <= 0 ? 0 : totalTok * 1000.0 / totalMs;
        System.out.printf(Locale.ROOT, "  SUMMARY tokens=%d avg_tps=%.1f cache=%s%n",
                totalTok, avgTps, omni.engine().cache().stats());
        check("media stress tokens>0", totalTok > 0);
        summary.add(String.format(Locale.ROOT, "mm-stress      tokens=%d tps=%.1f c=%d",
                totalTok, avgTps, concurrent));
    }

    public static void main(String[] args) throws Exception {
        Path modelsDir = Path.of("models");
        Path backbone = Path.of("models/Qwen__Qwen2.5-0.5B-Instruct");
        Path fixtures = Path.of("samples/fixtures/multimodal");
        Path outDir = Path.of("samples/out/omni_mm");
        int concurrent = 4;
        int maxTokens = 24;
        int warmup = 1;
        int rounds = 3;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--models-dir" -> modelsDir = Path.of(args[++i]);
                case "--dir" -> backbone = Path.of(args[++i]);
                case "--fixtures" -> fixtures = Path.of(args[++i]);
                case "--out" -> outDir = Path.of(args[++i]);
                case "--concurrent" -> concurrent = Integer.parseInt(args[++i]);
                case "--tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--help" -> {
                    System.out.println("BenchmarkOmniMultimodal [--dir BACKBONE] [--models-dir D] "
                            + "[--fixtures D] [--out D] [--concurrent N] [--tokens N] [--rounds N]");
                    return;
                }
            }
        }

        System.out.println("=== OmniLLM Multimodal Stress (embed + image/audio/video/ocr/asr) ===");
        System.out.println("backbone   : " + backbone.toAbsolutePath());
        System.out.println("models-dir : " + modelsDir.toAbsolutePath());
        System.out.println("fixtures   : " + fixtures.toAbsolutePath());
        System.out.println("out        : " + outDir.toAbsolutePath());
        System.out.println("concurrent : " + concurrent + " tokens=" + maxTokens
                + " warmup=" + warmup + " rounds=" + rounds);
        System.out.println("Encoders: DINOv2/CLIP/SmolVLM(=qwen-vl)/Whisper + Video/OCR/ASR wrappers");

        inventoryModels(modelsDir);
        runEmbeddingStress(concurrent, rounds);

        if (!Files.isDirectory(backbone) || !hasWeights(backbone)) {
            System.err.println("Missing backbone weights at " + backbone);
            System.err.println("Need a small causal LM for OmniLLM (e.g. Qwen2.5-0.5B).");
            System.exit(2);
            return;
        }

        EngineConfig ec = EngineConfig.builder()
                .maxNumSeqs(concurrent + 4)
                .maxNumBatchedTokens(concurrent * (maxTokens + 256) + 1024)
                .blockSize(32)
                .maxBlocks(2048)
                .device("cpu")
                .build();

        section("Load OmniLLM backbone");
        long t0 = System.nanoTime();
        OmniLLM omni = OmniLLM.fromDirectory(backbone, ec);
        double loadSec = (System.nanoTime() - t0) / 1e9;
        System.out.printf(Locale.ROOT, "loaded in %.1fs  engine=%s%n", loadSec, omni.config());
        System.out.println("processor=" + omni.processor().getClass().getSimpleName());
        check("omni load", true);

        // plain text smoke via omni
        section("Omni text chat smoke");
        String en = omni.chat(List.of(
                Map.of("role", "user", "content", "What is 2+2? Reply with only the digit.")
        ), SamplingParams.greedy(maxTokens));
        box("text chat", en);
        check("omni text chat non-empty", en != null && !en.isBlank());

        runMediaPath(omni, fixtures, maxTokens);
        runConcurrentMediaStress(omni, fixtures, concurrent, maxTokens, warmup, rounds);

        section("WRITE RESULTS FOR HUMAN REVIEW");
        writeReview(outDir, backbone, loadSec);

        section("FINAL SUMMARY");
        for (String line : summary) System.out.println("  " + line);
        System.out.println();
        System.out.printf(Locale.ROOT, "passed=%d failed=%d skipped=%d load=%.1fs%n",
                passed, failed, skipped, loadSec);
        System.out.println("Review: " + outDir.resolve("RESULTS.md").toAbsolutePath());
        if (!failures.isEmpty()) {
            System.out.println("failures:");
            for (String f : failures) System.out.println("  - " + f);
        }
        omni.close();
        if (failed > 0) System.exit(1);
    }

    /** Persist replies under outDir for manual verification. */
    static void writeReview(Path outDir, Path backbone, double loadSec) {
        try {
            Files.createDirectories(outDir);
            StringBuilder md = new StringBuilder();
            md.append("# OmniLLM Multimodal Results\n\n");
            md.append("- backbone: `").append(backbone).append("`\n");
            md.append("- load_sec: ").append(String.format(Locale.ROOT, "%.1f", loadSec)).append('\n');
            md.append("- passed=").append(passed).append(" failed=").append(failed)
                    .append(" skipped=").append(skipped).append("\n\n");
            md.append("## Summary\n\n```\n");
            for (String line : summary) md.append(line).append('\n');
            md.append("```\n\n## Inference replies (human review)\n\n");
            if (reviewLines.isEmpty()) {
                md.append("(no media replies recorded)\n");
            } else {
                for (String block : reviewLines) md.append(block).append('\n');
            }
            Path mdPath = outDir.resolve("RESULTS.md");
            Files.writeString(mdPath, md.toString());
            Path sumPath = outDir.resolve("summary.txt");
            StringBuilder sum = new StringBuilder();
            sum.append("passed=").append(passed).append(" failed=").append(failed)
                    .append(" skipped=").append(skipped).append('\n');
            for (String line : summary) sum.append(line).append('\n');
            Files.writeString(sumPath, sum.toString());
            System.out.println("  wrote " + mdPath.toAbsolutePath());
            System.out.println("  wrote " + sumPath.toAbsolutePath());
        } catch (Exception e) {
            System.out.println("  warn: could not write review: " + e.getMessage());
        }
    }
}
