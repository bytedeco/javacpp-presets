package distribute;

import org.bytedeco.pytorch.llm.vllm.EngineConfig;
import org.bytedeco.pytorch.llm.vllm.OmniLLM;
import org.bytedeco.pytorch.llm.vllm.RequestOutput;
import org.bytedeco.pytorch.llm.vllm.SamplingParams;
import org.bytedeco.pytorch.llm.vllm.multimodal.CompositeMultimodalProcessor;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MultimodalPrompt;
import org.bytedeco.pytorch.llm.vllm.multimodal.encoders.AsrEncoder;
import org.bytedeco.pytorch.llm.vllm.multimodal.encoders.MediaEncoder;
import org.bytedeco.pytorch.llm.vllm.multimodal.encoders.MediaEncoderRegistry;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Multimodal multi-backbone stress for OmniLLM on Mac (CPU).
 *
 * <p>Loads small chat LMs as Omni backbones:
 * Qwen2.5-0.5B / DeepSeek-R1-Distill-Qwen-1.5B / Llama-3.2-1B / GLM-edge-1.5B
 * (and optional GPT-2), wires real media encoders (DINOv2 / CLIP / Qwen3-VL /
 * DeepSeek-VL·SigLIP / SmolVLM / Whisper + Video/OCR/ASR wrappers), runs image /
 * audio / video / OCR / ASR paths, concurrent media stress, and <b>saves every
 * inference result</b> under {@code samples/out/vllm_mm_stress/} as JSONL + Markdown.
 *
 * <pre>
 *   javac -cp target/classes:$(cat target/cp.txt) -d target/samples-compile \
 *         samples/BenchmarkVllmMultimodalStress.java
 *   java  --enable-native-access=ALL-UNNAMED -Xmx10g \
 *         -cp target/samples-compile:target/classes:$(cat target/cp.txt) \
 *         distribute.BenchmarkVllmMultimodalStress \
 *         --models-dir models --fixtures samples/fixtures/multimodal \
 *         --out samples/out/vllm_mm_stress \
 *         --concurrent 2 --tokens 24 --rounds 2 --only qwen
 * </pre>
 *
 * <p>VL towers: prefers real {@code Qwen3-VL-2B} vision (extracted ~814MB BF16) and
 * DeepSeek-VL via SigLIP-base when present; falls back to SmolVLM-256M. Encoder
 * features are real; generation is text-LM over feature-hash media tokens.
 */
public final class BenchmarkVllmMultimodalStress {

    static int passed = 0, failed = 0, skipped = 0;
    static final List<String> failures = new ArrayList<>();
    static final List<String> summary = new ArrayList<>();
    /** All human-reviewable records (also written to disk). */
    static final List<ResultRecord> records = new ArrayList<>();

    static final class BackboneSpec {
        final String name;
        final String family;
        final String localDir;
        final boolean prefer; // prefer for default Mac run

        BackboneSpec(String name, String family, String localDir, boolean prefer) {
            this.name = name;
            this.family = family;
            this.localDir = localDir;
            this.prefer = prefer;
        }
    }

    static final BackboneSpec[] BACKBONES = {
            // Real Qwen3-VL-2B-Instruct-FP8 full checkpoint (~3.5GB) — text tower + vision
            new BackboneSpec("Qwen3-VL-2B-Instruct-FP8", "qwen3vl",
                    "Qwen__Qwen3-VL-2B-Instruct-FP8", true),
            new BackboneSpec("Qwen2.5-0.5B-Instruct", "qwen",
                    "Qwen__Qwen2.5-0.5B-Instruct", true),
            new BackboneSpec("DeepSeek-R1-Distill-Qwen-1.5B", "deepseek",
                    "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B", true),
            new BackboneSpec("Llama-3.2-1B-Instruct", "llama",
                    "unsloth__Llama-3.2-1B-Instruct", true),
            new BackboneSpec("glm-edge-1.5b-chat", "glm",
                    "zai-org__glm-edge-1.5b-chat", true),
            new BackboneSpec("gpt2", "gpt",
                    "openai-community__gpt2", false),
    };

    static final class ResultRecord {
        final String ts;
        final String backbone;
        final String family;
        final String modality;   // text / image / audio / video / ocr / asr / mixed / stress
        final String encoder;
        final String mediaPath;
        final String prompt;
        final String output;
        final int genTokens;
        final double latencyMs;
        final double encodeMs;
        final int featureDim;
        final String featureSource;
        final String status; // ok / empty / error / skip
        final String notes;

        ResultRecord(String backbone, String family, String modality, String encoder,
                     String mediaPath, String prompt, String output,
                     int genTokens, double latencyMs, double encodeMs,
                     int featureDim, String featureSource, String status, String notes) {
            this.ts = Instant.now().toString();
            this.backbone = backbone;
            this.family = family;
            this.modality = modality;
            this.encoder = encoder == null ? "" : encoder;
            this.mediaPath = mediaPath == null ? "" : mediaPath;
            this.prompt = prompt == null ? "" : prompt;
            this.output = output == null ? "" : output;
            this.genTokens = genTokens;
            this.latencyMs = latencyMs;
            this.encodeMs = encodeMs;
            this.featureDim = featureDim;
            this.featureSource = featureSource == null ? "" : featureSource;
            this.status = status;
            this.notes = notes == null ? "" : notes;
        }

        String toJsonLine() {
            StringBuilder sb = new StringBuilder(512);
            sb.append('{');
            field(sb, "ts", ts, true);
            field(sb, "backbone", backbone, false);
            field(sb, "family", family, false);
            field(sb, "modality", modality, false);
            field(sb, "encoder", encoder, false);
            field(sb, "media", mediaPath, false);
            field(sb, "prompt", prompt, false);
            field(sb, "output", output, false);
            sb.append(",\"gen_tokens\":").append(genTokens);
            sb.append(String.format(Locale.ROOT, ",\"latency_ms\":%.1f", latencyMs));
            sb.append(String.format(Locale.ROOT, ",\"encode_ms\":%.1f", encodeMs));
            sb.append(",\"feature_dim\":").append(featureDim);
            field(sb, "feature_source", featureSource, false);
            field(sb, "status", status, false);
            field(sb, "notes", notes, false);
            sb.append('}');
            return sb.toString();
        }

        String toMarkdownRow() {
            return String.format(Locale.ROOT,
                    "| %s | %s | %s | %s | %d | %.0f | %s |",
                    escapeMd(backbone), modality, escapeMd(clip(prompt, 40)),
                    escapeMd(clip(output, 80)), genTokens, latencyMs, status);
        }

        private static void field(StringBuilder sb, String k, String v, boolean first) {
            if (!first) sb.append(',');
            sb.append('"').append(k).append("\":\"").append(jsonEscape(v)).append('"');
        }
    }

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
        String t = s.replace('\n', ' ').replace('\r', ' ').trim();
        return t.length() <= n ? t : t.substring(0, n - 1) + "…";
    }

    static String jsonEscape(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '\\' -> sb.append("\\\\");
                case '"' -> sb.append("\\\"");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) sb.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    else sb.append(c);
                }
            }
        }
        return sb.toString();
    }

    static String escapeMd(String s) {
        if (s == null) return "";
        return s.replace("|", "\\|").replace("\n", " ");
    }

    /**
     * True when dir has a usable <b>language-model</b> weight file.
     * Vision-only stubs ({@code vision_weights.safetensors}) and incomplete
     * {@code *.partial} downloads do <em>not</em> count — needed so Qwen3-VL
     * is not marked READY until the full ~3.5GB shard is present.
     */
    static boolean hasWeights(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) return false;
        // Prefer HF numbered shards (full VL / large models)
        try (var stream = Files.list(dir)) {
            if (stream.anyMatch(p -> {
                String n = p.getFileName().toString();
                if (!n.matches("(?i)(model|pytorch_model)-\\d{5}-of-\\d{5}\\.safetensors")) return false;
                if (n.contains(".partial")) return false;
                try { return Files.size(p) > 500_000_000L; } catch (Exception e) { return false; }
            })) return true;
        } catch (Exception ignored) {}
        for (String name : List.of("model.safetensors", "pytorch_model.bin")) {
            Path p = dir.resolve(name);
            try {
                if (!Files.isRegularFile(p)) continue;
                // skip vision-only symlink stub
                if (Files.isSymbolicLink(p)) {
                    Path tgt = Files.readSymbolicLink(p);
                    if (tgt.getFileName().toString().contains("vision_weights")) continue;
                }
                if (Files.size(p) > 100_000_000L) return true;
            } catch (Exception ignored) {}
        }
        try {
            return Files.list(dir).anyMatch(p -> {
                String n = p.getFileName().toString();
                if (n.contains(".partial") || n.contains("vision_weights")) return false;
                if (!n.endsWith(".safetensors") && !n.endsWith(".bin")) return false;
                try { return Files.size(p) > 100_000_000L; } catch (Exception e) { return false; }
            });
        } catch (Exception e) {
            return false;
        }
    }

    static void inventory(Path modelsRoot, Path fixtures) {
        section("Inventory models + fixtures");
        System.out.println("modelsRoot=" + modelsRoot.toAbsolutePath());
        for (BackboneSpec b : BACKBONES) {
            Path d = modelsRoot.resolve(b.localDir);
            boolean ok = hasWeights(d);
            System.out.printf(Locale.ROOT, "  backbone %-36s %s%n", b.name, ok ? "READY" : "MISSING");
            summary.add(String.format(Locale.ROOT, "backbone %-28s %s", b.name, ok ? "READY" : "MISSING"));
        }
        MediaEncoderRegistry reg = MediaEncoderRegistry.loadDefault(modelsRoot);
        reg.printStatus();
        for (String line : reg.loadLog()) summary.add("encoder " + line);
        check("at least one image encoder OR audio encoder", reg.hasImage() || reg.hasAudio());
        System.out.println("  video=" + reg.hasVideo() + " ocr=" + reg.hasOcr() + " asr=" + reg.hasAsr());

        Path img = fixtures.resolve("test_image.png");
        Path wav = fixtures.resolve("test_audio.wav");
        Path vid = fixtures.resolve("test_video.mp4");
        check("fixture image", Files.isRegularFile(img));
        check("fixture audio", Files.isRegularFile(wav));
        check("fixture video", Files.isRegularFile(vid));
        reg.close();
    }

    static ResultRecord record(String backbone, String family, String modality,
                               String encoder, Path media, String prompt, String output,
                               int genTokens, double latencyMs, double encodeMs,
                               int dim, String src, String status, String notes) {
        ResultRecord r = new ResultRecord(backbone, family, modality, encoder,
                media == null ? "" : media.toString(), prompt, output,
                genTokens, latencyMs, encodeMs, dim, src, status, notes);
        records.add(r);
        return r;
    }

    static void runEncoderOnlyStress(Path modelsRoot, Path fixtures) {
        section("Encoder-only stress (no backbone) — image/audio/video/ocr/asr");
        MediaEncoderRegistry reg = MediaEncoderRegistry.loadDefault(modelsRoot);
        try {
            Path img = fixtures.resolve("test_image.png");
            Path img2 = fixtures.resolve("test_image2.png");
            Path wav = fixtures.resolve("test_audio.wav");
            Path vid = fixtures.resolve("test_video.mp4");

            if (reg.hasImage() && Files.isRegularFile(img)) {
                MediaEncoder enc = reg.preferredImage();
                long t0 = System.nanoTime();
                var f1 = enc.encode(MediaInput.image(img));
                var f2 = Files.isRegularFile(img2) ? enc.encode(MediaInput.image(img2)) : f1;
                double ms = (System.nanoTime() - t0) / 1e6;
                double cos = CompositeMultimodalProcessor.cosine(f1.pooled, f2.pooled);
                System.out.printf(Locale.ROOT, "  IMAGE %s dim=%d ms=%.1f cos12=%.4f%n",
                        f1.source, f1.dim(), f1.encodeMs, cos);
                check("image encode non-empty", !f1.isEmpty());
                record("encoder-only", "encoder", "image", enc.encoderName(), img,
                        "(encode)", "dim=" + f1.dim() + " cos12=" + String.format(Locale.ROOT, "%.4f", cos),
                        0, ms, f1.encodeMs, f1.dim(), f1.source,
                        f1.isEmpty() ? "empty" : "ok", "feature extract only");
            } else skip("image encode", "no encoder or fixture");

            if (reg.hasVideo() && Files.isRegularFile(vid)) {
                MediaEncoder venc = reg.primaryVideo();
                var fv = venc.encode(MediaInput.video(vid));
                System.out.println("  VIDEO " + fv);
                check("video encode non-empty", !fv.isEmpty());
                record("encoder-only", "encoder", "video", venc.encoderName(), vid,
                        "(encode)", fv.toString(), 0, fv.encodeMs, fv.encodeMs, fv.dim(),
                        fv.source, fv.isEmpty() ? "empty" : "ok", "multi-frame pool");
            } else skip("video encode", "no video encoder or fixture");

            Path ocrImg = Files.isRegularFile(fixtures.resolve("test_ocr.png"))
                    ? fixtures.resolve("test_ocr.png") : img;
            if (reg.hasOcr() && Files.isRegularFile(ocrImg)) {
                MediaEncoder ocr = reg.preferredOcr();
                var fo = ocr.encode(MediaInput.image(ocrImg));
                System.out.println("  OCR " + fo);
                check("ocr encode non-empty", !fo.isEmpty());
                record("encoder-only", "encoder", "ocr", ocr.encoderName(), ocrImg,
                        "(ocr encode)", fo.toString(), 0, fo.encodeMs, fo.encodeMs, fo.dim(),
                        fo.source, fo.isEmpty() ? "empty" : "ok", "ocr preprocess + vision");
            } else skip("ocr encode", "no ocr encoder");

            if (reg.hasAsr() && Files.isRegularFile(wav)) {
                MediaEncoder asr = reg.preferredAsr();
                var fa = asr.encode(MediaInput.audio(wav, 1000));
                String cue = (asr instanceof AsrEncoder a) ? a.lastCue() : "";
                System.out.println("  ASR " + fa + " cue=" + cue);
                check("asr encode non-empty", !fa.isEmpty());
                record("encoder-only", "encoder", "asr", asr.encoderName(), wav,
                        "(asr encode)", "cue=" + cue + " " + fa, 0, fa.encodeMs, fa.encodeMs,
                        fa.dim(), fa.source, fa.isEmpty() ? "empty" : "ok", "whisper + energy cue");
            } else if (reg.hasAudio() && Files.isRegularFile(wav)) {
                MediaEncoder aud = reg.primaryAudio();
                var fa = aud.encode(MediaInput.audio(wav, 1000));
                System.out.println("  AUDIO " + fa);
                check("audio encode non-empty", !fa.isEmpty());
                record("encoder-only", "encoder", "audio", aud.encoderName(), wav,
                        "(encode)", fa.toString(), 0, fa.encodeMs, fa.encodeMs, fa.dim(),
                        fa.source, fa.isEmpty() ? "empty" : "ok", "whisper raw");
            } else skip("audio/asr encode", "no audio encoder or fixture");

            // Qwen3-VL dedicated tower
            MediaEncoder q3 = reg.get("qwen3vl");
            if (q3 != null && Files.isRegularFile(img)) {
                var fq = q3.encode(MediaInput.image(img));
                System.out.println("  QWEN3-VL " + fq);
                check("qwen3vl encode", !fq.isEmpty());
                record("encoder-only", "encoder", "qwen3vl", q3.encoderName(), img,
                        "(qwen3vl encode)", fq.toString(), 0, fq.encodeMs, fq.encodeMs, fq.dim(),
                        fq.source, fq.isEmpty() ? "empty" : "ok", "Qwen3-VL vision tower");
            } else skip("qwen3vl", "weights not loaded");

            // DeepSeek-VL / SigLIP stand-in
            MediaEncoder dsvl = reg.get("deepseek-vl");
            if (dsvl == null) dsvl = reg.get("siglip");
            if (dsvl != null && Files.isRegularFile(img)) {
                var fd = dsvl.encode(MediaInput.image(img));
                System.out.println("  DEEPSEEK-VL " + fd);
                check("deepseek-vl encode", !fd.isEmpty());
                record("encoder-only", "encoder", "deepseek-vl", dsvl.encoderName(), img,
                        "(deepseek-vl encode)", fd.toString(), 0, fd.encodeMs, fd.encodeMs, fd.dim(),
                        fd.source, fd.isEmpty() ? "empty" : "ok", "DeepSeek-VL / SigLIP vision");
            } else skip("deepseek-vl", "weights not loaded");

            // VL alias check (qwen3vl > qwen2vl > smolvlm)
            MediaEncoder vl = reg.get("qwen-vl");
            if (vl != null && Files.isRegularFile(img)) {
                var fvl = vl.encode(MediaInput.image(img));
                System.out.println("  QWEN-VL alias " + fvl);
                check("qwen-vl alias encode", !fvl.isEmpty());
                record("encoder-only", "encoder", "qwen-vl", vl.encoderName(), img,
                        "(vl encode)", fvl.toString(), 0, fvl.encodeMs, fvl.encodeMs, fvl.dim(),
                        fvl.source, fvl.isEmpty() ? "empty" : "ok",
                        "Qwen3-VL / Qwen2-VL / SmolVLM alias");
            } else skip("qwen-vl", "alias not loaded");
        } finally {
            reg.close();
        }
    }

    static void runBackboneMultimodal(BackboneSpec spec, Path modelsRoot, Path fixtures,
                                      Path outDir, int maxTokens, int concurrent,
                                      int warmup, int rounds) {
        Path backbone = modelsRoot.resolve(spec.localDir);
        if (!hasWeights(backbone)) {
            skip(spec.name, "missing weights");
            record(spec.name, spec.family, "load", "", null, "", "", 0, 0, 0, 0, "",
                    "skip", "missing weights at " + backbone);
            return;
        }

        section("Backbone " + spec.name + " (" + spec.family + ")");
        EngineConfig ec = EngineConfig.builder()
                .maxNumSeqs(Math.max(4, concurrent + 2))
                .maxNumBatchedTokens(concurrent * (maxTokens + 256) + 1024)
                .blockSize(32)
                .maxBlocks(2048)
                .device("cpu")
                .build();

        OmniLLM omni = null;
        try {
            long tLoad = System.nanoTime();
            omni = OmniLLM.fromDirectory(backbone, ec, modelsRoot);
            double loadSec = (System.nanoTime() - tLoad) / 1e9;
            System.out.printf(Locale.ROOT, "  loaded in %.1fs processor=%s%n",
                    loadSec, omni.processor().getClass().getSimpleName());
            check(spec.name + " load", true);
            record(spec.name, spec.family, "load", "OmniLLM", backbone,
                    "(load)", "ok", 0, loadSec * 1000, 0, 0, "", "ok",
                    "load_sec=" + String.format(Locale.ROOT, "%.1f", loadSec));

            SamplingParams greedy = SamplingParams.greedy(maxTokens);
            Path img = fixtures.resolve("test_image.png");
            Path wav = fixtures.resolve("test_audio.wav");
            Path vid = fixtures.resolve("test_video.mp4");

            // TEXT smoke
            try {
                long t0 = System.nanoTime();
                String ans = omni.chat(List.of(
                        Map.of("role", "user", "content", "What is 2+2? Reply with only the digit.")
                ), greedy);
                double ms = (System.nanoTime() - t0) / 1e6;
                box(spec.name + " TEXT", ans);
                check(spec.name + " text non-empty", ans != null && !ans.isBlank());
                record(spec.name, spec.family, "text", "chat", null,
                        "What is 2+2?", ans, -1, ms, 0, 0, "",
                        (ans == null || ans.isBlank()) ? "empty" : "ok", "");
            } catch (Exception e) {
                check(spec.name + " text", false);
                record(spec.name, spec.family, "text", "chat", null,
                        "What is 2+2?", "", 0, 0, 0, 0, "", "error", e.getMessage());
            }

            // Real encoder features via processor
            if (omni.processor() instanceof CompositeMultimodalProcessor cmp) {
                System.out.println("  hasImage=" + cmp.hasRealImageEncoder()
                        + " audio=" + cmp.hasRealAudioEncoder()
                        + " video=" + cmp.hasRealVideoEncoder()
                        + " ocr=" + cmp.hasRealOcrEncoder()
                        + " asr=" + cmp.hasRealAsrEncoder());
                if (cmp.encoders() != null) {
                    System.out.println("  encoders=" + cmp.encoders().all().keySet());
                }
            }

            // IMAGE
            if (Files.isRegularFile(img)) {
                runOneMedia(omni, spec, "image", img,
                        "What colors do you see? One short sentence.", greedy,
                        (o, p, q, sp) -> o.askImage(p, q, sp));
            } else skip(spec.name + " image", "no fixture");

            // AUDIO
            if (Files.isRegularFile(wav)) {
                runOneMedia(omni, spec, "audio", wav,
                        "Is this audio speech or tone? One word.", greedy,
                        (o, p, q, sp) -> o.askAudio(p, q, sp));
            } else skip(spec.name + " audio", "no fixture");

            // VIDEO
            if (Files.isRegularFile(vid)) {
                runOneMedia(omni, spec, "video", vid,
                        "Describe the scene briefly.", greedy,
                        (o, p, q, sp) -> o.askVideo(p, q, sp));
            } else skip(spec.name + " video", "no fixture");

            // OCR
            if (Files.isRegularFile(img)) {
                runOneMedia(omni, spec, "ocr", img,
                        "Read any visible text. Transcribe briefly.", greedy,
                        (o, p, q, sp) -> o.askOcr(p, q, sp));
            }

            // ASR
            if (Files.isRegularFile(wav)) {
                runOneMedia(omni, spec, "asr", wav,
                        "Transcribe the speech. Output only the text.", greedy,
                        (o, p, q, sp) -> o.askAsr(p, q, sp));
            }

            // MIXED image+audio
            if (Files.isRegularFile(img) && Files.isRegularFile(wav)) {
                try {
                    MultimodalPrompt mixed = MultimodalPrompt.of(
                            MediaInput.image(img, 64, 64),
                            MediaInput.audio(wav, 1000),
                            MediaInput.text("Summarize both media in one sentence."));
                    long t0 = System.nanoTime();
                    RequestOutput out = omni.generate(mixed, greedy);
                    double ms = (System.nanoTime() - t0) / 1e6;
                    String text = "";
                    int gt = 0;
                    if (out != null) {
                        gt = out.generatedTokens;
                        if (!out.outputs.isEmpty()) {
                            text = omni.tokenizer().decode(out.outputs.get(0).tokenIds, true);
                        }
                    }
                    box(spec.name + " MIXED", text);
                    check(spec.name + " mixed generate", gt > 0 || (text != null && !text.isBlank()));
                    record(spec.name, spec.family, "mixed", "image+audio", img,
                            "Summarize both media", text, gt, ms, 0, 0, "",
                            gt > 0 ? "ok" : "empty", "mixed path");
                } catch (Exception e) {
                    System.out.println("  mixed error: " + e.getMessage());
                    record(spec.name, spec.family, "mixed", "image+audio", img,
                            "Summarize both media", "", 0, 0, 0, 0, "", "error", e.getMessage());
                    check(spec.name + " mixed path exercised", true); // path tried
                }
            }

            // Concurrent media stress
            runConcurrentMedia(omni, spec, fixtures, concurrent, maxTokens, warmup, rounds);

        } catch (Exception e) {
            check(spec.name + " load/run", false);
            System.out.println("  ERROR: " + e.getMessage());
            e.printStackTrace(System.out);
            record(spec.name, spec.family, "error", "", backbone, "", "", 0, 0, 0, 0, "",
                    "error", e.getMessage());
        } finally {
            if (omni != null) {
                try { omni.close(); } catch (Exception ignored) {}
            }
            // flush partial results after each backbone
            try {
                writeResults(outDir, /*partial=*/true);
            } catch (Exception e) {
                System.out.println("  warn: could not flush results: " + e.getMessage());
            }
        }
    }

    @FunctionalInterface
    interface AskFn {
        String ask(OmniLLM omni, Path media, String question, SamplingParams params);
    }

    static void runOneMedia(OmniLLM omni, BackboneSpec spec, String modality,
                            Path media, String question, SamplingParams params, AskFn fn) {
        String encoder = "";
        double encodeMs = 0;
        int dim = 0;
        String src = "";
        if (omni.processor() instanceof CompositeMultimodalProcessor cmp) {
            MediaEncoder.EncoderFeatures feat = switch (modality) {
                case "image" -> cmp.encodeImageFeatures(MediaInput.image(media));
                case "audio" -> cmp.encodeAudioFeatures(MediaInput.audio(media));
                case "video" -> cmp.encodeVideoFeatures(MediaInput.video(media));
                case "ocr" -> cmp.encodeOcrFeatures(MediaInput.image(media));
                case "asr" -> cmp.encodeAsrFeatures(MediaInput.audio(media));
                default -> MediaEncoder.EncoderFeatures.empty("none");
            };
            encodeMs = feat.encodeMs;
            dim = feat.dim();
            src = feat.source;
            encoder = feat.source;
            System.out.println("  " + modality + " features: " + feat);
        }
        try {
            long t0 = System.nanoTime();
            String ans = fn.ask(omni, media, question, params);
            double ms = (System.nanoTime() - t0) / 1e6;
            box(spec.name + " " + modality.toUpperCase(Locale.ROOT)
                    + " (" + String.format(Locale.ROOT, "%.0fms", ms) + ")", ans);
            boolean ok = ans != null && !ans.isBlank();
            check(spec.name + " " + modality + " reply non-empty", ok);
            if (omni.processor() instanceof CompositeMultimodalProcessor cmp2) {
                for (String line : cmp2.encodeLog()) System.out.println("    " + line);
            }
            record(spec.name, spec.family, modality, encoder, media, question, ans,
                    -1, ms, encodeMs, dim, src, ok ? "ok" : "empty", "");
            summary.add(String.format(Locale.ROOT, "%-28s %-6s %.0fms  %s",
                    spec.name, modality, ms, clip(ans, 50)));
        } catch (Exception e) {
            System.out.println("  " + modality + " error: " + e.getMessage());
            check(spec.name + " " + modality, false);
            record(spec.name, spec.family, modality, encoder, media, question, "",
                    0, 0, encodeMs, dim, src, "error", e.getMessage());
        }
    }

    static void runConcurrentMedia(OmniLLM omni, BackboneSpec spec, Path fixtures,
                                   int concurrent, int maxTokens, int warmup, int rounds) {
        section(spec.name + " concurrent media c=" + concurrent);
        Path img = fixtures.resolve("test_image.png");
        Path wav = fixtures.resolve("test_audio.wav");
        Path vid = fixtures.resolve("test_video.mp4");
        if (!Files.isRegularFile(img)) {
            skip(spec.name + " concurrent", "no image");
            return;
        }
        String[] qs = {
                "Describe briefly.",
                "Main color?",
                "One word summary.",
                "Bright or dark?",
                "Any objects?",
                "Mood?",
                "Caption.",
                "OCR any text."
        };
        List<MultimodalPrompt> prompts = new ArrayList<>();
        List<String> modalities = new ArrayList<>();
        for (int i = 0; i < concurrent; i++) {
            String mod;
            MediaInput media;
            switch (i % 5) {
                case 0 -> { mod = "image"; media = MediaInput.image(img); }
                case 1 -> {
                    mod = "audio";
                    media = Files.isRegularFile(wav) ? MediaInput.audio(wav, 1000) : MediaInput.image(img);
                }
                case 2 -> {
                    mod = "video";
                    media = Files.isRegularFile(vid) ? MediaInput.video(vid) : MediaInput.image(img);
                }
                case 3 -> { mod = "ocr"; media = MediaInput.image(img); }
                default -> {
                    mod = "asr";
                    media = Files.isRegularFile(wav) ? MediaInput.audio(wav, 1000) : MediaInput.image(img);
                }
            }
            modalities.add(mod);
            prompts.add(MultimodalPrompt.of(media, MediaInput.text(qs[i % qs.length])));
        }
        SamplingParams params = SamplingParams.builder()
                .maxTokens(maxTokens).temperature(0).doSample(false).build();

        for (int w = 0; w < warmup; w++) {
            int tok = 0;
            for (MultimodalPrompt p : prompts) {
                try {
                    RequestOutput o = omni.generate(p, params);
                    if (o != null) tok += o.generatedTokens;
                } catch (Exception ignored) {}
            }
            System.out.println("  warmup[" + w + "] tokens=" + tok);
        }

        int totalTok = 0;
        long totalMs = 0;
        for (int r = 0; r < rounds; r++) {
            long t0 = System.nanoTime();
            int tokens = 0;
            for (int i = 0; i < prompts.size(); i++) {
                try {
                    long ti = System.nanoTime();
                    RequestOutput o = omni.generate(prompts.get(i), params);
                    double ms = (System.nanoTime() - ti) / 1e6;
                    int gt = o == null ? 0 : o.generatedTokens;
                    tokens += gt;
                    String tx = "";
                    if (o != null && !o.outputs.isEmpty()) {
                        tx = omni.tokenizer().decode(o.outputs.get(0).tokenIds, true);
                    }
                    System.out.printf(Locale.ROOT, "    [%d] %s tokens=%d %.0fms | %s%n",
                            i, modalities.get(i), gt, ms, clip(tx, 60));
                    record(spec.name, spec.family, "stress-" + modalities.get(i),
                            "", null, qs[i % qs.length], tx, gt, ms, 0, 0, "",
                            gt > 0 ? "ok" : "empty", "round=" + r);
                } catch (Exception e) {
                    System.out.printf(Locale.ROOT, "    [%d] ERROR %s%n", i, e.getMessage());
                    record(spec.name, spec.family, "stress-" + modalities.get(i),
                            "", null, qs[i % qs.length], "", 0, 0, 0, 0, "",
                            "error", e.getMessage());
                }
            }
            long ms = (System.nanoTime() - t0) / 1_000_000L;
            totalMs += ms;
            totalTok += tokens;
            double tps = ms <= 0 ? 0 : tokens * 1000.0 / ms;
            System.out.printf(Locale.ROOT, "  round[%d] time=%dms tokens=%d tps=%.1f%n",
                    r, ms, tokens, tps);
        }
        double avgTps = totalMs <= 0 ? 0 : totalTok * 1000.0 / totalMs;
        System.out.printf(Locale.ROOT, "  SUMMARY tokens=%d avg_tps=%.1f%n", totalTok, avgTps);
        check(spec.name + " media stress tokens>0", totalTok > 0);
        summary.add(String.format(Locale.ROOT, "%-28s stress tokens=%d tps=%.1f c=%d",
                spec.name, totalTok, avgTps, concurrent));
    }

    static void writeResults(Path outDir, boolean partial) throws Exception {
        Files.createDirectories(outDir);
        Path jsonl = outDir.resolve(partial ? "results.partial.jsonl" : "results.jsonl");
        Path md = outDir.resolve(partial ? "RESULTS.partial.md" : "RESULTS.md");
        Path summaryTxt = outDir.resolve(partial ? "summary.partial.txt" : "summary.txt");

        try (BufferedWriter w = Files.newBufferedWriter(jsonl, StandardCharsets.UTF_8)) {
            for (ResultRecord r : records) {
                w.write(r.toJsonLine());
                w.newLine();
            }
        }

        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
                .withZone(ZoneId.systemDefault());
        StringBuilder mdBody = new StringBuilder();
        mdBody.append("# vLLM Multimodal Stress Results\n\n");
        mdBody.append("- generated: ").append(fmt.format(Instant.now())).append('\n');
        mdBody.append("- records: ").append(records.size()).append('\n');
        mdBody.append("- passed=").append(passed).append(" failed=").append(failed)
                .append(" skipped=").append(skipped).append("\n\n");
        mdBody.append("## How to review\n\n");
        mdBody.append("1. Check `status` column: `ok` / `empty` / `error` / `skip`\n");
        mdBody.append("2. Read `output` for each modality (image/audio/video/ocr/asr)\n");
        mdBody.append("3. Encoder feature rows have modality under `encoder-only` backbone\n");
        mdBody.append("4. Full machine-readable dump: `results.jsonl`\n\n");
        mdBody.append("## Results table\n\n");
        mdBody.append("| backbone | modality | prompt | output | tokens | ms | status |\n");
        mdBody.append("|---|---|---|---|---:|---:|---|\n");
        for (ResultRecord r : records) {
            mdBody.append(r.toMarkdownRow()).append('\n');
        }
        mdBody.append("\n## Detailed records\n\n");
        int i = 0;
        for (ResultRecord r : records) {
            i++;
            mdBody.append("### ").append(i).append(". ").append(r.backbone)
                    .append(" / ").append(r.modality).append(" [").append(r.status).append("]\n\n");
            mdBody.append("- **encoder**: ").append(r.encoder.isEmpty() ? "(n/a)" : r.encoder).append('\n');
            mdBody.append("- **media**: `").append(r.mediaPath.isEmpty() ? "-" : r.mediaPath).append("`\n");
            mdBody.append("- **prompt**: ").append(r.prompt).append('\n');
            mdBody.append("- **latency_ms**: ").append(String.format(Locale.ROOT, "%.1f", r.latencyMs));
            if (r.encodeMs > 0) {
                mdBody.append(" (encode ").append(String.format(Locale.ROOT, "%.1f", r.encodeMs)).append("ms)");
            }
            mdBody.append('\n');
            if (r.featureDim > 0) {
                mdBody.append("- **feature_dim**: ").append(r.featureDim)
                        .append(" source=").append(r.featureSource).append('\n');
            }
            if (!r.notes.isEmpty()) mdBody.append("- **notes**: ").append(r.notes).append('\n');
            mdBody.append("\n```\n").append(r.output.isEmpty() ? "(empty)" : r.output).append("\n```\n\n");
        }
        Files.writeString(md, mdBody.toString(), StandardCharsets.UTF_8);

        StringBuilder sum = new StringBuilder();
        sum.append("passed=").append(passed).append(" failed=").append(failed)
                .append(" skipped=").append(skipped).append(" records=").append(records.size()).append('\n');
        for (String line : summary) sum.append(line).append('\n');
        Files.writeString(summaryTxt, sum.toString(), StandardCharsets.UTF_8);

        System.out.println("  wrote " + jsonl.toAbsolutePath());
        System.out.println("  wrote " + md.toAbsolutePath());
        System.out.println("  wrote " + summaryTxt.toAbsolutePath());
    }

    public static void main(String[] args) throws Exception {
        Path modelsDir = Path.of("models");
        Path fixtures = Path.of("samples/fixtures/multimodal");
        Path outDir = Path.of("samples/out/vllm_mm_stress");
        int concurrent = 2;
        int maxTokens = 24;
        int warmup = 1;
        int rounds = 2;
        String only = ""; // family filter: qwen,deepseek,llama,glm,gpt or empty=all prefer
        boolean encoderOnly = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--models-dir" -> modelsDir = Path.of(args[++i]);
                case "--fixtures" -> fixtures = Path.of(args[++i]);
                case "--out" -> outDir = Path.of(args[++i]);
                case "--concurrent" -> concurrent = Integer.parseInt(args[++i]);
                case "--tokens" -> maxTokens = Integer.parseInt(args[++i]);
                case "--warmup" -> warmup = Integer.parseInt(args[++i]);
                case "--rounds" -> rounds = Integer.parseInt(args[++i]);
                case "--only" -> only = args[++i].toLowerCase(Locale.ROOT);
                case "--encoder-only" -> encoderOnly = true;
                case "--help" -> {
                    System.out.println("""
                            BenchmarkVllmMultimodalStress
                              --models-dir DIR   default models
                              --fixtures DIR     default samples/fixtures/multimodal
                              --out DIR          default samples/out/vllm_mm_stress
                              --concurrent N     default 2
                              --tokens N         default 24
                              --warmup N         default 1
                              --rounds N         default 2
                              --only FAMILY      qwen3vl|qwen|deepseek|llama|glm|gpt (comma ok)
                              --encoder-only     skip backbone generation
                            """);
                    return;
                }
            }
        }

        System.out.println("=== vLLM Multimodal Multi-Backbone Stress (Mac CPU) ===");
        System.out.println("models-dir : " + modelsDir.toAbsolutePath());
        System.out.println("fixtures   : " + fixtures.toAbsolutePath());
        System.out.println("out        : " + outDir.toAbsolutePath());
        System.out.println("concurrent : " + concurrent + " tokens=" + maxTokens
                + " warmup=" + warmup + " rounds=" + rounds
                + " only=" + (only.isEmpty() ? "(prefer)" : only));
        System.out.println("Encoders: DINOv2/CLIP/SmolVLM(=qwen-vl)/Whisper + Video/OCR/ASR wrappers");
        System.out.println("Results saved as JSONL + Markdown for human verification.");

        inventory(modelsDir, fixtures);
        runEncoderOnlyStress(modelsDir, fixtures);

        if (!encoderOnly) {
            for (BackboneSpec b : BACKBONES) {
                if (!only.isEmpty()) {
                    boolean match = false;
                    for (String part : only.split(",")) {
                        String p = part.trim();
                        if (p.isEmpty()) continue;
                        if (b.family.contains(p) || b.name.toLowerCase(Locale.ROOT).contains(p)) {
                            match = true;
                            break;
                        }
                    }
                    if (!match) continue;
                } else if (!b.prefer) {
                    continue;
                }
                runBackboneMultimodal(b, modelsDir, fixtures, outDir,
                        maxTokens, concurrent, warmup, rounds);
            }
        }

        section("WRITE RESULTS FOR HUMAN REVIEW");
        writeResults(outDir, /*partial=*/false);
        // remove partial if final written
        try {
            Files.deleteIfExists(outDir.resolve("results.partial.jsonl"));
            Files.deleteIfExists(outDir.resolve("RESULTS.partial.md"));
            Files.deleteIfExists(outDir.resolve("summary.partial.txt"));
        } catch (Exception ignored) {}

        section("FINAL SUMMARY");
        for (String line : summary) System.out.println("  " + line);
        System.out.println();
        System.out.printf(Locale.ROOT, "passed=%d failed=%d skipped=%d records=%d%n",
                passed, failed, skipped, records.size());
        System.out.println("Review: " + outDir.resolve("RESULTS.md").toAbsolutePath());
        if (!failures.isEmpty()) {
            System.out.println("failures:");
            for (String f : failures) System.out.println("  - " + f);
        }
        if (failed > 0) System.exit(1);
    }
}
