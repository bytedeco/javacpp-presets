package media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.vision.opencv.OpenCVIO;
import org.bytedeco.pytorch.llm.tokenizers.Encoding;
import org.bytedeco.pytorch.llm.tokenizers.Tiktoken;
import org.bytedeco.pytorch.vision.ffmpeg.FFmpegLoader;
import org.bytedeco.pytorch.vision.opencv.MatToTensor;

import java.nio.file.*;
import java.util.*;

/**
 * Multi-dimensional correctness + performance benchmark for:
 * <ol>
 *   <li>Tiktoken (cl100k_base, o200k_base, p50k_base) — pure-Java BPE</li>
 *   <li>FFmpeg JavaGLue — video/audio decode via javacpp-ffmpeg</li>
 *   <li>OpenCV JavaGLue — image I/O via javacpp-opencv</li>
 * </ol>
 *
 * <p>Run with:
 * <pre>{@code
 * java --add-opens=java.base/java.nio=ALL-UNNAMED \
 *      -cp "target/classes:$(mvn dependency:build-classpath -q -DincludeScope=runtime -Dmdep.outputFile=/dev/stdout)" \
 *      media.BenchmarkTiktokenFFmpegOpenCV
 * }</pre>
 */
public class BenchmarkTiktokenFFmpegOpenCV {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void benchmark(String name, CheckedRunnable r) {
        long t0 = System.nanoTime();
        try {
            r.run();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println("  OK  " + name + " (" + ms + " ms)");
        } catch (Throwable e) {
            failed++;
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.println(" FAIL " + name + " (" + ms + " ms): " + e.getMessage());
            report.append("FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static void check(String name, boolean ok) {
        if (ok) {
            passed++;
            System.out.println("    CHECK " + name + ": OK");
        } else {
            failed++;
            report.append("CHECK FAILED: ").append(name).append('\n');
            System.out.println("    CHECK " + name + ": FAIL");
        }
    }

    static void checkEq(String name, Object expected, Object actual) {
        boolean ok;
        if (expected instanceof int[] ea && actual instanceof int[] aa) {
            ok = java.util.Arrays.equals(ea, aa);
        } else if (expected instanceof long[] el && actual instanceof long[] al) {
            ok = java.util.Arrays.equals(el, al);
        } else if (expected instanceof byte[] eb && actual instanceof byte[] ab) {
            ok = java.util.Arrays.equals(eb, ab);
        } else {
            ok = Objects.equals(expected, actual);
        }
        if (!ok && expected instanceof Number && actual instanceof Number) {
            double diff = Math.abs(((Number) expected).doubleValue() - ((Number) actual).doubleValue());
            ok = Double.isNaN(diff) ? Double.isNaN(((Number) expected).doubleValue()) : diff < 1e-9;
        }
        String expStr = expected instanceof int[] a ? java.util.Arrays.toString(a) : String.valueOf(expected);
        String actStr = actual instanceof int[] a ? java.util.Arrays.toString(a) : String.valueOf(actual);
        if (ok) {
            passed++;
            System.out.println("    CHECK " + name + ": OK (" + expStr + ")");
        } else {
            failed++;
            report.append("CHECK FAILED [").append(name).append("]: expected=").append(expStr)
                    .append(", actual=").append(actStr).append('\n');
            System.out.println("    CHECK " + name + ": FAIL (expected=" + expStr + ", got=" + actStr + ")");
        }
    }

    // ---- Reference outputs from Python tiktoken 0.13+ ----

    // cl100k_base
    static final Map<String, int[]> CL100K_CASES = Map.of(
            "Hello world", new int[]{9906, 1917},
            "Hello", new int[]{9906},
            "Hello world!", new int[]{9906, 1917, 0},
            "日本語", new int[]{9080, 22656, 45918, 252},
            "🎉", new int[]{9468, 236, 231} // 🎉 as surrogate pair
    );

    // o200k_base
    static final Map<String, int[]> O200K_CASES = Map.of(
            "Hello world", new int[]{13225, 2375},
            "Hello", new int[]{13225},
            "日本語", new int[]{9048, 40909}
    );

    // p50k_base
    static final Map<String, int[]> P50K_CASES = Map.of(
            "Hello world", new int[]{15496, 995},
            "Hello", new int[]{15496}
    );

    public static void main(String[] args) throws Exception {
        Path tmpDir = Files.createTempDirectory("tiktoken_ffmpeg_opencv_bench");
        System.out.println("Temp: " + tmpDir + "\n");

        // ─── 1. Tiktoken ────────────────────────────────────────────────────
        System.out.println("══ 1. Tiktoken ══");
        section("cl100k_base encode/decode", () -> {
            Tiktoken enc = Tiktoken.forEncoding("cl100k_base");
            check("vocabSize", enc.nVocab() == 100277); // max_token_value+1 (Python n_vocab)

            for (var e : CL100K_CASES.entrySet()) {
                String text = e.getKey();
                int[] expected = e.getValue();
                Encoding encResult = enc.encode(text, false);
                checkEq("encode(" + repr(text) + ")", expected, encResult.ids());
                String decoded = enc.decode(expected, false);
                // Round-trip: only for ASCII
                if (text.codePoints().allMatch(cp -> cp < 128)) {
                    checkEq("decode(encode(" + repr(text) + "))", text, decoded);
                }
            }
        });

        section("cl100k_base special tokens", () -> {
            Tiktoken enc = Tiktoken.forEncoding("cl100k_base");
            // EOT id = 100257
            checkEq("EOT id", 100257, enc.specialTokenId("<|endoftext|>"));
            check("EOT is special", enc.isSpecialToken(100257));
            // Encode with allowed special
            Encoding eot = enc.encode("<|endoftext|>", false, Set.of("<|endoftext|>"));
            checkEq("EOT encode", new int[]{100257}, eot.ids());
            // Decode special
            String decoded = enc.decode(new int[]{100257}, false);
            checkEq("EOT decode", "<|endoftext|>", decoded);
        });

        section("cl100k_base encodeBatch", () -> {
            Tiktoken enc = Tiktoken.forEncoding("cl100k_base");
            List<String> texts = List.of("Hello", " world", "!");
            List<Encoding> batch = enc.encodeBatch(texts, false);
            check("batch size", batch.size() == 3);
            checkEq("batch[0]", new int[]{9906}, batch.get(0).ids());
        });

        section("cl100k_base throughput", () -> {
            Tiktoken enc = Tiktoken.forEncoding("cl100k_base");
            String longText = "Hello world ".repeat(1000);
            // Warmup
            for (int i = 0; i < 10; i++) enc.encode(longText, false);
            // Measure
            long t0 = System.nanoTime();
            int iters = 100;
            for (int i = 0; i < iters; i++) enc.encode(longText, false);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            int charsPerIter = longText.length();
            long totalChars = (long) charsPerIter * iters;
            double charsPerSec = totalChars / (ms / 1000.0);
            double toksPerSec = charsPerSec / 5.5; // ~5.5 chars/token
            System.out.println("    Throughput: " + String.format("%.0f", charsPerSec) + " chars/s, "
                    + String.format("%.0f", toksPerSec) + " tokens/s (" + ms + " ms / " + iters + " iters)");
            check("throughput > 0", charsPerSec > 0);
        });

        section("o200k_base encode", () -> {
            Tiktoken enc = Tiktoken.forEncoding("o200k_base");
            check("vocabSize", enc.nVocab() == 200019); // Python n_vocab = max_token_value+1
            for (var e : O200K_CASES.entrySet()) {
                String text = e.getKey();
                int[] expected = e.getValue();
                Encoding encResult = enc.encode(text, false);
                checkEq("o200k encode(" + repr(text) + ")", expected, encResult.ids());
            }
        });

        section("p50k_base encode", () -> {
            Tiktoken enc = Tiktoken.forEncoding("p50k_base");
            check("vocabSize", enc.nVocab() == 50281);
            for (var e : P50K_CASES.entrySet()) {
                String text = e.getKey();
                int[] expected = e.getValue();
                Encoding encResult = enc.encode(text, false);
                checkEq("p50k encode(" + repr(text) + ")", expected, encResult.ids());
            }
        });

        section("p50k_edit encode", () -> {
            Tiktoken enc = Tiktoken.forEncoding("p50k_edit");
            check("p50k_edit vocabSize", enc.nVocab() == 50284);
            // Basic encode
            Encoding e = enc.encode("Hello world", false);
            check("p50k_edit encode non-empty", e.ids().length > 0);
        });

        section("FastTokenizer adapter", () -> {
            Tiktoken enc = Tiktoken.forEncoding("cl100k_base");
            org.bytedeco.pytorch.llm.tokenizers.FastTokenizer ft = enc.toFastTokenizer();
            Encoding e = ft.encode("Hello world", true);
            checkEq("FastTokenizer adapter", new int[]{9906, 1917}, e.ids());
        });

        // ─── 2. FFmpeg ────────────────────────────────────────────────────
        System.out.println("\n══ 2. FFmpeg ══");
        Path testVideo = tmpDir.resolve("test.mp4");
        Path testAudio = tmpDir.resolve("test.wav");
        boolean ffmpegAvailable = generateTestMedia(testVideo, testAudio);
        if (!ffmpegAvailable) {
            System.out.println("  [FFmpeg not available — skipping FFmpeg section]");
        } else {
            section("FFmpeg video metadata", () -> {
                var vf = FFmpegLoader.openVideo(testVideo.toString());
                try {
                    check("width > 0", vf.width() > 0);
                    check("height > 0", vf.height() > 0);
                    check("fps > 0", vf.fps() > 0);
                    System.out.println("    Video: " + vf.width() + "x" + vf.height() + " @" + vf.fps() + " fps");
                } finally {
                    vf.close();
                }
            });

            section("FFmpeg video decode", () -> {
                try (var vf = FFmpegLoader.openVideo(testVideo.toString())) {
                    java.util.List<Tensor> frames = vf.readFrames();
                    check("at least 1 frame", frames.size() >= 1);
                    if (!frames.isEmpty()) {
                        Tensor f = frames.get(0);
                        long[] shape = shapes(f);
                        checkEq("frame shape[0]=3", 3, shape[0]);
                        checkEq("frame shape[1]>0", vf.height(), shape[1]);
                        checkEq("frame shape[2]>0", vf.width(), shape[2]);
                        check("frame dtype float", isFloatdtype(f));
                    }
                }
            });

            section("FFmpeg audio metadata", () -> {
                var af = FFmpegLoader.openAudio(testAudio.toString());
                try {
                    check("sampleRate > 0", af.sampleRate() > 0);
                    check("channels > 0", af.channels() > 0);
                    System.out.println("    Audio: " + af.sampleRate() + " Hz, " + af.channels() + " ch");
                } finally {
                    af.close();
                }
            });

            section("FFmpeg audio decode", () -> {
                try (var af = FFmpegLoader.openAudio(testAudio.toString())) {
                    Tensor wave = af.read();
                    long[] shape = shapes(wave);
                    check("wave shape[0]>0", shape[0] > 0); // channels
                    check("wave shape[1]>0", shape[1] > 0); // time
                    check("wave dtype float", isFloatdtype(wave));
                }
            });

            section("FFmpeg decode throughput", () -> {
                long t0 = System.nanoTime();
                int iters = 5;
                for (int i = 0; i < iters; i++) {
                    try (var vf = FFmpegLoader.openVideo(testVideo.toString())) {
                        vf.readFrames();
                    }
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                double fps = (10.0 * iters) / (ms / 1000.0);
                System.out.println("    Decode throughput: " + String.format("%.1f", fps) + " frames/s (" + ms + " ms / " + iters + " iters of ~10 frames)");
                check("ffmpeg throughput > 0", fps > 0);
            });
        }

        // ─── 3. OpenCV ───────────────────────────────────────────────────
        System.out.println("\n══ 3. OpenCV ══");
        Path testImage = tmpDir.resolve("test.png");
        Path testJpg = tmpDir.resolve("test.jpg");
        boolean opencvAvailable = generateTestImage(testImage, testJpg);
        if (!opencvAvailable) {
            System.out.println("  [OpenCV not available — skipping OpenCV section]");
        } else {
            section("OpenCV readImage", () -> {
                Tensor img = OpenCVIO.readImage(testImage.toString());
                long[] shape = shapes(img);
                check("shape[0]=3", shape.length >= 1 && shape[0] == 3);
                check("height>0", shape.length >= 2 && shape[1] > 0);
                check("width>0", shape.length >= 3 && shape[2] > 0);
                check("float dtype", isFloatdtype(img));
                System.out.println("    Image shape: " + Arrays.toString(shape));
            });

            section("OpenCV readImageGray", () -> {
                Tensor gray = OpenCVIO.readImageGray(testImage.toString());
                long[] shape = shapes(gray);
                check("grayscale channels=1", shape[0] == 1);
                check("grayscale float", isFloatdtype(gray));
            });

            section("OpenCV write + read roundtrip", () -> {
                Tensor original = OpenCVIO.readImage(testImage.toString());
                Path roundtripPath = tmpDir.resolve("roundtrip.png");
                OpenCVIO.writeImage(roundtripPath, original);
                Tensor roundtrip = OpenCVIO.readImage(roundtripPath);
                check("roundtrip shape matches", Arrays.equals(shapes(original), shapes(roundtrip)));
                // Pixel values should be close
                Tensor diff = original.sub(roundtrip.abs());
                float maxDiff = diff.max().item().toFloat();
                check("roundtrip pixel diff < 2", maxDiff < 2.0f);
            });

            section("OpenCV resize", () -> {
                Tensor img = OpenCVIO.readImage(testImage.toString());
                Tensor small = OpenCVIO.resize(img, 224, 224);
                long[] shape = shapes(small);
                checkEq("resize H=224", 224, shape[1]);
                checkEq("resize W=224", 224, shape[2]);
            });

            section("OpenCV encode JPEG", () -> {
                Tensor img = OpenCVIO.readImage(testImage.toString());
                byte[] jpgBytes = OpenCVIO.encode(img, "jpg");
                check("jpg bytes > 0", jpgBytes.length > 0);
                // Verify JPEG magic bytes
                checkEq("JPEG SOI", 0xFF, jpgBytes[0] & 0xFF);
                checkEq("JPEG SOI2", 0xD8, jpgBytes[1] & 0xFF);
            });

            section("OpenCV toGrayscale", () -> {
                Tensor img = OpenCVIO.readImage(testImage.toString());
                Tensor gray = OpenCVIO.toGrayscale(img);
                long[] shape = shapes(gray);
                checkEq("grayscale C=1", 1, shape[0]);
            });

            section("OpenCV crop", () -> {
                Tensor img = OpenCVIO.readImage(testImage.toString());
                long[] orig = shapes(img);
                int h = (int) orig[1];
                int w = (int) orig[2];
                if (h > 10 && w > 10) {
                    Tensor crop = OpenCVIO.crop(img, 0, 0, h / 2, w / 2);
                    long[] cs = shapes(crop);
                    checkEq("crop H", h / 2, cs[1]);
                    checkEq("crop W", w / 2, cs[2]);
                } else {
                    check("skip crop (image too small)", true);
                }
            });

            section("OpenCV read throughput", () -> {
                long t0 = System.nanoTime();
                int iters = 20;
                for (int i = 0; i < iters; i++) {
                    OpenCVIO.readImage(testImage.toString());
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                double imgPerSec = iters / (ms / 1000.0);
                System.out.println("    Read throughput: " + String.format("%.1f", imgPerSec) + " images/s (" + ms + " ms / " + iters + " iters)");
                check("opencv throughput > 0", imgPerSec > 0);
            });

            section("OpenCV MatToTensor roundtrip", () -> {
                // Mat→Tensor→Mat roundtrip
                var mat = new org.bytedeco.opencv.opencv_core.Mat(
                        org.bytedeco.opencv.global.opencv_core.CV_8UC3, 64, 64);
                MatToTensor.toMat(
                        MatToTensor.fromMat(mat));
                check("MatToTensor roundtrip no-exception", true);
                mat.close();
            });
        }

        // ─── Summary ────────────────────────────────────────────────────
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }

        // Cleanup
        deleteRecursive(tmpDir);
    }

    // ---- Helpers ----

    static String repr(String s) {
        if (s == null) return "null";
        if (s.codePoints().allMatch(cp -> cp < 128)) return "\"" + s + "\"";
        return "U+" + s.codePoints().toArray()[0] + "…";
    }

    static long[] shapes(Tensor t) {
        long ndim = t.dim();
        long[] s = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) s[i] = t.size(i);
        return s;
    }

    static boolean isFloatdtype(Tensor t) {
        try {
            return t.dtype().toString().contains("Float") || t.dtype().toString().contains("float");
        } catch (Exception e) {
            return false;
        }
    }

    static void section(String name, CheckedRunnable r) {
        System.out.println("\n── " + name + " ──");
        try {
            r.run();
        } catch (Exception e) {
            failed++;
            System.out.println("  FAIL " + name + ": " + e.getMessage());
            report.append("SECTION FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    // ---- Synthetic test data generators ----

    static boolean generateTestMedia(Path videoPath, Path audioPath) {
        try {
            // Try using FFmpegLoader to write test files via raw PCM
            // Generate a minimal WAV with sine wave
            int sr = 16000;
            int ch = 1;
            int duration = 1; // seconds
            int numSamples = sr * duration;
            float[] samples = new float[numSamples];
            for (int i = 0; i < numSamples; i++) {
                samples[i] = (float) Math.sin(2 * Math.PI * 440 * i / sr); // 440 Hz sine
            }
            // Write WAV using FFmpegLoader.decodeAudio (just the data write)
            // Actually we just create a small binary WAV
            writeWav(audioPath, samples, sr, ch);
            return true;
        } catch (Exception e) {
            System.out.println("    [Could not generate test media: " + e.getMessage() + "]");
            return false;
        }
    }

    static void writeWav(Path path, float[] samples, int sampleRate, int channels) throws Exception {
        int bitsPerSample = 16;
        int byteRate = sampleRate * channels * bitsPerSample / 8;
        int blockAlign = channels * bitsPerSample / 8;
        int dataSize = samples.length * 2; // 16-bit = 2 bytes per sample

        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        java.io.DataOutputStream dos = new java.io.DataOutputStream(baos);
        dos.writeBytes("RIFF");
        dos.writeInt(Integer.reverseBytes(36 + dataSize));
        dos.writeBytes("WAVE");
        dos.writeBytes("fmt ");
        dos.writeInt(Integer.reverseBytes(16)); // subchunk1 size
        dos.writeShort(Short.reverseBytes((short) 1)); // PCM
        dos.writeShort(Short.reverseBytes((short) channels));
        dos.writeInt(Integer.reverseBytes(sampleRate));
        dos.writeInt(Integer.reverseBytes(byteRate));
        dos.writeShort(Short.reverseBytes((short) blockAlign));
        dos.writeShort(Short.reverseBytes((short) bitsPerSample));
        dos.writeBytes("data");
        dos.writeInt(Integer.reverseBytes(dataSize));
        for (float s : samples) {
            short sample = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, (int) (s * 32767)));
            dos.writeShort(Short.reverseBytes(sample));
        }
        dos.close();
        Files.write(path, baos.toByteArray());
    }

    static boolean generateTestImage(Path pngPath, Path jpgPath) {
        try {
            // Create a 256x256 RGB test image using raw PNG writing
            // Since we may not have ffmpeg, use Java2D to create a test image
            java.awt.image.BufferedImage bi = new java.awt.image.BufferedImage(256, 256, java.awt.image.BufferedImage.TYPE_INT_RGB);
            java.awt.Graphics2D g = bi.createGraphics();
            // Checkerboard
            for (int y = 0; y < 256; y++) {
                for (int x = 0; x < 256; x++) {
                    if ((x / 32 + y / 32) % 2 == 0) {
                        g.setColor(java.awt.Color.RED);
                    } else {
                        g.setColor(java.awt.Color.BLUE);
                    }
                    g.fillRect(x, y, 1, 1);
                }
            }
            g.setColor(java.awt.Color.GREEN);
            g.drawLine(0, 128, 256, 128);
            g.dispose();
            javax.imageio.ImageIO.write(bi, "PNG", pngPath.toFile());
            javax.imageio.ImageIO.write(bi, "JPG", jpgPath.toFile());
            return true;
        } catch (Exception e) {
            System.out.println("    [Could not generate test image: " + e.getMessage() + "]");
            return false;
        }
    }

    static void deleteRecursive(Path path) {
        try {
            if (Files.isDirectory(path)) {
                try (var entries = Files.list(path)) {
                    entries.forEach(BenchmarkTiktokenFFmpegOpenCV::deleteRecursive);
                }
            }
            Files.deleteIfExists(path);
        } catch (Exception ignored) {}
    }
}
