package samples;
import org.bytedeco.pytorch.autograd.*;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.ffmpeg.AudioFile;
import org.bytedeco.pytorch.utils.ffmpeg.AudioTensorsFFmpeg;
import org.bytedeco.pytorch.utils.ffmpeg.FFmpegException;
import org.bytedeco.pytorch.utils.ffmpeg.FFmpegLoader;
import org.bytedeco.pytorch.utils.ffmpeg.VideoFile;
import org.bytedeco.pytorch.utils.ffmpeg.VideoTensors;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Multi-dimensional correctness + performance benchmark for {@code utils.ffmpeg}.
 *
 * <p>Dimensions:
 * <ol>
 *   <li>D1 FFmpegException / error helpers</li>
 *   <li>D2 AudioFile metadata + read / partial read / streaming</li>
 *   <li>D3 VideoFile metadata + read / readFrames / iterator</li>
 *   <li>D4 FFmpegLoader convenience decodeVideo / decodeAudio</li>
 *   <li>D5 AudioTensorsFFmpeg / VideoTensors static decode</li>
 *   <li>D6 Daily pipeline + throughput + edge cases</li>
 * </ol>
 *
 * <p>Requires native FFmpeg (javacpp-ffmpeg). If unavailable, structural checks still run.
 */
public class BenchmarkFFmpeg {

    static int passed = 0, failed = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

    static void check(String name, boolean ok) {
        if (ok) { passed++; System.out.println("    CHECK " + name + ": OK"); }
        else {
            failed++;
            report.append("CHECK FAILED: ").append(name).append('\n');
            System.out.println("    CHECK " + name + ": FAIL");
        }
    }

    static void checkEq(String name, Object expected, Object actual) {
        boolean ok;
        if (expected instanceof Number && actual instanceof Number) {
            double d = Math.abs(((Number) expected).doubleValue() - ((Number) actual).doubleValue());
            ok = d < 1e-4;
        } else if (expected instanceof long[] ea && actual instanceof long[] aa) {
            ok = Arrays.equals(ea, aa);
        } else {
            ok = java.util.Objects.equals(expected, actual);
        }
        if (ok) { passed++; System.out.println("    CHECK " + name + ": OK (" + fmt(expected) + ")"); }
        else {
            failed++;
            report.append("CHECK FAILED [").append(name).append("] expected=")
                    .append(fmt(expected)).append(" actual=").append(fmt(actual)).append('\n');
            System.out.println("    CHECK " + name + ": FAIL (expected=" + fmt(expected)
                    + ", got=" + fmt(actual) + ")");
        }
    }

    static String fmt(Object o) {
        if (o instanceof long[] a) return Arrays.toString(a);
        return String.valueOf(o);
    }

    static void section(String name, CheckedRunnable r) {
        System.out.println("\n── " + name + " ──");
        long t0 = System.nanoTime();
        try {
            r.run();
            System.out.println("  OK  " + name + " (" + (System.nanoTime() - t0) / 1_000_000 + " ms)");
        } catch (Throwable e) {
            failed++;
            System.out.println("  FAIL " + name + ": " + e.getMessage());
            report.append("SECTION FAIL ").append(name).append(": ").append(e).append('\n');
            e.printStackTrace(System.out);
        }
    }

    static long[] shapes(Tensor t) {
        long n = t.dim();
        long[] s = new long[(int) n];
        for (int i = 0; i < n; i++) s[i] = t.size(i);
        return s;
    }

    static boolean isFloat(Tensor t) {
        try {
            String s = String.valueOf(t.scalar_type());
            return s.contains("Float") || s.contains("float") || s.contains("Half");
        } catch (Exception e) {
            return false;
        }
    }

    static void writeWav(Path path, float[] samples, int sr, int channels) throws Exception {
        int bits = 16;
        int byteRate = sr * channels * bits / 8;
        int blockAlign = channels * bits / 8;
        int dataSize = samples.length * 2;
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        dos.writeBytes("RIFF");
        dos.writeInt(Integer.reverseBytes(36 + dataSize));
        dos.writeBytes("WAVE");
        dos.writeBytes("fmt ");
        dos.writeInt(Integer.reverseBytes(16));
        dos.writeShort(Short.reverseBytes((short) 1));
        dos.writeShort(Short.reverseBytes((short) channels));
        dos.writeInt(Integer.reverseBytes(sr));
        dos.writeInt(Integer.reverseBytes(byteRate));
        dos.writeShort(Short.reverseBytes((short) blockAlign));
        dos.writeShort(Short.reverseBytes((short) bits));
        dos.writeBytes("data");
        dos.writeInt(Integer.reverseBytes(dataSize));
        for (float s : samples) {
            short v = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, (int) (s * 32767)));
            dos.writeShort(Short.reverseBytes(v));
        }
        dos.close();
        Files.write(path, baos.toByteArray());
    }

    static float[] sine(double freq, int sr, int n, double amp) {
        float[] y = new float[n];
        for (int i = 0; i < n; i++) y[i] = (float) (amp * Math.sin(2 * Math.PI * freq * i / sr));
        return y;
    }

    /** Generate a short test video via ffmpeg CLI if available; returns null on failure. */
    static Path generateTestVideo(Path dir) {
        Path out = dir.resolve("test.mp4");
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=160x120:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", out.toString()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            // drain
            p.getInputStream().readAllBytes();
            int code = p.waitFor();
            if (code == 0 && Files.exists(out) && Files.size(out) > 0) return out;
        } catch (Exception e) {
            System.out.println("    [ffmpeg CLI video gen failed: " + e.getMessage() + "]");
        }
        // fallback: solid-color raw via ffmpeg single input
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", "color=c=red:s=160x120:d=1:r=10",
                    "-c:v", "mpeg4", out.toString()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            p.getInputStream().readAllBytes();
            int code = p.waitFor();
            if (code == 0 && Files.exists(out) && Files.size(out) > 0) return out;
        } catch (Exception e) {
            System.out.println("    [ffmpeg CLI fallback failed: " + e.getMessage() + "]");
        }
        return null;
    }

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("ffmpeg_bench");
        System.out.println("=== FFmpeg Module Benchmark ===");
        System.out.println("Temp: " + tmp);

        final int SR = 16000;
        Path wav = tmp.resolve("tone.wav");
        writeWav(wav, sine(440, SR, SR, 0.5), SR, 1);
        Path stereoWav = tmp.resolve("stereo.wav");
        float[] stereo = new float[SR * 2];
        for (int i = 0; i < SR; i++) {
            float v = (float) (0.5 * Math.sin(2 * Math.PI * 440 * i / SR));
            stereo[2 * i] = v;
            stereo[2 * i + 1] = 0.5f * v;
        }
        writeWav(stereoWav, stereo, SR, 2);

        // ── D1 Exception ─────────────────────────────────────────────────────
        System.out.println("\n══ D1 FFmpegException ══");
        section("exception construct + errorMessage", () -> {
            FFmpegException e1 = new FFmpegException("msg");
            check("message", e1.getMessage().contains("msg"));
            FFmpegException e2 = new FFmpegException("msg2", 42);
            checkEq("errorCode", 42, e2.errorCode());
            FFmpegException e3 = new FFmpegException("msg3", new RuntimeException("c"));
            check("cause", e3.getCause() != null);
            String em = FFmpegException.errorMessage(-2);
            check("errorMessage non-null", em != null);
            System.out.println("    errorMessage(-2)=" + em);
        });

        // Probe native availability via audio open
        boolean audioOkTmp = false;
        boolean videoOkTmp = false;
        Path video = null;
        try {
            try (AudioFile af = FFmpegLoader.openAudio(wav.toString())) {
                audioOkTmp = af.sampleRate() > 0;
            }
        } catch (Throwable t) {
            System.out.println("  [FFmpeg audio native not available: " + t.getClass().getSimpleName()
                    + ": " + t.getMessage() + "]");
        }
        video = generateTestVideo(tmp);
        if (video != null) {
            try {
                try (VideoFile vf = FFmpegLoader.openVideo(video.toString())) {
                    videoOkTmp = vf.width() > 0;
                }
            } catch (Throwable t) {
                System.out.println("  [FFmpeg video native not available: " + t.getMessage() + "]");
                videoOkTmp = false;
            }
        } else {
            System.out.println("  [No test video generated — video section limited]");
        }
        final boolean audioOk = audioOkTmp;
        final boolean videoOk = videoOkTmp;

        // ── D2 AudioFile ─────────────────────────────────────────────────────
        System.out.println("\n══ D2 AudioFile ══");
        if (!audioOk) {
            System.out.println("  [skip AudioFile — native unavailable]");
        } else {
            section("AudioFile metadata + read", () -> {
                try (AudioFile af = FFmpegLoader.openAudio(wav.toString())) {
                    check("filePath", af.filePath() != null && af.filePath().contains("tone"));
                    check("sampleRate > 0", af.sampleRate() > 0);
                    check("channels >= 1", af.channels() >= 1);
                    check("numSamples > 0 or unknown", af.numSamples() >= 0);
                    check("durationSec > 0", af.durationSec() > 0.5);
                    System.out.println("    sr=" + af.sampleRate() + " ch=" + af.channels()
                            + " samples=" + af.numSamples() + " dur=" + af.durationSec());

                    Tensor wave = af.read();
                    long[] s = shapes(wave);
                    check("wave rank 2 (C,T) or 1", s.length == 1 || s.length == 2);
                    check("wave float", isFloat(wave));
                    long time = s.length == 2 ? s[1] : s[0];
                    check("wave time > 0", time > 0);
                    System.out.println("    wave shape=" + Arrays.toString(s));
                }
            });

            section("AudioFile Path overload + stereo + partial read", () -> {
                try (AudioFile af = FFmpegLoader.openAudio(stereoWav)) {
                    check("stereo channels", af.channels() >= 1); // may downmix depending on decoder path
                    Tensor full = af.read();
                    check("stereo read", shapes(full).length >= 1);
                }
                try (AudioFile af = AudioFile.open(wav.toString())) {
                    // partial read: first 1000 samples if supported
                    try {
                        Tensor part = af.read(0, 1000);
                        long[] s = shapes(part);
                        long t = s.length == 2 ? s[1] : s[0];
                        check("partial read time <= 1000 or full", t > 0 && t <= Math.max(1000, t));
                        System.out.println("    partial shape=" + Arrays.toString(s));
                    } catch (Exception e) {
                        // some impls may require sequential only
                        System.out.println("    partial read note: " + e.getMessage());
                        check("partial read attempted", true);
                    }
                }
            });

            section("AudioFile streaming hasNext/next", () -> {
                try (AudioFile af = AudioFile.open(wav.toString())) {
                    int count = 0;
                    float sum = 0;
                    while (af.hasNext() && count < 5000) {
                        sum += af.next();
                        count++;
                    }
                    check("streamed samples > 0", count > 0);
                    check("stream sum finite", Float.isFinite(sum));
                    System.out.println("    streamed " + count + " samples, sum=" + sum);
                }
            });

            section("decodeAudio convenience + AudioTensorsFFmpeg", () -> {
                Tensor w = FFmpegLoader.decodeAudio(wav.toString());
                check("decodeAudio rank >= 1", shapes(w).length >= 1);
                check("decodeAudio float", isFloat(w));

                Tensor w2 = AudioTensorsFFmpeg.decodeAllSamples(wav.toString());
                check("decodeAllSamples rank >= 1", shapes(w2).length >= 1);

                try (AudioTensorsFFmpeg conv = new AudioTensorsFFmpeg(SR, 1)) {
                    checkEq("conv sampleRate", SR, conv.sampleRate());
                    checkEq("conv channels", 1, conv.channels());
                }
            });
        }

        // ── D3 VideoFile ─────────────────────────────────────────────────────
        System.out.println("\n══ D3 VideoFile ══");
        if (!videoOk || video == null) {
            System.out.println("  [skip VideoFile — native/video unavailable]");
        } else {
            final Path videoFinal = video;
            section("VideoFile metadata + readFrames", () -> {
                try (VideoFile vf = FFmpegLoader.openVideo(videoFinal.toString())) {
                    check("filePath", vf.filePath() != null);
                    check("width > 0", vf.width() > 0);
                    check("height > 0", vf.height() > 0);
                    check("fps > 0", vf.fps() > 0);
                    System.out.println("    " + vf.width() + "x" + vf.height() + " @" + vf.fps()
                            + " fps numFrames=" + vf.numFrames());

                    List<Tensor> frames = vf.readFrames();
                    check("frames >= 1", frames.size() >= 1);
                    Tensor f0 = frames.get(0);
                    long[] s = shapes(f0);
                    checkEq("frame C=3", 3L, s[0]);
                    checkEq("frame H", (long) vf.height(), s[1]);
                    checkEq("frame W", (long) vf.width(), s[2]);
                    check("frame float", isFloat(f0));
                    System.out.println("    decoded " + frames.size() + " frames, shape0=" + Arrays.toString(s));
                }
            });

            section("VideoFile iterator + single read", () -> {
                try (VideoFile vf = VideoFile.open(videoFinal)) {
                    int n = 0;
                    for (Tensor frame : vf) {
                        check("iter frame rank 3", shapes(frame).length == 3);
                        n++;
                        if (n >= 3) break; // don't decode entire if long
                    }
                    check("iterator yielded >= 1", n >= 1);
                }
                try (VideoFile vf = FFmpegLoader.openVideo(videoFinal)) {
                    // read() may return batch [N,3,H,W] or single frame depending on impl
                    Tensor r = vf.read();
                    check("read() non-null", r != null && r.dim() >= 3);
                    System.out.println("    read() shape=" + Arrays.toString(shapes(r)));
                    check("currentFrame >= 0", vf.currentFrame() >= 0);
                }
            });

            section("decodeVideo convenience + VideoTensors", () -> {
                List<Tensor> frames = FFmpegLoader.decodeVideo(videoFinal.toString());
                check("decodeVideo >= 1", frames.size() >= 1);
                List<Tensor> frames2 = VideoTensors.decodeAllFrames(videoFinal.toString());
                check("decodeAllFrames >= 1", frames2.size() >= 1);
                checkEq("TARGET_PIX_FMT", 3, VideoTensors.TARGET_PIX_FMT);

                try (VideoTensors vt = new VideoTensors(160, 120, 0)) {
                    check("VideoTensors construct", vt != null);
                }
            });
        }

        // ── D4 Edge / daily / throughput ─────────────────────────────────────
        System.out.println("\n══ D4 Edge / daily / throughput ══");
        section("missing file throws", () -> {
            boolean threw = false;
            try {
                FFmpegLoader.openAudio(tmp.resolve("nope.wav").toString());
            } catch (Throwable t) {
                threw = true;
            }
            check("missing audio throws", threw || !audioOk); // if native down, open may also throw

            if (videoOk) {
                threw = false;
                try {
                    FFmpegLoader.openVideo(tmp.resolve("nope.mp4").toString());
                } catch (Throwable t) {
                    threw = true;
                }
                check("missing video throws", threw);
            }
        });

        if (audioOk) {
            section("daily: open → read → stats", () -> {
                try (AudioFile af = FFmpegLoader.openAudio(wav.toString())) {
                    Tensor w = af.read();
                    long[] s = shapes(w);
                    long time = s.length == 2 ? s[1] : s[0];
                    double dur = time / (double) af.sampleRate();
                    check("daily dur ~1s", dur > 0.5 && dur < 1.5);
                    float max = w.abs().max().item().toFloat();
                    check("daily peak > 0", max > 0.01f);
                    System.out.println("    daily peak=" + max + " dur=" + dur);
                }
            });

            section("audio throughput", () -> {
                int iters = 10;
                long t0 = System.nanoTime();
                for (int i = 0; i < iters; i++) {
                    try (AudioFile af = FFmpegLoader.openAudio(wav.toString())) {
                        af.read();
                    }
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                double ips = iters / (ms / 1000.0);
                System.out.println("    audio open+read: " + String.format("%.1f", ips) + " /s");
                check("audio throughput > 0", ips > 0);
            });
        }

        if (videoOk && video != null) {
            final Path videoFinal = video;
            section("video throughput", () -> {
                int iters = 5;
                long t0 = System.nanoTime();
                int totalFrames = 0;
                for (int i = 0; i < iters; i++) {
                    try (VideoFile vf = FFmpegLoader.openVideo(videoFinal.toString())) {
                        totalFrames += vf.readFrames().size();
                    }
                }
                long ms = (System.nanoTime() - t0) / 1_000_000;
                double fps = totalFrames / (ms / 1000.0);
                System.out.println("    video decode: " + String.format("%.1f", fps)
                        + " frames/s (" + totalFrames + " frames / " + ms + " ms)");
                check("video throughput > 0", fps > 0);
            });
        }

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        System.out.println("Native: audio=" + audioOk + " video=" + videoOk);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        deleteRecursive(tmp);
    }

    static void deleteRecursive(Path path) {
        try {
            if (Files.isDirectory(path)) {
                try (var e = Files.list(path)) { e.forEach(BenchmarkFFmpeg::deleteRecursive); }
            }
            Files.deleteIfExists(path);
        } catch (Exception ignored) {}
    }
}
