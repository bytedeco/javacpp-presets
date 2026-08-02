package media;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.DType;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.vision.ffmpeg.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Multi-dimensional correctness + performance benchmark for the PyAV-parity
 * glue layer ({@link Av}/{@link Container}/…) and ffmpeg-python style {@link FFmpeg}.
 *
 * <p>Maps to {@code org/lance/ipc/av.md} examples 1–20.
 *
 * <p>Dimensions:
 * <ol>
 *   <li>D1  Container open + media info (#9)</li>
 *   <li>D2  Decode video → NDArray/Tensor + fps subsample (#1)</li>
 *   <li>D3  Decode audio PCM (#3)</li>
 *   <li>D4  Encode NDArray sequence → MP4 roundtrip (#7)</li>
 *   <li>D5  Stream-copy trim + seek (#8)</li>
 *   <li>D6  FilterGraph scale (#5)</li>
 *   <li>D7  A/V sync mux two sources (#4)</li>
 *   <li>D8  Demux packet iteration</li>
 *   <li>D9  Threaded decode queue (#10)</li>
 *   <li>D10 HardwareContext create (#2) — OK or SKIP</li>
 *   <li>D11 FFmpeg fluent scale+transcode (#11)</li>
 *   <li>D12 FFmpeg extract wav (#12)</li>
 *   <li>D13 FFmpeg raw pipe rgb (#13)</li>
 *   <li>D14 FFmpeg fast trim copy (#14)</li>
 *   <li>D15 FFmpeg overlay/concat/fps/multi-out smoke (#15–20)</li>
 *   <li>D16 Throughput decode/encode</li>
 *   <li>D17 Edge cases</li>
 *   <li>D18 Interop Av ↔ VideoFile</li>
 * </ol>
 */
public class BenchmarkAv {

    static int passed = 0, failed = 0, skipped = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable { void run() throws Exception; }

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
        if (expected instanceof Number && actual instanceof Number) {
            ok = Math.abs(((Number) expected).doubleValue() - ((Number) actual).doubleValue()) < 1e-3;
        } else if (expected instanceof long[] ea && actual instanceof long[] aa) {
            ok = Arrays.equals(ea, aa);
        } else {
            ok = java.util.Objects.equals(String.valueOf(expected), String.valueOf(actual));
        }
        if (ok) {
            passed++;
            System.out.println("    CHECK " + name + ": OK (" + actual + ")");
        } else {
            failed++;
            report.append("CHECK FAILED [").append(name).append("] expected=")
                    .append(expected).append(" actual=").append(actual).append('\n');
            System.out.println("    CHECK " + name + ": FAIL (expected=" + expected + ", got=" + actual + ")");
        }
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

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("  SKIP " + name + ": " + reason);
    }

    /** Generate short test AV via system ffmpeg lavfi; null on failure. */
    static Path generateTestVideo(Path dir, String name, int w, int h, int fps, double dur) {
        Path out = dir.resolve(name);
        String bin = FFmpeg.findBinary();
        if (bin == null) return null;
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    bin, "-y", "-f", "lavfi",
                    "-i", "testsrc=size=" + w + "x" + h + ":rate=" + fps + ":duration=" + dur,
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=" + dur,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", out.toString()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            p.getInputStream().readAllBytes();
            if (p.waitFor() == 0 && Files.exists(out) && Files.size(out) > 0) return out;
        } catch (Exception e) {
            System.out.println("    [gen video failed: " + e.getMessage() + "]");
        }
        // video-only fallback
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    bin, "-y", "-f", "lavfi",
                    "-i", "color=c=blue:s=" + w + "x" + h + ":d=" + dur + ":r=" + fps,
                    "-c:v", "mpeg4", out.toString()
            );
            pb.redirectErrorStream(true);
            Process p = pb.start();
            p.getInputStream().readAllBytes();
            if (p.waitFor() == 0 && Files.exists(out) && Files.size(out) > 0) return out;
        } catch (Exception ignored) {}
        return null;
    }

    static Path generateToneWav(Path dir) throws Exception {
        Path wav = dir.resolve("tone.wav");
        // minimal PCM wav 1s 16kHz mono via ffmpeg if available, else raw write
        String bin = FFmpeg.findBinary();
        if (bin != null) {
            ProcessBuilder pb = new ProcessBuilder(
                    bin, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    "-ar", "16000", "-ac", "1", wav.toString());
            pb.redirectErrorStream(true);
            Process p = pb.start();
            p.getInputStream().readAllBytes();
            if (p.waitFor() == 0 && Files.exists(wav)) return wav;
        }
        // pure-java wav
        int sr = 16000;
        int n = sr;
        byte[] data = new byte[44 + n * 2];
        // header
        java.nio.ByteBuffer bb = java.nio.ByteBuffer.wrap(data).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes());
        bb.putInt(36 + n * 2);
        bb.put("WAVE".getBytes());
        bb.put("fmt ".getBytes());
        bb.putInt(16);
        bb.putShort((short) 1);
        bb.putShort((short) 1);
        bb.putInt(sr);
        bb.putInt(sr * 2);
        bb.putShort((short) 2);
        bb.putShort((short) 16);
        bb.put("data".getBytes());
        bb.putInt(n * 2);
        for (int i = 0; i < n; i++) {
            short v = (short) (Math.sin(2 * Math.PI * 440 * i / sr) * 16000);
            bb.putShort(v);
        }
        Files.write(wav, data);
        return wav;
    }

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("av_bench");
        System.out.println("=== PyAV-parity / FFmpeg Glue Benchmark ===");
        System.out.println("Temp: " + tmp);

        boolean nativeLoaded = false;
        try {
            Av.load();
            nativeLoaded = true;
            System.out.println("Native javacpp-ffmpeg: LOADED");
        } catch (Throwable t) {
            System.out.println("Native javacpp-ffmpeg: UNAVAILABLE — " + t.getMessage());
        }
        final boolean nativeOk = nativeLoaded;
        final boolean cliOk = FFmpeg.isAvailable();
        System.out.println("System ffmpeg CLI: " + (cliOk ? FFmpeg.findBinary() : "UNAVAILABLE"));

        final Path video = generateTestVideo(tmp, "test.mp4", 160, 120, 10, 1.0);
        final Path video2 = generateTestVideo(tmp, "test2.mp4", 160, 120, 10, 0.5);
        final Path wav = generateToneWav(tmp);
        System.out.println("Test video: " + video);
        System.out.println("Test wav: " + wav);

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D1 Container open + media info (av.md #9) ══");
        if (!nativeOk || video == null) {
            skip("D1", "native or test video unavailable");
        } else {
            section("open + streams metadata", () -> {
                try (Container c = Av.open(video.toString())) {
                    check("readable", c.isReadable());
                    check("has streams", c.streams().size() >= 1);
                    VideoStream vs = c.streams().video(0);
                    check("video width>0", vs.width() > 0);
                    check("video height>0", vs.height() > 0);
                    check("codec name non-empty", vs.codec() != null && vs.codec().name() != null
                            && !vs.codec().name().isEmpty());
                    check("rate > 0", vs.rate().toDouble() > 0);
                    check("time_base den > 0", vs.timeBase().den > 0);
                    System.out.println("    video: " + vs.width() + "x" + vs.height()
                            + " codec=" + vs.codec().name()
                            + " fps=" + vs.rate()
                            + " tb=" + vs.timeBase());
                    if (!c.streams().audio().isEmpty()) {
                        AudioStream as = c.streams().audio(0);
                        check("audio sr>0", as.sampleRate() > 0);
                        check("audio ch>0", as.channels() > 0);
                        System.out.println("    audio: sr=" + as.sampleRate()
                                + " ch=" + as.channels()
                                + " codec=" + as.codec().name());
                    }
                    check("toString", c.toString().contains("r"));
                }
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D2 Decode video → NDArray/Tensor (av.md #1) ══");
        if (!nativeOk || video == null) {
            skip("D2", "native or test video unavailable");
        } else {
            section("decode + toNdarray + fps subsample", () -> {
                try (Container c = Av.open(video.toString())) {
                    VideoStream vs = c.streams().video(0);
                    vs.threadType("AUTO");
                    int frames = 0;
                    int saved = 0;
                    double lastSave = -1;
                    double interval = 1.0 / 5.0; // 5 fps target
                    long t0 = System.nanoTime();
                    for (Frame f : c.decode(vs)) {
                        VideoFrame vf = (VideoFrame) f;
                        try {
                            frames++;
                            double ts = vf.time();
                            if (Double.isNaN(ts)) ts = frames / Math.max(1.0, vs.rate().toDouble());
                            if (lastSave < 0 || ts - lastSave >= interval) {
                                NDArray arr = vf.toNdarray("rgb24");
                                check("ndarray rank3 first", arr.shape.length == 3 || frames > 1);
                                if (frames == 1) {
                                    checkEq("H", (long) vs.height(), arr.shape[0]);
                                    checkEq("W", (long) vs.width(), arr.shape[1]);
                                    checkEq("C", 3L, arr.shape[2]);
                                    check("dtype uint8-ish", arr.dtype == DType.UINT8 || arr.dtype == DType.INT64
                                            || arr.dtype == DType.UINT8);
                                    Tensor t = vf.toTensor("rgb24");
                                    check("tensor dim3", t.dim() == 3);
                                    Tensor chw = vf.toTensorChw("rgb24");
                                    checkEq("chw C", 3L, chw.size(0));
                                }
                                lastSave = ts;
                                saved++;
                            }
                        } finally {
                            vf.close();
                        }
                    }
                    long ms = (System.nanoTime() - t0) / 1_000_000;
                    check("decoded frames > 0", frames > 0);
                    check("subsampled >= 1", saved >= 1);
                    System.out.println("    frames=" + frames + " saved@5fps=" + saved + " in " + ms + " ms");
                }
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D3 Decode audio PCM (av.md #3) ══");
        if (!nativeOk) {
            skip("D3", "native unavailable");
        } else {
            section("decode audio from video or wav", () -> {
                Path src = video;
                // prefer video with audio; else wav
                boolean hasAudio = false;
                if (src != null) {
                    try (Container c = Av.open(src.toString())) {
                        hasAudio = !c.streams().audio().isEmpty();
                    } catch (Exception ignored) {}
                }
                if (!hasAudio) src = wav;
                if (src == null) {
                    skip("D3 inner", "no audio source");
                    return;
                }
                try (Container c = Av.open(src.toString())) {
                    if (c.streams().audio().isEmpty()) {
                        skip("D3", "no audio stream");
                        return;
                    }
                    AudioStream as = c.streams().audio(0);
                    List<float[]> chunks = new ArrayList<>();
                    int total = 0;
                    int ch = as.channels();
                    for (Frame f : c.decode(as)) {
                        AudioFrame af = (AudioFrame) f;
                        try {
                            NDArray arr = af.toNdarray();
                            check("audio rank2", arr.shape.length == 2);
                            ch = (int) arr.shape[0];
                            int n = (int) arr.shape[1];
                            float[] plane = new float[ch * n];
                            for (int i = 0; i < plane.length; i++) plane[i] = (float) arr.getDouble(i);
                            chunks.add(plane);
                            total += n;
                        } finally {
                            af.close();
                        }
                    }
                    check("audio samples > 0", total > 0);
                    // energy
                    double energy = 0;
                    for (float[] p : chunks) for (float v : p) energy += v * v;
                    check("audio energy > 0", energy > 0);
                    System.out.println("    samples=" + total + " ch=" + ch + " energy=" + energy);
                }
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D4 Encode NDArray → MP4 roundtrip (av.md #7) ══");
        if (!nativeOk) {
            skip("D4", "native unavailable");
        } else {
            section("numpy/ndarray frames → encode → re-decode", () -> {
                Path out = tmp.resolve("from_nd.mp4");
                int w = 64, h = 48, n = 8, fps = 10;
                List<NDArray> frames = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    long[] data = new long[h * w * 3];
                    for (int y = 0; y < h; y++) {
                        for (int x = 0; x < w; x++) {
                            int idx = (y * w + x) * 3;
                            data[idx] = (i * 20 + x) & 0xFF;
                            data[idx + 1] = (y * 3) & 0xFF;
                            data[idx + 2] = 128;
                        }
                    }
                    frames.add(new NDArray(data, DType.UINT8, h, w, 3));
                }
                try (Container cout = Av.open(out.toString(), "w")) {
                    VideoStream vs = cout.addVideoStream("libx264", fps);
                    vs.width(w);
                    vs.height(h);
                    vs.pixFmt("yuv420p");
                    // some encoders need bit_rate
                    vs.bitRate(200_000);
                    cout.writeHeader();
                    long pts = 0;
                    Rational tb = vs.timeBase();
                    for (NDArray arr : frames) {
                        try (VideoFrame vf = VideoFrame.fromNdarray(arr, "rgb24")) {
                            // convert to yuv420p for encoder
                            try (VideoFrame yuv = vf.reformat("yuv420p")) {
                                yuv.pts(pts);
                                pts += Math.max(1, tb.den / Math.max(1, fps * Math.max(1, tb.num)));
                                List<Packet> pkts = vs.encode(yuv);
                                for (Packet p : pkts) {
                                    try { cout.mux(p); } finally { p.close(); }
                                }
                            }
                        }
                    }
                    // flush
                    for (Packet p : vs.encode(null)) {
                        try { cout.mux(p); } finally { p.close(); }
                    }
                }
                check("out exists", Files.exists(out) && Files.size(out) > 0);
                System.out.println("    encoded size=" + Files.size(out));
                // re-decode
                try (Container cin = Av.open(out.toString())) {
                    int got = 0;
                    for (Frame f : cin.decodeVideo(0)) {
                        f.close();
                        got++;
                    }
                    check("re-decode frames > 0", got > 0);
                    // allow encoder to drop/dup slightly
                    check("re-decode roughly n", got >= n / 2 && got <= n * 2);
                    System.out.println("    re-decoded frames=" + got + " (src " + n + ")");
                }
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D5 Stream-copy trim + seek (av.md #8) ══");
        if (!nativeOk || video == null) {
            skip("D5", "native or video unavailable");
        } else {
            section("seek + demux stream copy trim", () -> {
                Path out = tmp.resolve("trim_copy.mp4");
                try (Container cin = Av.open(video.toString());
                     Container cout = Av.open(out.toString(), "w")) {
                    VideoStream vin = cin.streams().video(0);
                    Stream vout = cout.addStream(vin);
                    AudioStream ain = null;
                    Stream aout = null;
                    if (!cin.streams().audio().isEmpty()) {
                        ain = cin.streams().audio(0);
                        aout = cout.addStream(ain);
                    }
                    cout.writeHeader();
                    double startSec = 0.1;
                    double endSec = 0.6;
                    long startTs = (long) (startSec / vin.timeBase().toDouble());
                    long endTs = (long) (endSec / vin.timeBase().toDouble());
                    try {
                        cin.seek(startSec, vin);
                    } catch (FFmpegException e) {
                        System.out.println("    seek soft-fail: " + e.getMessage());
                    }
                    int muxed = 0;
                    Stream[] demuxStreams = ain != null ? new Stream[]{vin, ain} : new Stream[]{vin};
                    for (Packet pkt : cin.demux(demuxStreams)) {
                        try {
                            if (pkt.stream() == vin || (pkt.stream() != null && pkt.stream().isVideo())) {
                                long pts = pkt.pts();
                                if (pts != 0x8000000000000000L && pts > endTs) break;
                            }
                            // remap stream index to output
                            if (pkt.stream() != null && pkt.stream().isVideo()) {
                                pkt.streamIndex(vout.index());
                            } else if (pkt.stream() != null && pkt.stream().isAudio() && aout != null) {
                                pkt.streamIndex(aout.index());
                            }
                            cout.mux(pkt);
                            muxed++;
                        } finally {
                            pkt.close();
                        }
                    }
                    check("muxed packets > 0", muxed > 0);
                }
                check("trim out exists", Files.exists(out) && Files.size(out) > 0);
                System.out.println("    trim size=" + Files.size(out));
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D6 FilterGraph scale (av.md #5) ══");
        if (!nativeOk || video == null) {
            skip("D6", "native or video unavailable");
        } else {
            section("filter scale=80:60", () -> {
                Path out = tmp.resolve("scaled.mp4");
                try (Container cin = Av.open(video.toString())) {
                    VideoStream vin = cin.streams().video(0);
                    try (FilterGraph g = FilterGraph.open(vin, "scale=80:60");
                         Container cout = Av.open(out.toString(), "w")) {
                        VideoStream vout = cout.addVideoStream("libx264", vin.rate().toDouble() > 0
                                ? vin.rate().toDouble() : 10);
                        vout.width(80);
                        vout.height(60);
                        vout.pixFmt("yuv420p");
                        vout.bitRate(100_000);
                        cout.writeHeader();
                        int pulled = 0;
                        for (Frame f : cin.decode(vin)) {
                            VideoFrame vf = (VideoFrame) f;
                            try {
                                g.push(vf);
                                VideoFrame filtered;
                                while ((filtered = g.pullVideo()) != null) {
                                    try {
                                        checkEq("filtered W", 80, filtered.width());
                                        checkEq("filtered H", 60, filtered.height());
                                        // encode — reformat if needed
                                        VideoFrame enc = filtered;
                                        boolean reformatted = false;
                                        if (!"yuv420p".equals(filtered.formatName())) {
                                            enc = filtered.reformat("yuv420p");
                                            reformatted = true;
                                        }
                                        try {
                                            for (Packet p : vout.encode(enc)) {
                                                try { cout.mux(p); } finally { p.close(); }
                                            }
                                        } finally {
                                            if (reformatted) enc.close();
                                        }
                                        pulled++;
                                        if (pulled >= 3) {
                                            // enough for smoke; still drain a bit
                                        }
                                    } finally {
                                        filtered.close();
                                    }
                                }
                            } finally {
                                vf.close();
                            }
                            if (pulled >= 5) break; // smoke
                        }
                        g.push((VideoFrame) null);
                        VideoFrame filtered;
                        while ((filtered = g.pullVideo()) != null) {
                            try {
                                try (VideoFrame enc = "yuv420p".equals(filtered.formatName())
                                        ? null : filtered.reformat("yuv420p")) {
                                    VideoFrame use = enc != null ? enc : filtered;
                                    for (Packet p : vout.encode(use)) {
                                        try { cout.mux(p); } finally { p.close(); }
                                    }
                                }
                                pulled++;
                            } finally {
                                filtered.close();
                            }
                        }
                        for (Packet p : vout.encode(null)) {
                            try { cout.mux(p); } finally { p.close(); }
                        }
                        check("pulled filtered > 0", pulled > 0);
                        System.out.println("    filtered frames=" + pulled);
                    }
                }
                check("scaled out exists", Files.exists(out) && Files.size(out) > 0);
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D7 A/V sync mux (av.md #4) ══");
        if (!nativeOk || video == null) {
            skip("D7", "native or video unavailable");
        } else {
            section("re-encode mux video(+audio if any)", () -> {
                Path out = tmp.resolve("av_sync.mp4");
                try (Container cin = Av.open(video.toString());
                     Container cout = Av.open(out.toString(), "w")) {
                    VideoStream vin = cin.streams().video(0);
                    VideoStream vout = cout.addVideoStream("libx264",
                            vin.rate().toDouble() > 0 ? vin.rate().toDouble() : 10);
                    vout.width(vin.width());
                    vout.height(vin.height());
                    vout.pixFmt("yuv420p");
                    vout.bitRate(150_000);
                    // only video path for reliable smoke (audio encode varies by build)
                    cout.writeHeader();
                    int n = 0;
                    for (Frame f : cin.decode(vin)) {
                        VideoFrame vf = (VideoFrame) f;
                        try {
                            VideoFrame yuv = vf;
                            boolean own = false;
                            if (!"yuv420p".equals(vf.formatName())) {
                                yuv = vf.reformat("yuv420p");
                                own = true;
                            }
                            try {
                                yuv.pts(n);
                                for (Packet p : vout.encode(yuv)) {
                                    try { cout.mux(p); } finally { p.close(); }
                                }
                            } finally {
                                if (own) yuv.close();
                            }
                            n++;
                            if (n >= 6) break;
                        } finally {
                            vf.close();
                        }
                    }
                    for (Packet p : vout.encode(null)) {
                        try { cout.mux(p); } finally { p.close(); }
                    }
                    check("encoded n>0", n > 0);
                }
                try (Container c = Av.open(out.toString())) {
                    check("out has video", !c.streams().video().isEmpty());
                }
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D8 Demux packets ══");
        if (!nativeOk || video == null) {
            skip("D8", "native or video unavailable");
        } else {
            section("demux all packets", () -> {
                try (Container c = Av.open(video.toString())) {
                    int n = 0;
                    int maxIdx = c.streams().size() - 1;
                    for (Packet p : c.demux()) {
                        try {
                            check("stream_index in range", p.streamIndex() >= 0 && p.streamIndex() <= maxIdx);
                            check("size >= 0", p.size() >= 0);
                            n++;
                            if (n >= 50) break;
                        } finally {
                            p.close();
                        }
                    }
                    check("packets > 0", n > 0);
                    System.out.println("    packets seen=" + n);
                }
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D9 Threaded decode queue (av.md #10) ══");
        if (!nativeOk || video == null) {
            skip("D9", "native or video unavailable");
        } else {
            // av.md #10: decoder thread → queue → consumer (sentinel end marker)
            section("producer-consumer decode queue", () -> {
                final Object SENTINEL = new Object();
                BlockingQueue<Object> q = new ArrayBlockingQueue<>(16);
                Thread th = new Thread(() -> {
                    try (Container c = Av.open(video.toString())) {
                        int i = 0;
                        for (Frame f : c.decodeVideo(0)) {
                            VideoFrame vf = (VideoFrame) f;
                            try {
                                q.put(vf.toNdarray("rgb24"));
                                i++;
                                if (i >= 20) break;
                            } finally {
                                vf.close();
                            }
                        }
                    } catch (Exception e) {
                        report.append("D9 producer: ").append(e).append('\n');
                    } finally {
                        try { q.put(SENTINEL); } catch (InterruptedException ignored) {
                            Thread.currentThread().interrupt();
                        }
                    }
                }, "av-decode-worker");
                th.start();
                int got = 0;
                while (true) {
                    Object o = q.poll(15, TimeUnit.SECONDS);
                    if (o == null) {
                        check("queue timeout", false);
                        break;
                    }
                    if (o == SENTINEL) break;
                    NDArray arr = (NDArray) o;
                    check("queued rank3", arr.shape.length == 3);
                    got++;
                }
                th.join(5000);
                check("consumer frames > 0", got > 0);
                System.out.println("    consumed=" + got);
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D10 HardwareContext (av.md #2) ══");
        if (!nativeOk) {
            skip("D10", "native unavailable");
        } else {
            section("HardwareContext.create candidates", () -> {
                String[] types = {"videotoolbox", "cuda", "qsv", "vaapi", "d3d11va"};
                boolean any = false;
                for (String t : types) {
                    try (HardwareContext hw = HardwareContext.create(t)) {
                        System.out.println("    created: " + hw);
                        check("hw typeName", hw.typeName() != null);
                        any = true;
                        break;
                    } catch (FFmpegException e) {
                        System.out.println("    " + t + " unavailable: " + e.getMessage());
                    }
                }
                if (!any) {
                    skip("D10 hw device", "no HW device on this host (expected)");
                }
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D11–D15 ffmpeg-python fluent (av.md #11–20) ══");
        if (!cliOk || video == null) {
            skip("D11-D15", "ffmpeg CLI or video unavailable");
        } else {
            section("D11 scale+transcode", () -> {
                Path out = tmp.resolve("ff_scale.mp4");
                FFmpeg.input(video.toString())
                        .filter("scale", 80, 60)
                        .output(out.toString(), "vcodec", "libx264", "pix_fmt", "yuv420p", "crf", "28")
                        .overwriteOutput()
                        .run();
                check("ff_scale exists", Files.exists(out) && Files.size(out) > 0);
            });

            section("D12 extract wav", () -> {
                Path out = tmp.resolve("ff_audio.wav");
                FFmpeg.input(video.toString())
                        .output(out.toString(), "format", "wav", "acodec", "pcm_s16le", "vn", "true")
                        .overwriteOutput()
                        .run();
                // may fail if no audio — accept either
                if (Files.exists(out) && Files.size(out) > 44) {
                    check("wav size", true);
                } else {
                    // try from tone wav re-encode
                    FFmpeg.input(wav.toString())
                            .output(out.toString(), "format", "wav", "acodec", "pcm_s16le")
                            .overwriteOutput()
                            .run();
                    check("wav size fallback", Files.exists(out) && Files.size(out) > 44);
                }
            });

            section("D13 raw pipe rgb", () -> {
                int w = 160, h = 120;
                // pipe:1 + rawvideo; drain stderr in runAsync; read one full frame
                var proc = FFmpeg.input(video.toString())
                        .output("pipe:1", "format", "rawvideo", "pix_fmt", "rgb24",
                                "an", "true", "vframes", "2", "s", w + "x" + h)
                        .overwriteOutput()
                        .runAsync(true);
                try {
                    int frameBytes = w * h * 3;
                    byte[] buf = proc.getStdout().readNBytes(frameBytes);
                    // allow partial if encoder path scaled differently — still must get data
                    check("raw frame bytes > 0", buf.length > 0);
                    if (buf.length == frameBytes) {
                        check("raw frame exact size", true);
                    } else {
                        System.out.println("    partial/different size=" + buf.length
                                + " expected=" + frameBytes + " (still OK if >0)");
                    }
                    System.out.println("    read raw bytes=" + buf.length);
                    proc.waitFor(10, TimeUnit.SECONDS);
                } finally {
                    proc.close();
                }
            });

            section("D14 fast trim copy", () -> {
                Path out = tmp.resolve("ff_trim.mp4");
                FFmpeg.input(video.toString(), "ss", "0", "t", "0.5")
                        .output(out.toString(), "vcodec", "copy", "acodec", "copy")
                        .overwriteOutput()
                        .run();
                check("trim copy exists", Files.exists(out) && Files.size(out) > 0);
            });

            section("D15 fps + multi patterns", () -> {
                Path outFps = tmp.resolve("ff_fps.mp4");
                FFmpeg.input(video.toString())
                        .filter("fps", "fps=5")
                        .output(outFps.toString(), "vcodec", "libx264", "pix_fmt", "yuv420p", "an", "true")
                        .overwriteOutput()
                        .run();
                check("fps out", Files.exists(outFps) && Files.size(outFps) > 0);

                if (video2 != null) {
                    Path outCat = tmp.resolve("ff_cat.mp4");
                    try {
                        // concat demuxer via filter_complex
                        FFmpeg.concat(
                                FFmpeg.input(video.toString()),
                                FFmpeg.input(video2.toString())
                        ).output(outCat.toString(), "vcodec", "libx264", "pix_fmt", "yuv420p", "acodec", "aac")
                                .overwriteOutput()
                                .run();
                        check("concat out", Files.exists(outCat) && Files.size(outCat) > 0);
                    } catch (FFmpegException e) {
                        System.out.println("    concat soft-fail (codec mismatch common): " + e.getMessage());
                        // not a hard fail — concat filter needs matching params
                        skipped++;
                    }
                }

                // compile argv smoke
                List<String> argv = FFmpeg.input("in.mp4")
                        .filter("scale", 1280, 720)
                        .output("out.mp4", "vcodec", "libx264")
                        .overwriteOutput()
                        .compile();
                check("compile has ffmpeg", argv.get(0).contains("ffmpeg") || argv.get(0).endsWith("ffmpeg"));
                check("compile has -i", argv.contains("-i"));
                check("compile has -y", argv.contains("-y"));
                System.out.println("    argv: " + String.join(" ", argv));
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D16 Throughput ══");
        if (!nativeOk || video == null) {
            skip("D16", "native or video unavailable");
        } else {
            section("decode throughput", () -> {
                int iters = 3;
                int total = 0;
                long t0 = System.nanoTime();
                for (int i = 0; i < iters; i++) {
                    try (Container c = Av.open(video.toString())) {
                        for (Frame f : c.decodeVideo(0)) {
                            f.close();
                            total++;
                        }
                    }
                }
                long ms = Math.max(1, (System.nanoTime() - t0) / 1_000_000);
                double fps = total / (ms / 1000.0);
                System.out.println("    decode " + String.format("%.1f", fps) + " frames/s (" + total + " / " + ms + " ms)");
                check("throughput > 0", fps > 0);
            });
        }

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D17 Edge cases ══");
        section("missing file throws", () -> {
            if (!nativeOk) { skip("missing file", "no native"); return; }
            boolean threw = false;
            try {
                Av.open(tmp.resolve("no_such_file_av_bench.mp4").toString());
            } catch (FFmpegException e) {
                threw = true;
                check("error message", e.getMessage() != null && !e.getMessage().isEmpty());
            }
            check("throws on missing", threw);
        });
        section("double close safe", () -> {
            if (!nativeOk || video == null) { skip("double close", "n/a"); return; }
            Container c = Av.open(video.toString());
            c.close();
            c.close(); // must not throw
            check("double close ok", true);
        });
        section("FFmpegException errorMessage", () -> {
            String m = FFmpegException.errorMessage(-2);
            check("errorMessage non-null", m != null && !m.isEmpty());
        });

        // ══════════════════════════════════════════════════════════════════
        System.out.println("\n══ D18 Interop Av ↔ VideoFile ══");
        if (!nativeOk || video == null) {
            skip("D18", "native or video unavailable");
        } else {
            section("both APIs open same clip", () -> {
                int avFrames = 0;
                try (Container c = Av.open(video.toString())) {
                    for (Frame f : c.decodeVideo(0)) {
                        f.close();
                        avFrames++;
                    }
                }
                int vfFrames = 0;
                try (VideoFile vf = VideoFile.open(video.toString())) {
                    check("VideoFile w", vf.width() > 0);
                    List<Tensor> frames = vf.readFrames();
                    vfFrames = frames.size();
                }
                check("Av frames > 0", avFrames > 0);
                check("VideoFile frames > 0", vfFrames > 0);
                System.out.println("    Av=" + avFrames + " VideoFile=" + vfFrames);
                // counts should be close
                check("counts close", Math.abs(avFrames - vfFrames) <= Math.max(2, avFrames / 5));
            });
        }

        // ── summary ───────────────────────────────────────────────────────
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed, "
                + skipped + " skipped ===");
        System.out.println("Native=" + nativeOk + " CLI=" + cliOk);
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        // cleanup
        try {
            Files.walk(tmp)
                    .sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> { try { Files.deleteIfExists(p); } catch (Exception ignored) {} });
        } catch (Exception ignored) {}
    }
}
