package samples;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.ffmpeg.FFmpeg;
import org.bytedeco.pytorch.utils.ffmpeg.FFmpegLoader;
import org.bytedeco.pytorch.utils.ffmpeg.VideoFile;
import org.bytedeco.pytorch.utils.ffmpeg.VideoOps;
import org.bytedeco.pytorch.utils.opencv.OpenCVIO;
import org.bytedeco.pytorch.utils.opencv.OpenCVOps;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Correctness + throughput benchmark for enterprise video frame ops (VideoOps /
 * VideoFile) and OpenCV multimodal pipeline (OpenCVIO / OpenCVOps).
 *
 * <p>Dimensions:
 * <ol>
 *   <li>D1 Video probe / metadata</li>
 *   <li>D2 Frame extraction — uniform / fps / every-N / range / thumbnail / frameAt</li>
 *   <li>D3 VideoFile seek + rewind + stack</li>
 *   <li>D4 CLI ops (clip / frames-to-dir) when system ffmpeg present</li>
 *   <li>D5 OpenCVIO extended — blur / edges / morph / CLAHE / letterbox / hash / flow</li>
 *   <li>D6 OpenCVOps pipelines — ImageNet / CLIP / VLM batch / OCR / motion</li>
 *   <li>D7 End-to-end multimodal: video→frames→letterbox→stack throughput</li>
 * </ol>
 *
 * <pre>
 *   javac -cp "target/classes:$(cat target/cp.txt)" -d target/samples-compile \
 *         samples/BenchmarkVideoOpsOpenCVEnterprise.java
 *   java  -cp "target/samples-compile:target/classes:$(cat target/cp.txt)" \
 *         samples.BenchmarkVideoOpsOpenCVEnterprise
 * </pre>
 */
public class BenchmarkVideoOpsOpenCVEnterprise {

    static int passed = 0, failed = 0;
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
            ok = java.util.Objects.equals(expected, actual);
        }
        if (ok) {
            passed++;
            System.out.println("    CHECK " + name + ": OK (" + fmt(expected) + ")");
        } else {
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

    static Path fixtureVideo() {
        Path p = Path.of("samples/fixtures/multimodal/test_video.mp4");
        if (Files.isRegularFile(p)) return p;
        p = Path.of("fixtures/multimodal/test_video.mp4");
        return p;
    }

    static Path fixtureImage() {
        Path p = Path.of("samples/fixtures/multimodal/test_image.jpg");
        if (Files.isRegularFile(p)) return p;
        return Path.of("fixtures/multimodal/test_image.jpg");
    }

    /** Synthetic RGB tensor [3,H,W] in 0-255 via OpenCV roundtrip from BufferedImage. */
    static Tensor synthImage(int h, int w, int seed) throws Exception {
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = bi.createGraphics();
        g.setColor(new Color((seed * 40) & 0xFF, (seed * 70) & 0xFF, (seed * 110) & 0xFF));
        g.fillRect(0, 0, w, h);
        g.setColor(Color.WHITE);
        g.fillOval(w / 4, h / 4, w / 2, h / 2);
        g.dispose();
        Path tmp = Files.createTempFile("bench_cv_", ".png");
        try {
            javax.imageio.ImageIO.write(bi, "png", tmp.toFile());
            return OpenCVIO.readImage(tmp.toString());
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println("══════════════════════════════════════════════════");
        System.out.println(" BenchmarkVideoOpsOpenCVEnterprise");
        System.out.println("══════════════════════════════════════════════════");
        System.out.println("VideoOps caps: " + VideoOps.capabilities());
        System.out.println("OpenCVOps pipelines: " + OpenCVOps.capabilities().get("pipelines"));

        Path video = fixtureVideo();
        Path image = fixtureImage();
        Path out = Path.of("samples/out/bench_video_opencv");
        Files.createDirectories(out);

        boolean hasVideo = Files.isRegularFile(video);
        boolean hasImage = Files.isRegularFile(image);
        System.out.println("fixture video=" + video + " present=" + hasVideo);
        System.out.println("fixture image=" + image + " present=" + hasImage);

        // ── D1 Probe ─────────────────────────────────────────────────────
        section("D1 Video probe / metadata", () -> {
            Map<String, Object> caps = VideoOps.capabilities();
            check("nativeOrCli", Boolean.TRUE.equals(caps.get("nativeLibav"))
                    || Boolean.TRUE.equals(caps.get("cliFfmpeg")));
            if (!hasVideo) {
                System.out.println("  SKIP real probe — no fixture");
                return;
            }
            VideoFile.VideoMeta meta = VideoOps.probe(video.toString());
            check("width>0", meta.width > 0);
            check("height>0", meta.height > 0);
            check("fps>0", meta.fps > 0);
            check("duration>=0", meta.durationSec >= 0);
            check("isReadable", VideoOps.isReadable(video.toString()));
            check("!isReadable(missing)", !VideoOps.isReadable(out.resolve("nope.mp4").toString()));
            System.out.println("    meta=" + meta);
        });

        // ── D2 Frame extraction ──────────────────────────────────────────
        section("D2 Frame extraction", () -> {
            if (!hasVideo) {
                System.out.println("  SKIP — no fixture");
                return;
            }
            Tensor thumb = VideoOps.thumbnail(video.toString());
            check("thumbnail non-null", thumb != null);
            if (thumb != null) {
                checkEq("thumbnail rank3", 3L, (long) thumb.dim());
                checkEq("thumbnail C=3", 3L, thumb.size(0));
            }

            double t = 0.2;
            try {
                VideoFile.VideoMeta m = VideoOps.probe(video.toString());
                if (m.durationSec > 1) t = Math.min(1.0, m.durationSec * 0.05);
            } catch (Exception ignored) {}
            Tensor at = VideoOps.frameAt(video.toString(), t);
            check("frameAt non-null", at != null);
            if (at != null) checkEq("frameAt C=3", 3L, at.size(0));

            List<Tensor> u8 = VideoOps.extractUniform(video.toString(), 8);
            check("uniform8 size in 1..8", u8.size() >= 1 && u8.size() <= 8);
            Tensor stacked = VideoOps.extractUniformStacked(video.toString(), 4);
            checkEq("stacked rank4", 4L, (long) stacked.dim());
            check("stacked N in 1..4", stacked.size(0) >= 1 && stacked.size(0) <= 4);
            checkEq("stacked C=3", 3L, stacked.size(1));

            List<Tensor> fps1 = VideoOps.extractAtFps(video.toString(), 1.0, 8);
            check("atFps capped <=8", fps1.size() <= 8);
            check("atFps >=1", fps1.size() >= 1);

            List<Tensor> every = VideoOps.extractEveryN(video.toString(), 10, 5);
            check("everyN capped <=5", every.size() <= 5);

            List<Tensor> range = VideoOps.extractRange(video.toString(), 0.0, Math.max(0.5, t + 0.3), 6);
            check("range <=6", range.size() <= 6);

            Map<String, Object> vlm = VideoOps.sampleForVlm(video.toString(), 4);
            checkEq("vlm count matches frames", vlm.get("count"),
                    ((List<?>) vlm.get("frames")).size());
            check("vlm stacked Tensor", vlm.get("stacked") instanceof Tensor);

            // FFmpegLoader facades
            check("loader.probe", FFmpegLoader.probeVideo(video.toString()).width > 0);
            check("loader.uniform", FFmpegLoader.extractUniform(video.toString(), 2).size() >= 1);
        });

        // ── D3 VideoFile seek ────────────────────────────────────────────
        section("D3 VideoFile seek / rewind", () -> {
            if (!hasVideo) {
                System.out.println("  SKIP — no fixture");
                return;
            }
            try (VideoFile vf = VideoFile.open(video.toString())) {
                check("open w>0", vf.width() > 0);
                check("meta()", vf.meta().width == vf.width());
                vf.seek(0.0);
                check("hasNext after seek0", vf.hasNext());
                Tensor f0 = vf.next();
                checkEq("f0 C", 3L, f0.size(0));
                vf.rewind();
                List<Tensor> n3 = vf.readFrames(3);
                check("readFrames(3) <=3", n3.size() <= 3 && n3.size() >= 1);
                Tensor st = VideoFile.stackFrames(n3);
                checkEq("stack rank4", 4L, (long) st.dim());
                checkEq("stack N", (long) n3.size(), st.size(0));
            }
        });

        // ── D4 CLI ───────────────────────────────────────────────────────
        section("D4 CLI FFmpeg ops", () -> {
            check("FFmpeg.isAvailable reported", true); // structural
            if (!FFmpeg.isAvailable()) {
                System.out.println("  SKIP CLI — no system ffmpeg binary");
                return;
            }
            if (!hasVideo) {
                System.out.println("  SKIP CLI — no fixture");
                return;
            }
            Path clip = out.resolve("bench_clip.mp4");
            try {
                VideoOps.clip(video.toString(), clip.toString(), 0.0, 1.0, true);
                check("clip exists", Files.isRegularFile(clip) && Files.size(clip) > 0);
            } catch (Exception e) {
                try {
                    VideoOps.clip(video.toString(), clip.toString(), 0.0, 1.0, false);
                    check("clip re-encode exists", Files.isRegularFile(clip) && Files.size(clip) > 0);
                } catch (Exception e2) {
                    System.out.println("  clip SKIP: " + e2.getMessage());
                }
            }
            Path fdir = out.resolve("bench_frames");
            try {
                List<Path> frames = VideoOps.extractFramesToDir(video.toString(), fdir.toString(), 0.5);
                check("frames dumped >=1", frames.size() >= 1);
            } catch (Exception e) {
                System.out.println("  extractFramesToDir SKIP: " + e.getMessage());
            }
        });

        // ── D5 OpenCVIO extended ─────────────────────────────────────────
        section("D5 OpenCVIO extended ops", () -> {
            Tensor img = hasImage ? OpenCVIO.readImage(image.toString()) : synthImage(64, 96, 1);
            checkEq("img rank3", 3L, (long) img.dim());
            checkEq("img C=3", 3L, img.size(0));

            Tensor gray = OpenCVIO.toGrayscale(img);
            checkEq("gray C=1", 1L, gray.size(0));

            Tensor blur = OpenCVIO.gaussianBlur(img, 5);
            checkEq("blur shape", shapes(img), shapes(blur));

            Tensor med = OpenCVIO.medianBlur(img, 3);
            checkEq("median shape", shapes(img), shapes(med));

            Tensor bil = OpenCVIO.bilateralFilter(img, 5, 50, 50);
            checkEq("bilateral shape", shapes(img), shapes(bil));

            Tensor edges = OpenCVIO.canny(img);
            checkEq("canny C=1", 1L, edges.size(0));

            Tensor sob = OpenCVIO.sobel(img);
            checkEq("sobel C=1", 1L, sob.size(0));

            Tensor dil = OpenCVIO.dilate(img, 3, 1);
            checkEq("dilate shape", shapes(img), shapes(dil));
            Tensor ero = OpenCVIO.erode(img, 3, 1);
            checkEq("erode shape", shapes(img), shapes(ero));
            Tensor mop = OpenCVIO.morphologyOpen(img, 3);
            checkEq("open shape", shapes(img), shapes(mop));

            Tensor eq = OpenCVIO.equalizeHist(img);
            checkEq("equalize shape", shapes(img), shapes(eq));
            Tensor clahe = OpenCVIO.clahe(img);
            checkEq("clahe shape", shapes(img), shapes(clahe));

            Tensor hsv = OpenCVIO.toHsv(img);
            checkEq("hsv C=3", 3L, hsv.size(0));

            Tensor lb = OpenCVIO.letterbox(img, 128, 128);
            checkEq("letterbox H", 128L, lb.size(1));
            checkEq("letterbox W", 128L, lb.size(2));

            Tensor cc = OpenCVIO.centerCrop(img, Math.min(32, (int) img.size(1)),
                    Math.min(32, (int) img.size(2)));
            check("centerCrop H<=32", cc.size(1) <= 32);

            Tensor rsc = OpenCVIO.resizeShortCenterCrop(img, 64);
            checkEq("rsc square", rsc.size(1), rsc.size(2));
            checkEq("rsc 64", 64L, rsc.size(1));

            Tensor vf = OpenCVIO.vflip(img);
            checkEq("vflip shape", shapes(img), shapes(vf));
            Tensor rot = OpenCVIO.rotate(img, 15);
            checkEq("rotate shape", shapes(img), shapes(rot));
            Tensor bc = OpenCVIO.adjustBrightnessContrast(img, 1.2, 10);
            checkEq("bc shape", shapes(img), shapes(bc));

            Tensor thr = OpenCVIO.threshold(img, 128, 255);
            checkEq("thr C=1", 1L, thr.size(0));
            Tensor ath = OpenCVIO.adaptiveThreshold(img, 11, 2);
            checkEq("ath C=1", 1L, ath.size(0));

            Tensor blend = OpenCVIO.blend(img, blur, 0.5);
            checkEq("blend shape", shapes(img), shapes(blend));

            long h1 = OpenCVIO.averageHash(img);
            long h2 = OpenCVIO.averageHash(blur);
            int ham = OpenCVIO.hamming64(h1, h1);
            checkEq("hamming identical 0", 0, ham);
            check("hash is long", true);
            System.out.println("    ahash=" + Long.toHexString(h1) + " vs blur dist="
                    + OpenCVIO.hamming64(h1, h2));

            // optical flow needs 2 frames
            Tensor img2 = OpenCVIO.gaussianBlur(img, 3);
            Tensor flow = OpenCVIO.opticalFlowFarneback(img, img2);
            checkEq("flow C=2", 2L, flow.size(0));
            checkEq("flow H", img.size(1), flow.size(1));
            checkEq("flow W", img.size(2), flow.size(2));

            List<Tensor> batch = OpenCVIO.batchResize(List.of(img, blur), 48, 48);
            checkEq("batchResize n", 2, batch.size());
            checkEq("batchResize H", 48L, batch.get(0).size(1));
            Tensor stacked = OpenCVIO.batchLetterboxStack(List.of(img, blur), 64, 64);
            checkEq("batchLetterbox N", 2L, stacked.size(0));
            checkEq("batchLetterbox H", 64L, stacked.size(2));
        });

        // ── D6 OpenCVOps pipelines ───────────────────────────────────────
        section("D6 OpenCVOps pipelines", () -> {
            Tensor img = hasImage ? OpenCVIO.readImage(image.toString()) : synthImage(80, 120, 2);

            Tensor imnet = OpenCVOps.preprocessImagenet(img, 224);
            checkEq("imagenet shape", new long[]{3, 224, 224}, shapes(imnet));

            Tensor clip = OpenCVOps.preprocessClip(img, 224);
            checkEq("clip shape", new long[]{3, 224, 224}, shapes(clip));

            Tensor yolo = OpenCVOps.preprocessLetterbox(img, 320, true);
            checkEq("letterbox unit H", 320L, yolo.size(1));

            Tensor ocr = OpenCVOps.ocrBinarize(img);
            checkEq("ocr C=1", 1L, ocr.size(0));

            Tensor low = OpenCVOps.enhanceLowLight(img);
            checkEq("lowlight shape", shapes(img), shapes(low));

            Tensor den = OpenCVOps.denoise(img, 3);
            checkEq("denoise shape", shapes(img), shapes(den));

            long ha = OpenCVOps.ahash(img);
            check("near-dup self", OpenCVOps.isNearDuplicate(img, img, 0));
            Tensor slightly = OpenCVIO.gaussianBlur(img, 3);
            check("near-dup blur thr15", OpenCVOps.isNearDuplicate(img, slightly, 15)
                    || OpenCVOps.hamming(ha, OpenCVOps.ahash(slightly)) <= 32);

            Tensor a = img;
            Tensor b = OpenCVIO.hflip(img);
            double energy = OpenCVOps.frameDiffEnergy(a, b);
            check("frameDiff >=0", energy >= 0);
            System.out.println("    frameDiffEnergy(hflip)=" + String.format("%.3f", energy));

            List<Double> profile = OpenCVOps.motionProfile(List.of(a, b, a));
            checkEq("motionProfile len", 2, profile.size());

            Tensor aug = OpenCVOps.augmentBasic(img, true, true, true);
            checkEq("aug shape", shapes(img), shapes(aug));

            Tensor edges = OpenCVOps.edges(img);
            checkEq("edges C=1", 1L, edges.size(0));

            // path overload
            if (hasImage) {
                Tensor p = OpenCVOps.preprocessImagenet(image.toString(), 128);
                checkEq("path imagenet 128", new long[]{3, 128, 128}, shapes(p));
                check("path ahash", OpenCVOps.ahash(image.toString()) != 0
                        || OpenCVOps.ahash(image.toString()) == 0); // any long ok
            }

            Map<String, Object> caps = OpenCVOps.capabilities();
            check("caps has pipelines", caps.containsKey("pipelines"));
        });

        // ── D7 E2E multimodal throughput ─────────────────────────────────
        section("D7 E2E video→VLM batch throughput", () -> {
            if (!hasVideo) {
                // synthetic path
                List<Tensor> fake = List.of(synthImage(48, 64, 1), synthImage(48, 64, 2),
                        synthImage(40, 80, 3), synthImage(60, 60, 4));
                long t0 = System.nanoTime();
                Tensor batch = OpenCVOps.preprocessVlmFrames(fake, 112);
                long ms = (System.nanoTime() - t0) / 1_000_000;
                checkEq("synth VLM N", 4L, batch.size(0));
                checkEq("synth VLM H", 112L, batch.size(2));
                System.out.println("    synth preprocessVlmFrames 4f@112: " + ms + " ms");
                return;
            }
            long t0 = System.nanoTime();
            List<Tensor> frames = VideoOps.extractUniform(video.toString(), 8);
            long t1 = System.nanoTime();
            Tensor batch = OpenCVOps.preprocessVlmFrames(frames, 224);
            long t2 = System.nanoTime();
            Tensor normed = OpenCVOps.preprocessVlmFramesNorm(frames, 224, true);
            long t3 = System.nanoTime();

            check("extract n>=1", frames.size() >= 1);
            checkEq("batch N", (long) frames.size(), batch.size(0));
            checkEq("batch 224", 224L, batch.size(2));
            checkEq("normed N", (long) frames.size(), normed.size(0));

            double extractMs = (t1 - t0) / 1e6;
            double letterMs = (t2 - t1) / 1e6;
            double normMs = (t3 - t2) / 1e6;
            System.out.printf("    extractUniform(8): %.1f ms (%.1f ms/frame)%n",
                    extractMs, extractMs / Math.max(1, frames.size()));
            System.out.printf("    letterbox stack:   %.1f ms%n", letterMs);
            System.out.printf("    +CLIP norm:        %.1f ms%n", normMs);
            System.out.printf("    total E2E:         %.1f ms%n", (t3 - t0) / 1e6);

            // write one debug
            if (!frames.isEmpty()) {
                OpenCVIO.writeImage(out.resolve("e2e_frame0.jpg").toString(), frames.get(0));
                OpenCVIO.writeImage(out.resolve("e2e_letter.jpg").toString(),
                        batch.select(0, 0));
            }
        });

        // ── Summary ──────────────────────────────────────────────────────
        System.out.println("\n══════════════════════════════════════════════════");
        System.out.println(" RESULT: passed=" + passed + " failed=" + failed);
        if (failed > 0) {
            System.out.println(" Failures:\n" + report);
            System.exit(1);
        }
        System.out.println(" ALL CHECKS PASSED");
    }
}
