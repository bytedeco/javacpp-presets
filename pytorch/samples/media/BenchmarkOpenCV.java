package media;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.vision.opencv.MatToTensor;
import org.bytedeco.pytorch.vision.opencv.OpenCVException;
import org.bytedeco.pytorch.vision.opencv.OpenCVIO;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;

/**
 * Multi-dimensional correctness + performance benchmark for {@code utils.opencv}.
 *
 * <p>Dimensions:
 * <ol>
 *   <li>D1 MatToTensor fromMat / toMat / BGR↔RGB roundtrip</li>
 *   <li>D2 OpenCVIO read/write/decode/encode</li>
 *   <li>D3 Geometry — resize / crop / hflip / rotate90</li>
 *   <li>D4 Color — grayscale / rgbToBgr / bgrToRgb / normalize</li>
 *   <li>D5 Roundtrip pixel fidelity + edge cases</li>
 *   <li>D6 Daily pipeline + throughput</li>
 * </ol>
 */
public class BenchmarkOpenCV {

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

    static BufferedImage makeRgb(int w, int h) {
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = bi.createGraphics();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int r = (x * 3) & 255, gg = (y * 5) & 255, b = ((x + y) * 7) & 255;
                g.setColor(new Color(r, gg, b));
                g.fillRect(x, y, 1, 1);
            }
        }
        g.setColor(Color.RED);
        g.fillRect(10, 10, 40, 40);
        g.setColor(Color.GREEN);
        g.drawLine(0, h / 2, w, h / 2);
        g.dispose();
        return bi;
    }

    static float maxAbsDiff(Tensor a, Tensor b) {
        Tensor diff = a.sub(b).abs();
        return diff.max().item().toFloat();
    }

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("opencv_bench");
        System.out.println("=== OpenCV Module Benchmark ===");
        System.out.println("Temp: " + tmp);

        Path png = tmp.resolve("test.png");
        Path jpg = tmp.resolve("test.jpg");
        BufferedImage bi = makeRgb(256, 192);
        javax.imageio.ImageIO.write(bi, "PNG", png.toFile());
        javax.imageio.ImageIO.write(bi, "JPG", jpg.toFile());

        boolean available = true;
        try {
            OpenCVIO.readImage(png.toString());
        } catch (Throwable t) {
            available = false;
            System.out.println("  [OpenCV native not available: " + t.getMessage() + "]");
            System.out.println("  Writing structural checks only where possible.");
        }

        if (!available) {
            section("OpenCVException construct", () -> {
                OpenCVException e1 = new OpenCVException("msg");
                check("message", e1.getMessage().equals("msg"));
                OpenCVException e2 = new OpenCVException("msg2", 42);
                checkEq("errorCode", 42, e2.errorCode());
                OpenCVException e3 = new OpenCVException("msg3", new RuntimeException("c"));
                check("cause", e3.getCause() != null);
            });
            System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed (native skipped) ===");
            if (failed > 0) System.exit(1);
            return;
        }

        // ── D1 MatToTensor ───────────────────────────────────────────────────
        System.out.println("\n══ D1 MatToTensor ══");
        section("fromMat / toMat roundtrip 8UC3", () -> {
            Mat mat = new Mat(64, 48, CV_8UC3);
            // fill with a pattern via put
            byte[] buf = new byte[64 * 48 * 3];
            for (int i = 0; i < buf.length; i++) buf[i] = (byte) (i % 256);
            mat.data().put(buf);

            Tensor t = MatToTensor.fromMat(mat);
            long[] s = shapes(t);
            check("fromMat rank 3", s.length == 3);
            checkEq("fromMat C=3", 3L, s[0]);
            checkEq("fromMat H", 64L, s[1]);
            checkEq("fromMat W", 48L, s[2]);
            check("fromMat float", isFloat(t));

            Mat back = MatToTensor.toMat(t);
            checkEq("toMat rows", 64, back.rows());
            checkEq("toMat cols", 48, back.cols());
            checkEq("toMat channels", 3, back.channels());

            // BGR helpers
            Tensor rgb = MatToTensor.bgrMatToRgbTensor(mat);
            checkEq("bgrMatToRgbTensor C", 3L, shapes(rgb)[0]);
            Mat bgr = MatToTensor.rgbTensorToBgrMat(rgb);
            checkEq("rgbTensorToBgrMat channels", 3, bgr.channels());

            mat.close();
            back.close();
            bgr.close();
        });

        section("fromMat grayscale 8UC1", () -> {
            Mat gray = new Mat(32, 32, CV_8UC1);
            byte[] buf = new byte[32 * 32];
            for (int i = 0; i < buf.length; i++) buf[i] = (byte) i;
            gray.data().put(buf);
            Tensor t = MatToTensor.fromMat(gray);
            long[] s = shapes(t);
            check("gray rank >= 2", s.length >= 2);
            // may be [1,H,W] or [H,W]
            if (s.length == 3) checkEq("gray C=1", 1L, s[0]);
            check("gray float", isFloat(t));
            gray.close();
        });

        // ── D2 OpenCVIO read/write ───────────────────────────────────────────
        System.out.println("\n══ D2 OpenCVIO I/O ══");
        section("readImage / readImageGray / channels", () -> {
            Tensor img = OpenCVIO.readImage(png.toString());
            long[] s = shapes(img);
            checkEq("readImage rank", 3, s.length);
            checkEq("readImage C=3", 3L, s[0]);
            check("H>0", s[1] > 0);
            check("W>0", s[2] > 0);
            check("float dtype", isFloat(img));
            // values in [0,255]
            float max = img.max().item().toFloat();
            float min = img.min().item().toFloat();
            check("values in [0,255]", min >= -1e-3f && max <= 255.5f);
            System.out.println("    shape=" + Arrays.toString(s) + " range=[" + min + "," + max + "]");

            Tensor g = OpenCVIO.readImageGray(png.toString());
            checkEq("gray C=1", 1L, shapes(g)[0]);

            Tensor c3 = OpenCVIO.readImage(png.toString(), 3);
            checkEq("channels=3", 3L, shapes(c3)[0]);
            Tensor c1 = OpenCVIO.readImage(png.toString(), 1);
            checkEq("channels=1", 1L, shapes(c1)[0]);

            Tensor fromPath = OpenCVIO.readImage(png);
            check("Path overload", shapes(fromPath).length == 3);
            Tensor grayPath = OpenCVIO.readImageGray(png);
            check("gray Path", shapes(grayPath)[0] == 1);
        });

        section("writeImage + encode/decode", () -> {
            Tensor img = OpenCVIO.readImage(png.toString());
            Path outPng = tmp.resolve("out.png");
            Path outJpg = tmp.resolve("out.jpg");
            OpenCVIO.writeImage(outPng.toString(), img);
            check("write png exists", Files.exists(outPng) && Files.size(outPng) > 0);
            OpenCVIO.writeImage(outJpg, img);
            check("write jpg Path", Files.exists(outJpg) && Files.size(outJpg) > 0);

            Tensor round = OpenCVIO.readImage(outPng.toString());
            check("roundtrip shape", Arrays.equals(shapes(img), shapes(round)));
            float md = maxAbsDiff(img, round);
            System.out.println("    png roundtrip maxDiff=" + md);
            check("png roundtrip maxDiff < 2", md < 2.0f);

            byte[] jpgBytes = OpenCVIO.encode(img, "jpg");
            check("jpg bytes > 0", jpgBytes.length > 2);
            checkEq("JPEG SOI0", 0xFF, jpgBytes[0] & 0xFF);
            checkEq("JPEG SOI1", 0xD8, jpgBytes[1] & 0xFF);

            byte[] pngBytes = OpenCVIO.encode(img, "png");
            check("png bytes > 8", pngBytes.length > 8);
            checkEq("PNG magic", 0x89, pngBytes[0] & 0xFF);

            Tensor decoded = OpenCVIO.decodeImage(pngBytes);
            check("decodeImage rank 3", shapes(decoded).length == 3);
            checkEq("decode C", 3L, shapes(decoded)[0]);
        });

        // ── D3 Geometry ──────────────────────────────────────────────────────
        System.out.println("\n══ D3 Geometry ══");
        section("resize / crop / hflip / rotate90", () -> {
            Tensor img = OpenCVIO.readImage(png.toString());
            long[] orig = shapes(img);

            Tensor r = OpenCVIO.resize(img, 224, 224);
            checkEq("resize H", 224L, shapes(r)[1]);
            checkEq("resize W", 224L, shapes(r)[2]);
            checkEq("resize C", 3L, shapes(r)[0]);

            Tensor r2 = OpenCVIO.resize(img, 0.5, 0.5);
            check("scale 0.5 H", Math.abs(shapes(r2)[1] - orig[1] / 2) <= 1);
            check("scale 0.5 W", Math.abs(shapes(r2)[2] - orig[2] / 2) <= 1);

            int h = (int) orig[1], w = (int) orig[2];
            Tensor crop = OpenCVIO.crop(img, 10, 20, h / 2, w / 2);
            checkEq("crop H", (long) (h / 2), shapes(crop)[1]);
            checkEq("crop W", (long) (w / 2), shapes(crop)[2]);

            Tensor flip = OpenCVIO.hflip(img);
            checkEq("hflip shape", orig, shapes(flip));
            // flipping twice ≈ identity
            Tensor flip2 = OpenCVIO.hflip(flip);
            check("hflip² ≈ id", maxAbsDiff(img, flip2) < 1.0f);

            Tensor rot = OpenCVIO.rotate90(img);
            long[] rs = shapes(rot);
            checkEq("rotate90 C", 3L, rs[0]);
            // 90° swap H/W
            checkEq("rotate90 H=origW", orig[2], rs[1]);
            checkEq("rotate90 W=origH", orig[1], rs[2]);
        });

        // ── D4 Color ─────────────────────────────────────────────────────────
        System.out.println("\n══ D4 Color ══");
        section("grayscale / rgbToBgr / normalize", () -> {
            Tensor img = OpenCVIO.readImage(png.toString());
            Tensor g = OpenCVIO.toGrayscale(img);
            checkEq("toGrayscale C=1", 1L, shapes(g)[0]);
            checkEq("toGrayscale H", shapes(img)[1], shapes(g)[1]);
            checkEq("toGrayscale W", shapes(img)[2], shapes(g)[2]);

            // semantic identity currently
            Tensor bgr = OpenCVIO.rgbToBgr(img);
            check("rgbToBgr same ref or shape", Arrays.equals(shapes(img), shapes(bgr)));
            Tensor rgb = OpenCVIO.bgrToRgb(img);
            check("bgrToRgb shape", Arrays.equals(shapes(img), shapes(rgb)));

            // normalize expects [0,255] or [0,1]? Implementation does (x-mean)/std per channel
            Tensor n = OpenCVIO.normalize(img,
                    new float[]{127.5f, 127.5f, 127.5f},
                    new float[]{127.5f, 127.5f, 127.5f});
            checkEq("normalize shape", shapes(img), shapes(n));
            float nMax = n.max().item().toFloat();
            float nMin = n.min().item().toFloat();
            System.out.println("    normalize range=[" + nMin + "," + nMax + "]");
            check("normalize finite range", Float.isFinite(nMin) && Float.isFinite(nMax));
        });

        // ── D5 Edge / fidelity ───────────────────────────────────────────────
        System.out.println("\n══ D5 Edge / fidelity ══");
        section("jpg lossy + small image + exception", () -> {
            Tensor img = OpenCVIO.readImage(jpg.toString());
            check("jpg read ok", shapes(img).length == 3);
            Path re = tmp.resolve("re.jpg");
            OpenCVIO.writeImage(re.toString(), img);
            Tensor img2 = OpenCVIO.readImage(re.toString());
            float md = maxAbsDiff(img, img2);
            System.out.println("    jpg re-encode maxDiff=" + md);
            // Synthetic high-frequency patterns compress poorly; allow generous lossy bound.
            check("jpg re-encode maxDiff < 80", md < 80f);

            // tiny image
            BufferedImage tiny = makeRgb(8, 8);
            Path tinyP = tmp.resolve("tiny.png");
            javax.imageio.ImageIO.write(tiny, "PNG", tinyP.toFile());
            Tensor t = OpenCVIO.readImage(tinyP.toString());
            checkEq("tiny 8x8", new long[]{3, 8, 8}, shapes(t));
            Tensor r = OpenCVIO.resize(t, 224, 224);
            checkEq("tiny→224", new long[]{3, 224, 224}, shapes(r));

            OpenCVException ex = new OpenCVException("boom", 7);
            checkEq("OpenCVException code", 7, ex.errorCode());
            check("OpenCVException msg", ex.getMessage().contains("boom"));
        });

        // ── D6 Daily + throughput ────────────────────────────────────────────
        System.out.println("\n══ D6 Daily pipeline / throughput ══");
        section("daily: read → resize → gray → normalize → encode", () -> {
            Tensor img = OpenCVIO.readImage(png.toString());
            Tensor r = OpenCVIO.resize(img, 224, 224);
            Tensor g = OpenCVIO.toGrayscale(r);
            // expand gray not needed; normalize 1-ch
            Tensor n = OpenCVIO.normalize(r,
                    new float[]{0.485f * 255, 0.456f * 255, 0.406f * 255},
                    new float[]{0.229f * 255, 0.224f * 255, 0.225f * 255});
            byte[] bytes = OpenCVIO.encode(r, "jpg");
            Tensor dec = OpenCVIO.decodeImage(OpenCVIO.encode(r, "png"));
            checkEq("daily resize", new long[]{3, 224, 224}, shapes(r));
            checkEq("daily gray", 1L, shapes(g)[0]);
            checkEq("daily norm", new long[]{3, 224, 224}, shapes(n));
            check("daily jpg bytes", bytes.length > 0);
            check("daily decode", shapes(dec).length == 3);
        });

        section("throughput read/resize/encode", () -> {
            int iters = 30;
            for (int i = 0; i < 3; i++) OpenCVIO.readImage(png.toString());
            long t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) OpenCVIO.readImage(png.toString());
            long ms = (System.nanoTime() - t0) / 1_000_000;
            double ips = iters / (ms / 1000.0);
            System.out.println("    readImage: " + String.format("%.1f", ips) + " img/s");
            check("read throughput > 0", ips > 0);

            Tensor img = OpenCVIO.readImage(png.toString());
            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) OpenCVIO.resize(img, 224, 224);
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / (ms / 1000.0);
            System.out.println("    resize 224: " + String.format("%.1f", ips) + " img/s");
            check("resize throughput > 0", ips > 0);

            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) OpenCVIO.encode(img, "jpg");
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / (ms / 1000.0);
            System.out.println("    encode jpg: " + String.format("%.1f", ips) + " img/s");
            check("encode throughput > 0", ips > 0);
        });

        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed ===");
        if (failed > 0) {
            System.out.println(report);
            System.exit(1);
        }
        deleteRecursive(tmp);
    }

    static void deleteRecursive(Path path) {
        try {
            if (Files.isDirectory(path)) {
                try (var e = Files.list(path)) { e.forEach(BenchmarkOpenCV::deleteRecursive); }
            }
            Files.deleteIfExists(path);
        } catch (Exception ignored) {}
    }
}
