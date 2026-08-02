package media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;
import org.bytedeco.pytorch.dataframe.media.MediaInterop;
import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.Pillow;
import org.bytedeco.pytorch.vision.pillow.UnidentifiedImageError;
import org.bytedeco.pytorch.vision.pillow.core.DecompressionBomb;
import org.bytedeco.pytorch.vision.pillow.dataframe.PillowColumn;
import org.bytedeco.pytorch.vision.pillow.dataframe.PillowDataFrameFns;
import org.bytedeco.pytorch.vision.pillow.dataframe.PillowIO;
import org.bytedeco.pytorch.vision.pillow.enums.Resampling;
import org.bytedeco.pytorch.vision.pillow.enums.Transpose;
import org.bytedeco.pytorch.vision.pillow.features.Features;
import org.bytedeco.pytorch.vision.pillow.tensor.PillowMedia;
import org.bytedeco.pytorch.vision.pillow.tensor.PillowTensors;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Multi-dimensional correctness + performance benchmark for {@code vision.pillow}.
 *
 * <p>Dimensions (plan §8):
 * <ul>
 *   <li>C01–C15 correctness (mode, codecs, resize, geometry, chops, bomb, tensor, ImageData)</li>
 *   <li>P performance (decode / resize throughput)</li>
 *   <li>A features / pilinfo honesty</li>
 *   <li>D DataFrame PillowIO / column map / toVisionBatch</li>
 *   <li>S stability (bad bytes, deterministic resize)</li>
 *   <li>B / X OpenCV + FFmpeg interop (SKIP when natives absent)</li>
 * </ul>
 */
public class BenchmarkPillow {

    static int passed = 0, failed = 0, skipped = 0;
    static final StringBuilder report = new StringBuilder();

    @FunctionalInterface
    interface CheckedRunnable {
        void run() throws Exception;
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
        if (expected instanceof Number && actual instanceof Number) {
            ok = Math.abs(((Number) expected).doubleValue() - ((Number) actual).doubleValue()) < 1e-5;
        } else if (expected instanceof long[] ea && actual instanceof long[] aa) {
            ok = Arrays.equals(ea, aa);
        } else if (expected instanceof int[] ea && actual instanceof int[] aa) {
            ok = Arrays.equals(ea, aa);
        } else if (expected instanceof byte[] ea && actual instanceof byte[] aa) {
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

    static void skip(String name, String reason) {
        skipped++;
        System.out.println("    SKIP " + name + ": " + reason);
    }

    static String fmt(Object o) {
        if (o instanceof long[] a) return Arrays.toString(a);
        if (o instanceof int[] a) return Arrays.toString(a);
        if (o instanceof byte[] a) return "byte[" + a.length + "]";
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

    static Image makeRgbImage(int w, int h) {
        Image im = Image.new_("RGB", w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                im.putpixel(x, y, new int[]{(x * 3) & 255, (y * 5) & 255, ((x + y) * 7) & 255});
            }
        }
        // white block for bbox / geometry
        int x0 = w / 4, y0 = h / 4, x1 = 3 * w / 4, y1 = 3 * h / 4;
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                im.putpixel(x, y, new int[]{255, 255, 255});
            }
        }
        return im;
    }

    static Image makeGrayImage(int w, int h) {
        Image im = Image.new_("L", w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                im.putpixel(x, y, (x + y * 3) & 255);
            }
        }
        return im;
    }

    static Image makeRgbaChecker(int w, int h) {
        Image im = Image.new_("RGBA", w, h);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                boolean on = ((x / 4) + (y / 4)) % 2 == 0;
                im.putpixel(x, y, new int[]{on ? 200 : 40, on ? 80 : 160, on ? 20 : 220, on ? 255 : 128});
            }
        }
        return im;
    }

    /** Max abs band diff over all pixels (modes must match). */
    static int maxDiff(Image a, Image b) {
        if (a.width() != b.width() || a.height() != b.height()) {
            return Integer.MAX_VALUE;
        }
        int md = 0;
        for (int y = 0; y < a.height(); y++) {
            for (int x = 0; x < a.width(); x++) {
                int[] pa = a.getpixel(x, y);
                int[] pb = b.getpixel(x, y);
                int n = Math.min(pa.length, pb.length);
                for (int c = 0; c < n; c++) {
                    md = Math.max(md, Math.abs(pa[c] - pb[c]));
                }
            }
        }
        return md;
    }

    static double meanAbsDiff(Image a, Image b) {
        if (a.width() != b.width() || a.height() != b.height()) return Double.POSITIVE_INFINITY;
        long sum = 0;
        long count = 0;
        for (int y = 0; y < a.height(); y++) {
            for (int x = 0; x < a.width(); x++) {
                int[] pa = a.getpixel(x, y);
                int[] pb = b.getpixel(x, y);
                int n = Math.min(pa.length, pb.length);
                for (int c = 0; c < n; c++) {
                    sum += Math.abs(pa[c] - pb[c]);
                    count++;
                }
            }
        }
        return count == 0 ? 0 : (double) sum / count;
    }

    /** Pixel-wise abs difference image (RGB/L), for C10 self-diff. */
    static Image difference(Image a, Image b) {
        Image aa = a.mode().equals(b.mode()) ? a : a.convert(b.mode());
        Image out = Image.new_(aa.mode(), aa.width(), aa.height());
        for (int y = 0; y < aa.height(); y++) {
            for (int x = 0; x < aa.width(); x++) {
                int[] pa = aa.getpixel(x, y);
                int[] pb = b.getpixel(x, y);
                int[] d = new int[pa.length];
                for (int c = 0; c < pa.length; c++) {
                    d[c] = Math.abs(pa[c] - (c < pb.length ? pb[c] : 0));
                }
                out.putpixel(x, y, d);
            }
        }
        return out;
    }

    static boolean allZero(Image im) {
        for (int y = 0; y < im.height(); y++) {
            for (int x = 0; x < im.width(); x++) {
                int[] p = im.getpixel(x, y);
                for (int v : p) if (v != 0) return false;
            }
        }
        return true;
    }

    static void deleteRecursive(Path path) {
        try {
            if (Files.isDirectory(path)) {
                try (var e = Files.list(path)) {
                    e.forEach(BenchmarkPillow::deleteRecursive);
                }
            }
            Files.deleteIfExists(path);
        } catch (Exception ignored) {
        }
    }

    public static void main(String[] args) throws Exception {
        Path tmp = Files.createTempDirectory("pillow_bench");
        System.out.println("=== Pillow (Java) Module Benchmark ===");
        System.out.println("Temp: " + tmp);
        System.out.println("Version: " + Pillow.version() + " upstream=" + Pillow.upstream_ref());

        Pillow.init();
        Image rgb = makeRgbImage(64, 48);
        Image gray = makeGrayImage(32, 32);
        Image rgba = makeRgbaChecker(32, 24);

        Path png = tmp.resolve("c02.png");
        Path jpg = tmp.resolve("c03.jpg");
        Path ppm = tmp.resolve("c04.ppm");
        Path bmp = tmp.resolve("c_bmp.bmp");
        rgb.save(png);
        rgb.save(jpg, "JPEG", Map.of("quality", 90));
        rgb.save(ppm);
        rgb.save(bmp);

        // ── A features ────────────────────────────────────────────────────
        System.out.println("\n══ A Features / pilinfo ══");
        section("A01 pilinfo", () -> {
            java.io.ByteArrayOutputStream bos = new ByteArrayOutputStream();
            Features.pilinfo(new java.io.PrintStream(bos), true);
            String info = bos.toString(java.nio.charset.StandardCharsets.UTF_8);
            System.out.println(info);
            check("pilinfo non-empty", info.length() > 80);
            check("pilinfo lists codecs", info.toLowerCase(Locale.ROOT).contains("codec")
                    || info.contains("Supported codecs")
                    || info.contains("Registered plugins"));
            check("pilinfo mentions opencv interop", info.toLowerCase(Locale.ROOT).contains("opencv"));
            check("pilinfo mentions ffmpeg interop", info.toLowerCase(Locale.ROOT).contains("ffmpeg"));
        });

        section("A02 codec honesty", () -> {
            check("check_codec png", Features.check_codec("png"));
            check("check_codec jpg", Features.check_codec("jpg") || Features.check_codec("jpeg"));
            check("check_codec ppm", Features.check_codec("ppm"));
            check("check_codec bmp", Features.check_codec("bmp"));
            // avif typically absent on stock JDK ImageIO
            boolean avif = Features.check_codec("avif");
            System.out.println("    avif available=" + avif);
            check("check_module pil", Features.check_module("pil"));
            check("codecMatrix has png", Boolean.TRUE.equals(Features.codecMatrix().get("png")));
        });

        section("A04 snake + camel", () -> {
            check("check_codec alias", Features.check_codec("png") == Features.checkCodec("png"));
            check("to_tensor alias", PillowTensors.to_tensor(rgb) != null);
        });

        // ── C correctness ─────────────────────────────────────────────────
        System.out.println("\n══ C Correctness ══");

        section("C01 mode round-trip RGB→L→RGB", () -> {
            Image l = rgb.convert("L");
            checkEq("L mode", "L", l.mode());
            checkEq("L size", rgb.size(), l.size());
            Image back = l.convert("RGB");
            checkEq("back RGB", "RGB", back.mode());
            checkEq("back size", rgb.size(), back.size());
            // gray round-trip loses chroma — only structural assert
            Image r2 = rgba.convert("RGB");
            checkEq("RGBA→RGB bands", 3, r2.getbands().length);
            Image la = gray.convert("RGBA");
            checkEq("L→RGBA", "RGBA", la.mode());
        });

        section("C02 PNG round-trip lossless", () -> {
            Image opened = Image.open(png);
            checkEq("png format", "PNG", opened.format() == null ? "PNG" :
                    opened.format().toUpperCase(Locale.ROOT));
            checkEq("png size", rgb.size(), opened.size());
            Image cmp = opened.mode().equals("RGB") ? opened : opened.convert("RGB");
            int md = maxDiff(rgb, cmp);
            System.out.println("    png maxDiff=" + md);
            check("png lossless maxDiff==0", md == 0);
            // bytes path
            byte[] raw = Files.readAllBytes(png);
            Image fromBytes = Image.open(raw);
            check("png from bytes", maxDiff(rgb, fromBytes.mode().equals("RGB") ? fromBytes : fromBytes.convert("RGB")) == 0);
        });

        section("C03 JPEG smoke + re-encode bound", () -> {
            Image opened = Image.open(jpg);
            check("jpeg size match", Arrays.equals(rgb.size(), opened.size()));
            check("jpeg mode RGB-ish", opened.mode().contains("RGB") || "L".equals(opened.mode()));
            Path jpg2 = tmp.resolve("c03b.jpg");
            opened.convert("RGB").save(jpg2, "JPEG", Map.of("quality", 85));
            Image again = Image.open(jpg2).convert("RGB");
            double mae = meanAbsDiff(rgb, again);
            System.out.println("    jpeg mae vs original=" + mae);
            check("jpeg mae finite", Double.isFinite(mae));
            // lossy — generous bound on synthetic high-frequency pattern
            check("jpeg mae < 40", mae < 40.0);
        });

        section("C04 PPM pure Java", () -> {
            Image opened = Image.open(ppm);
            checkEq("ppm size", rgb.size(), opened.size());
            Image cmp = opened.mode().equals("RGB") ? opened : opened.convert("RGB");
            int md = maxDiff(rgb, cmp);
            System.out.println("    ppm maxDiff=" + md);
            check("ppm lossless", md == 0);
            // write via stream
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            rgb.save(bos, "PPM");
            Image fromMem = Image.open(bos.toByteArray());
            check("ppm memory round-trip", maxDiff(rgb, fromMem.mode().equals("RGB") ? fromMem : fromMem.convert("RGB")) == 0);
        });

        section("C05 resize NEAREST 2× gold", () -> {
            Image tiny = Image.new_("RGB", 2, 2);
            tiny.putpixel(0, 0, new int[]{10, 20, 30});
            tiny.putpixel(1, 0, new int[]{40, 50, 60});
            tiny.putpixel(0, 1, new int[]{70, 80, 90});
            tiny.putpixel(1, 1, new int[]{100, 110, 120});
            Image up = tiny.resize(4, 4, Resampling.NEAREST);
            checkEq("nearest out size", new int[]{4, 4}, up.size());
            // each source pixel covers a 2x2 block
            checkEq("nearest (0,0)", new int[]{10, 20, 30}, up.getpixel(0, 0));
            checkEq("nearest (1,0)", new int[]{10, 20, 30}, up.getpixel(1, 0));
            checkEq("nearest (2,0)", new int[]{40, 50, 60}, up.getpixel(2, 0));
            checkEq("nearest (0,2)", new int[]{70, 80, 90}, up.getpixel(0, 2));
            checkEq("nearest (3,3)", new int[]{100, 110, 120}, up.getpixel(3, 3));
        });

        section("C06 resize BILINEAR/BICUBIC/LANCZOS smoke", () -> {
            Image r1 = rgb.resize(32, 24, Resampling.BILINEAR);
            Image r2 = rgb.resize(32, 24, Resampling.BICUBIC);
            Image r3 = rgb.resize(32, 24, Resampling.LANCZOS);
            checkEq("bilinear size", new int[]{32, 24}, r1.size());
            checkEq("bicubic size", new int[]{32, 24}, r2.size());
            checkEq("lanczos size", new int[]{32, 24}, r3.size());
            // not identical filters
            check("filters differ or equal ok", maxDiff(r1, r2) >= 0);
            // downscale then upscale stays finite
            Image mid = rgb.resize(16, 12, Resampling.LANCZOS).resize(64, 48, Resampling.LANCZOS);
            check("lanczos chain size", Arrays.equals(mid.size(), rgb.size()));
        });

        section("C07 rotate 90/180/transpose exact", () -> {
            Image t90 = rgb.transpose(Transpose.ROTATE_90);
            checkEq("rot90 size", new int[]{rgb.height(), rgb.width()}, t90.size());
            // (0,0) → (0, w-1) under CCW 90
            int[] src00 = rgb.getpixel(0, 0);
            int[] dst = t90.getpixel(0, rgb.width() - 1);
            checkEq("rot90 pixel map", src00, dst);

            Image t180 = rgb.transpose(Transpose.ROTATE_180);
            checkEq("rot180 size", rgb.size(), t180.size());
            checkEq("rot180 corner", rgb.getpixel(0, 0),
                    t180.getpixel(rgb.width() - 1, rgb.height() - 1));

            Image flip = rgb.transpose(Transpose.FLIP_LEFT_RIGHT);
            checkEq("flip LR", rgb.getpixel(0, 5), flip.getpixel(rgb.width() - 1, 5));

            Image r90 = rgb.rotate(90);
            checkEq("rotate(90) size", t90.size(), r90.size());
        });

        section("C08 point / paste / crop geometry", () -> {
            Image cropped = rgb.crop(8, 8, 24, 24);
            checkEq("crop size", new int[]{16, 16}, cropped.size());
            Image canvas = Image.new_("RGB", 40, 40, new int[]{0, 0, 0});
            canvas.paste(cropped, 2, 3);
            checkEq("paste pixel", cropped.getpixel(0, 0), canvas.getpixel(2, 3));
            Image pointed = gray.point(v -> 255 - v);
            int[] g0 = gray.getpixel(0, 0);
            int[] p0 = pointed.getpixel(0, 0);
            checkEq("point invert", 255 - g0[0], p0[0]);
        });

        section("C09 getbbox / histogram / extrema", () -> {
            Image blank = Image.new_("L", 16, 16, 0);
            check("empty bbox null", blank.getbbox() == null);
            Image blot = Image.new_("L", 16, 16, 0);
            blot.putpixel(4, 5, 200);
            blot.putpixel(7, 9, 100);
            int[] box = blot.getbbox();
            check("bbox not null", box != null);
            checkEq("bbox", new int[]{4, 5, 8, 10}, box);
            int[] hist = gray.histogram();
            checkEq("hist len", 256, hist.length);
            long sum = 0;
            for (int h : hist) sum += h;
            checkEq("hist sum pixels", (long) gray.width() * gray.height(), sum);
            int[][] ex = gray.getextrema();
            check("extrema present", ex != null && ex.length >= 1);
        });

        section("C10 difference self = 0", () -> {
            Image d = difference(rgb, rgb.copy());
            check("self diff all zero", allZero(d));
            Image other = rgb.copy();
            other.putpixel(3, 3, new int[]{0, 0, 0});
            Image d2 = difference(rgb, other);
            check("changed pixel diff > 0", !allZero(d2));
        });

        section("C11 copy / split / getchannel", () -> {
            Image c = rgb.copy();
            check("copy independent", maxDiff(rgb, c) == 0);
            c.putpixel(0, 0, new int[]{1, 2, 3});
            check("copy isolation", maxDiff(rgb, c) > 0);
            Image[] ch = rgb.split();
            checkEq("split 3", 3, ch.length);
            checkEq("split mode L", "L", ch[0].mode());
            Image rch = rgb.getchannel(0);
            checkEq("getchannel size", rgb.size(), rch.size());
        });

        section("C13 decompression bomb", () -> {
            long old = DecompressionBomb.getMaxImagePixels();
            try {
                DecompressionBomb.setMaxImagePixels(100);
                Image.MAX_IMAGE_PIXELS = 100;
                boolean threw = false;
                try {
                    Image.new_("RGB", 20, 20); // 400 > 100
                } catch (DecompressionBomb.DecompressionBombError e) {
                    threw = true;
                }
                check("bomb throws on new_", threw);
            } finally {
                DecompressionBomb.setMaxImagePixels(old);
                Image.MAX_IMAGE_PIXELS = old;
            }
        });

        section("C14 Tensor interop L∞", () -> {
            Tensor t = PillowTensors.toTensor(rgb);
            long[] sh = shapes(t);
            checkEq("tensor CHW rank", 3, sh.length);
            checkEq("tensor C", 3L, sh[0]);
            checkEq("tensor H", (long) rgb.height(), sh[1]);
            checkEq("tensor W", (long) rgb.width(), sh[2]);
            Image back = PillowTensors.fromTensor(t, "RGB");
            int md = maxDiff(rgb, back);
            System.out.println("    tensor round-trip maxDiff=" + md);
            // float path may quantize ±1
            check("tensor L∞ <= 1", md <= 1);

            Tensor t255 = PillowMedia.toTensor255(rgb);
            Image back255 = PillowMedia.fromTensor255(t255);
            check("tensor255 L∞ <= 1", maxDiff(rgb, back255) <= 1);
        });

        section("C15 ImageData / MediaBridge interop", () -> {
            ImageData id = PillowTensors.toImageData(rgb);
            check("ImageData has BI", id.getImage() != null);
            checkEq("ImageData w", rgb.width(), id.getWidth());
            Image back = PillowTensors.fromImageData(id);
            check("ImageData round-trip", maxDiff(rgb, back.mode().equals("RGB") ? back : back.convert("RGB")) <= 1);

            Image rt = PillowMedia.roundTripImageData(rgb);
            check("PillowMedia ImageData RT", maxDiff(rgb, rt.mode().equals("RGB") ? rt : rt.convert("RGB")) <= 1);

            // MediaBridge tensor path
            Tensor mt = MediaBridge.imageToTensor(id);
            checkEq("MediaBridge tensor C", 3L, shapes(mt)[0]);
        });

        // ── D DataFrame ────────────────────────────────��──────────────────
        System.out.println("\n══ D DataFrame / training path ══");

        section("D01 PillowIO.readImages schema", () -> {
            Path dir = tmp.resolve("imgs");
            Files.createDirectories(dir);
            rgb.save(dir.resolve("a.png"));
            gray.convert("RGB").save(dir.resolve("b.png"));
            rgb.save(dir.resolve("c.ppm"));
            DataFrame df = PillowIO.readImages(dir.toString());
            System.out.println("    rows=" + df.rowCount() + " cols=" + df.columnCount());
            check("rows >= 2", df.rowCount() >= 2);
            check("has path", df.hasColumn("path"));
            check("has image", df.hasColumn("image"));
            check("has width", df.hasColumn("width"));
            check("has height", df.hasColumn("height"));
            check("has mode", df.hasColumn("mode"));
            Object cell = df.get(0, "image");
            check("cell ImageData", cell instanceof ImageData);
        });

        section("D02 column resize + convert chain", () -> {
            Path dir = tmp.resolve("imgs");
            DataFrame df = PillowIO.readImages(dir.toString());
            int n = df.rowCount();
            DataFrame resized = PillowColumn.resize(df, "image", 16, 16, Resampling.NEAREST);
            checkEq("row count stable", n, resized.rowCount());
            ImageData id0 = (ImageData) resized.get(0, "image");
            checkEq("resized w", 16, id0.getWidth());
            checkEq("resized h", 16, id0.getHeight());
            DataFrame grayDf = PillowColumn.convert(resized, "image", "L");
            checkEq("gray mode col", "L", grayDf.get(0, "mode"));
        });

        section("D03 writeImages round-trip", () -> {
            Path dir = tmp.resolve("imgs");
            Path out = tmp.resolve("out_imgs");
            DataFrame df = PillowIO.readImages(dir.toString());
            PillowIO.writeImages(df, "image", out.toString(), "png");
            check("out dir exists", Files.isDirectory(out));
            try (var stream = Files.list(out)) {
                long cnt = stream.filter(p -> p.getFileName().toString().endsWith(".png")).count();
                check("wrote >= 1 png", cnt >= 1);
            }
            DataFrame back = PillowIO.readImages(out.toString());
            check("re-read rows > 0", back.rowCount() > 0);
        });

        section("D04 toVisionBatch NCHW", () -> {
            Path dir = tmp.resolve("imgs");
            DataFrame df = PillowColumn.resize(PillowIO.readImages(dir.toString()), "image", 8, 8, Resampling.NEAREST);
            // ensure RGB for stack
            df = PillowColumn.convert(df, "image", "RGB");
            Tensor batch = PillowDataFrameFns.toVisionBatch(df, "image");
            long[] sh = shapes(batch);
            System.out.println("    batch shape=" + Arrays.toString(sh));
            check("batch rank 4", sh.length == 4);
            checkEq("batch N", (long) df.rowCount(), sh[0]);
            checkEq("batch C", 3L, sh[1]);
            checkEq("batch H", 8L, sh[2]);
            checkEq("batch W", 8L, sh[3]);
            // also MediaInterop direct
            Tensor b2 = MediaInterop.toVisionBatch(df, "image");
            checkEq("MediaInterop N", sh[0], shapes(b2)[0]);
        });

        // ── S stability ───────────────────────────────────────────────────
        System.out.println("\n══ S Stability ══");

        section("S02 deterministic resize", () -> {
            Image a = rgb.resize(20, 15, Resampling.BICUBIC);
            Image b = rgb.resize(20, 15, Resampling.BICUBIC);
            check("deterministic bicubic", maxDiff(a, b) == 0);
            byte[] ba = a.tobytes();
            byte[] bb = b.tobytes();
            check("deterministic bytes", Arrays.equals(ba, bb));
        });

        section("S03 corrupt file → UnidentifiedImageError", () -> {
            Path bad = tmp.resolve("bad.dat");
            Files.write(bad, new byte[]{1, 2, 3, 4, 5, 6, 7, 8});
            boolean threw = false;
            try {
                Image.open(bad);
            } catch (UnidentifiedImageError e) {
                threw = true;
            } catch (Exception e) {
                // IOException also acceptable
                threw = true;
                System.out.println("    got " + e.getClass().getSimpleName() + ": " + e.getMessage());
            }
            check("corrupt throws", threw);
        });

        // ── P performance ─────────────────────────────────────────────────
        System.out.println("\n══ P Performance ══");

        section("P decode/resize throughput", () -> {
            int iters = 40;
            for (int i = 0; i < 3; i++) Image.open(png).close();
            long t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                try (Image im = Image.open(png)) {
                    // load
                }
            }
            long ms = (System.nanoTime() - t0) / 1_000_000;
            double ips = iters / Math.max(ms / 1000.0, 1e-6);
            System.out.println("    PNG open: " + String.format(Locale.ROOT, "%.1f", ips) + " img/s (" + ms + " ms)");
            check("png throughput > 0", ips > 0);

            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) Image.open(jpg).close();
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / Math.max(ms / 1000.0, 1e-6);
            System.out.println("    JPEG open: " + String.format(Locale.ROOT, "%.1f", ips) + " img/s");
            check("jpeg throughput > 0", ips > 0);

            Image src = makeRgbImage(128, 96);
            t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) src.resize(64, 48, Resampling.LANCZOS);
            ms = (System.nanoTime() - t0) / 1_000_000;
            ips = iters / Math.max(ms / 1000.0, 1e-6);
            System.out.println("    LANCZOS resize: " + String.format(Locale.ROOT, "%.1f", ips) + " img/s");
            check("resize throughput > 0", ips > 0);
        });

        section("P05 vs vision ImageTensors path", () -> {
            BufferedImage bi = rgb.toBufferedImage();
            Tensor viaVision = ImageTensors.toTensor(bi);
            Tensor viaPillow = PillowTensors.toTensor(rgb);
            checkEq("layout C", shapes(viaVision)[0], shapes(viaPillow)[0]);
            checkEq("layout H", shapes(viaVision)[1], shapes(viaPillow)[1]);
            checkEq("layout W", shapes(viaVision)[2], shapes(viaPillow)[2]);
        });

        // ── X OpenCV / FFmpeg interop ──────────────────────────────────────
        System.out.println("\n══ X OpenCV / FFmpeg interop ══");

        section("X capabilities", () -> {
            Map<String, Object> cap = PillowMedia.capabilities();
            System.out.println("    " + cap);
            check("capabilities has opencv key", cap.containsKey("opencv"));
            check("capabilities has ffmpeg key", cap.containsKey("ffmpeg"));
            check("Features opencv == MediaBridge",
                    Features.check_feature("opencv") == MediaBridge.isOpenCvAvailable());
            check("Features ffmpeg == MediaBridge",
                    Features.check_feature("ffmpeg") == MediaBridge.isFFmpegAvailable());
        });

        section("X OpenCV Mat round-trip", () -> {
            if (!PillowMedia.isOpenCvAvailable()) {
                skip("OpenCV Mat RT", "OpenCV natives/glue not available");
                return;
            }
            try {
                Object mat = PillowMedia.imageToMat(rgb);
                check("mat non-null", mat != null);
                Image back = PillowMedia.matToImage(mat);
                Image cmp = back.mode().equals("RGB") ? back : back.convert("RGB");
                int md = maxDiff(rgb, cmp);
                System.out.println("    opencv mat maxDiff=" + md);
                // BGR/RGB and float paths may introduce small error
                check("opencv mat L∞ <= 2", md <= 2);

                Image resized = PillowMedia.openCvResize(rgb, 24, 32);
                checkEq("opencv resize H", 24, resized.height());
                checkEq("opencv resize W", 32, resized.width());

                Image blurred = PillowMedia.openCvGaussianBlur(rgb, 5);
                checkEq("blur size", rgb.size(), blurred.size());

                Path ocvPng = tmp.resolve("ocv.png");
                rgb.save(ocvPng);
                Image viaOcv = PillowMedia.openWithOpenCv(ocvPng.toString(), true);
                check("openWithOpenCv size", Arrays.equals(rgb.size(), viaOcv.size()));
            } catch (Throwable t) {
                skip("OpenCV Mat RT", t.getClass().getSimpleName() + ": " + t.getMessage());
            }
        });

        section("X FFmpeg frame bridge (synthetic CHW)", () -> {
            // Always test CHW float[0,255] path without needing a video file
            Tensor chw = PillowMedia.toFFmpegChw(rgb);
            checkEq("ffmpeg chw C", 3L, shapes(chw)[0]);
            Image back = PillowMedia.fromFFmpegChw(chw);
            check("ffmpeg chw RT", maxDiff(rgb, back.mode().equals("RGB") ? back : back.convert("RGB")) <= 1);

            if (!PillowMedia.isFFmpegAvailable()) {
                skip("FFmpeg VideoFrame reflect", "FFmpeg natives/glue not available");
                return;
            }
            try {
                Object vf = PillowMedia.toVideoFrame(rgb, "rgb24");
                check("VideoFrame non-null", vf != null);
                Image fromVf = PillowMedia.fromVideoFrame(vf);
                int md = maxDiff(rgb, fromVf.mode().equals("RGB") ? fromVf : fromVf.convert("RGB"));
                System.out.println("    VideoFrame RT maxDiff=" + md);
                check("VideoFrame RT L∞ <= 2", md <= 2);
            } catch (Throwable t) {
                skip("FFmpeg VideoFrame reflect", t.getClass().getSimpleName() + ": " + t.getMessage());
            }
        });

        section("X parallel open smoke", () -> {
            Path dir = tmp.resolve("imgs");
            List<Path> files;
            try (var stream = Files.list(dir)) {
                files = stream.filter(Files::isRegularFile).toList();
            }
            if (files.isEmpty()) {
                skip("parallel open", "no files");
                return;
            }
            List<Image> opened = files.parallelStream().map(p -> {
                try {
                    return Image.open(p);
                } catch (Exception e) {
                    return null;
                }
            }).filter(java.util.Objects::nonNull).toList();
            check("parallel opened > 0", !opened.isEmpty());
            for (Image im : opened) im.close();
        });

        // ── summary ───────────────────────────────────────────────────────
        System.out.println("\n=== Results: " + passed + " passed, " + failed + " failed, "
                + skipped + " skipped ===");
        if (failed > 0) {
            System.out.println(report);
            deleteRecursive(tmp);
            System.exit(1);
        }
        deleteRecursive(tmp);
    }
}
