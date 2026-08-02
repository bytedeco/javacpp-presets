package org.bytedeco.pytorch.vision.pillow.features;

import org.bytedeco.pytorch.dataframe.media.MediaBridge;
import org.bytedeco.pytorch.vision.pillow.PillowVersion;
import org.bytedeco.pytorch.vision.pillow.codec.CodecRegistry;
import org.bytedeco.pytorch.vision.pillow.codec.ImagePlugin;
import org.bytedeco.pytorch.vision.pillow.core.DecompressionBomb;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Aligns with {@code PIL.features}: module/codec/feature checks and {@code pilinfo}.
 *
 * <p>Also reports optional OpenCV / FFmpeg interop availability (repo extensions, not upstream PIL).
 */
public final class Features {

    private Features() {}

    public static boolean check_module(String name) {
        return checkModule(name);
    }

    public static boolean checkModule(String name) {
        Objects.requireNonNull(name, "name");
        String n = name.toLowerCase(Locale.ROOT);
        return switch (n) {
            case "pil" -> true;
            case "tkinter", "imagecms", "littlecms2", "webp", "jpg_2000", "avif",
                 "freetype2", "raqm", "fribidi", "harfbuzz" -> check_feature(n);
            // repo extensions
            case "opencv", "cv2" -> MediaBridge.isOpenCvAvailable();
            case "ffmpeg", "av" -> MediaBridge.isFFmpegAvailable();
            default -> false;
        };
    }

    public static String version_module(String name) {
        return versionModule(name);
    }

    public static String versionModule(String name) {
        if (!checkModule(name)) return null;
        if ("pil".equalsIgnoreCase(name)) return PillowVersion.VERSION;
        if ("opencv".equalsIgnoreCase(name) || "cv2".equalsIgnoreCase(name)) return "javacpp-opencv";
        if ("ffmpeg".equalsIgnoreCase(name) || "av".equalsIgnoreCase(name)) return "javacpp-ffmpeg";
        return "jdk";
    }

    public static boolean check_codec(String name) {
        return checkCodec(name);
    }

    public static boolean checkCodec(String name) {
        CodecRegistry.init();
        return CodecRegistry.hasCodec(name);
    }

    public static String version_codec(String name) {
        return versionCodec(name);
    }

    public static String versionCodec(String name) {
        if (!checkCodec(name)) return null;
        String n = name.toLowerCase(Locale.ROOT);
        if ("ppm".equals(n) || "pgm".equals(n) || "pbm".equals(n)) {
            return "pure-java";
        }
        return "imageio-or-pure-java";
    }

    public static boolean check_feature(String name) {
        return checkFeature(name);
    }

    public static boolean checkFeature(String name) {
        Objects.requireNonNull(name, "name");
        String n = name.toLowerCase(Locale.ROOT);
        return switch (n) {
            case "pil", "libjpeg_turbo", "zlib" -> true;
            case "freetype2", "jdk-font" -> true; // JDK Font backend
            case "webp" -> CodecRegistry.hasCodec("webp");
            case "jpg_2000", "jpeg2000" -> CodecRegistry.hasCodec("jpg_2000");
            case "avif" -> CodecRegistry.hasCodec("avif");
            case "opencv", "cv2" -> MediaBridge.isOpenCvAvailable();
            case "ffmpeg", "av" -> MediaBridge.isFFmpegAvailable();
            case "littlecms2", "imagecms", "tkinter", "raqm", "fribidi", "harfbuzz", "libimagequant" -> false;
            case "xcb" -> false;
            default -> false;
        };
    }

    public static List<String> get_supported_codecs() {
        return getSupportedCodecs();
    }

    public static List<String> getSupportedCodecs() {
        CodecRegistry.init();
        List<String> out = new ArrayList<>();
        for (String c : List.of("jpg", "png", "gif", "bmp", "tiff", "ppm", "webp", "jpg_2000", "avif")) {
            if (CodecRegistry.hasCodec(c)) {
                out.add(c);
            }
        }
        return out;
    }

    /** Full codec matrix: name → available (honest; avif etc. may be false). */
    public static Map<String, Boolean> codecMatrix() {
        CodecRegistry.init();
        Map<String, Boolean> m = new LinkedHashMap<>();
        for (String c : List.of(
                "jpg", "png", "gif", "bmp", "tiff", "ppm", "pgm", "pbm",
                "webp", "jpg_2000", "avif", "ico")) {
            m.put(c, CodecRegistry.hasCodec(c));
        }
        return m;
    }

    public static List<String> get_supported_modules() {
        return getSupportedModules();
    }

    public static List<String> getSupportedModules() {
        List<String> out = new ArrayList<>();
        out.add("pil");
        if (checkFeature("freetype2")) out.add("freetype2");
        if (checkFeature("opencv")) out.add("opencv");
        if (checkFeature("ffmpeg")) out.add("ffmpeg");
        return out;
    }

    public static void pilinfo() {
        pilinfo(System.out, true);
    }

    public static void pilinfo(PrintStream out) {
        pilinfo(out, true);
    }

    public static void pilinfo(PrintStream out, boolean supportedFormats) {
        Objects.requireNonNull(out, "out");
        CodecRegistry.init();
        out.println("--------------------------------------------------------------------");
        out.println("Pillow (Java port) " + PillowVersion.VERSION
                + " (upstream ref " + PillowVersion.UPSTREAM_REF + ")");
        out.println("Python modules loaded from (N/A — pure Java)");
        out.println("--------------------------------------------------------------------");
        out.println("Python Pillow modules:");
        out.println(String.format("%-16s : %s", "Pil", checkModule("pil") + " (java " + PillowVersion.VERSION + ")"));
        out.println("--------------------------------------------------------------------");
        out.println("Optional dual-use features:");
        out.println(String.format("%-16s : %s", "freetype2", checkFeature("freetype2") + " (jdk-font)"));
        out.println(String.format("%-16s : %s", "littlecms2", checkFeature("littlecms2")));
        out.println(String.format("%-16s : %s", "webp", checkFeature("webp")));
        out.println(String.format("%-16s : %s", "jpg_2000", checkFeature("jpg_2000")));
        out.println(String.format("%-16s : %s", "avif", checkFeature("avif")));
        out.println("--------------------------------------------------------------------");
        out.println("Repo media interop (not upstream PIL):");
        out.println(String.format("%-16s : %s", "opencv",
                MediaBridge.isOpenCvAvailable() + " (PillowMedia.imageToMat / matToImage)"));
        out.println(String.format("%-16s : %s", "ffmpeg",
                MediaBridge.isFFmpegAvailable() + " (PillowMedia.fromVideoFrame / ffmpegFrames)"));
        out.println(String.format("%-16s : %s", "default decode", "PURE_JAVA (ImageIO + PpmPlugin)"));
        out.println("--------------------------------------------------------------------");
        out.println("MAX_IMAGE_PIXELS  : " + DecompressionBomb.getMaxImagePixels());
        if (supportedFormats) {
            out.println("Registered plugins:");
            for (ImagePlugin p : CodecRegistry.plugins()) {
                boolean readable = p.canRead();
                boolean writable = p.canSave();
                String flags = (readable ? "R" : "-") + (writable ? "W" : "-");
                out.println("  [" + flags + "] " + p.format() + "  ext=" + p.extensions()
                        + "  codec_ok=" + checkCodec(p.format()));
            }
            out.println("Supported codecs  : " + getSupportedCodecs());
            out.println("Codec matrix:");
            for (Map.Entry<String, Boolean> e : codecMatrix().entrySet()) {
                out.println(String.format("  %-12s %s", e.getKey(), e.getValue() ? "yes" : "no"));
            }
        }
        out.println("--------------------------------------------------------------------");
    }
}
