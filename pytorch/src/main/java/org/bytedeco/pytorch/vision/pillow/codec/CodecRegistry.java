package org.bytedeco.pytorch.vision.pillow.codec;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.UnidentifiedImageError;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Pillow-style {@code Image.preinit}/{@code Image.init} plugin registry.
 */
public final class CodecRegistry {

    private static final CopyOnWriteArrayList<ImagePlugin> PLUGINS = new CopyOnWriteArrayList<>();
    private static final Map<String, ImagePlugin> BY_EXT = new LinkedHashMap<>();
    private static final Map<String, ImagePlugin> BY_FORMAT = new LinkedHashMap<>();
    private static volatile boolean preinited;
    private static volatile boolean inited;

    private CodecRegistry() {}

    public static synchronized void preinit() {
        if (preinited) {
            return;
        }
        register(new PpmPlugin());
        register(new ImageIOPlugin("PNG", List.of("png"), List.of("image/png"),
                new byte[]{(byte) 0x89, 0x50, 0x4E, 0x47}, "png"));
        register(new ImageIOPlugin("JPEG", List.of("jpg", "jpeg", "jpe"), List.of("image/jpeg"),
                new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF}, "jpg"));
        register(new ImageIOPlugin("BMP", List.of("bmp", "dib"), List.of("image/bmp"),
                new byte[]{0x42, 0x4D}, "bmp"));
        register(new ImageIOPlugin("GIF", List.of("gif"), List.of("image/gif"),
                new byte[]{0x47, 0x49, 0x46, 0x38}, "gif"));
        register(new ImageIOPlugin("TIFF", List.of("tif", "tiff"), List.of("image/tiff"),
                null, "tiff"));
        preinited = true;
    }

    public static synchronized void init() {
        preinit();
        if (inited) {
            return;
        }
        // Conditional codecs: only mark available if ImageIO has a reader
        register(new ImageIOPlugin("WEBP", List.of("webp"), List.of("image/webp"), null, "webp"));
        register(new ImageIOPlugin("JPEG2000", List.of("jp2", "j2k", "jpx"), List.of("image/jp2"), null, "jpeg 2000"));
        inited = true;
    }

    public static void register(ImagePlugin plugin) {
        Objects.requireNonNull(plugin, "plugin");
        PLUGINS.addIfAbsent(plugin);
        BY_FORMAT.put(plugin.format().toUpperCase(Locale.ROOT), plugin);
        for (String ext : plugin.extensions()) {
            BY_EXT.put(ext.toLowerCase(Locale.ROOT), plugin);
        }
    }

    public static List<ImagePlugin> plugins() {
        init();
        return Collections.unmodifiableList(new ArrayList<>(PLUGINS));
    }

    public static ImagePlugin forExtension(String ext) {
        init();
        if (ext == null) return null;
        String e = ext.toLowerCase(Locale.ROOT);
        if (e.startsWith(".")) e = e.substring(1);
        return BY_EXT.get(e);
    }

    public static ImagePlugin forFormat(String format) {
        init();
        if (format == null) return null;
        return BY_FORMAT.get(format.toUpperCase(Locale.ROOT));
    }

    public static boolean hasCodec(String name) {
        init();
        if (name == null) return false;
        String n = name.toLowerCase(Locale.ROOT);
        // Pillow features names
        return switch (n) {
            case "jpg", "jpeg" -> BY_FORMAT.containsKey("JPEG") && readerExists("JPEG");
            case "png" -> BY_FORMAT.containsKey("PNG") && readerExists("PNG");
            case "gif" -> BY_FORMAT.containsKey("GIF") && readerExists("GIF");
            case "bmp" -> BY_FORMAT.containsKey("BMP") && readerExists("BMP");
            case "tiff", "tif" -> BY_FORMAT.containsKey("TIFF") && readerExists("TIFF");
            case "ppm", "pgm", "pbm" -> BY_FORMAT.containsKey("PPM");
            case "webp" -> readerExists("WEBP") || readerExists("webp");
            case "jpg_2000", "jpeg2k", "jpeg2000" -> readerExists("JPEG 2000") || readerExists("jpeg 2000") || readerExists("jpeg2000");
            case "avif" -> readerExists("AVIF") || readerExists("avif");
            default -> BY_FORMAT.containsKey(name.toUpperCase(Locale.ROOT));
        };
    }

    private static boolean readerExists(String formatName) {
        try {
            return javax.imageio.ImageIO.getImageReadersByFormatName(formatName).hasNext();
        } catch (Throwable t) {
            return false;
        }
    }

    public static Image open(Path path) throws IOException {
        init();
        Objects.requireNonNull(path, "path");
        String name = path.getFileName() == null ? "" : path.getFileName().toString();
        String ext = extensionOf(name);
        ImagePlugin byExt = forExtension(ext);
        byte[] prefix;
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path))) {
            prefix = in.readNBytes(32);
        }
        if (byExt != null && byExt.accept(prefix, prefix.length)) {
            try {
                Image im = byExt.open(path);
                if (im != null) return im;
            } catch (UnidentifiedImageError e) {
                // fall through
            } catch (IOException e) {
                // try other plugins
            }
        }
        for (ImagePlugin p : PLUGINS) {
            if (p == byExt) continue;
            if (!p.accept(prefix, prefix.length) && !prefixLooksGeneric(prefix)) {
                // still try if accept is loose
            }
            try {
                if (p.accept(prefix, prefix.length) || byExt == null) {
                    Image im = p.open(path);
                    if (im != null) return im;
                }
            } catch (UnidentifiedImageError ignored) {
            } catch (IOException ignored) {
            }
        }
        // last: try ImageIO generic via any ImageIOPlugin
        for (ImagePlugin p : PLUGINS) {
            if (p instanceof ImageIOPlugin) {
                try {
                    Image im = p.open(path);
                    if (im != null) return im;
                } catch (Throwable ignored) {
                }
            }
        }
        throw new UnidentifiedImageError("cannot identify image file: " + path);
    }

    public static Image open(InputStream in) throws IOException {
        init();
        Objects.requireNonNull(in, "in");
        BufferedInputStream bin = in instanceof BufferedInputStream b ? b : new BufferedInputStream(in);
        bin.mark(1 << 20);
        byte[] prefix = bin.readNBytes(32);
        bin.reset();
        for (ImagePlugin p : PLUGINS) {
            if (p.accept(prefix, prefix.length)) {
                try {
                    bin.mark(1 << 24);
                    Image im = p.open(bin);
                    if (im != null) return im;
                } catch (Throwable t) {
                    try {
                        bin.reset();
                    } catch (IOException ignored) {
                    }
                }
            }
        }
        // brute force
        byte[] all = bin.readAllBytes();
        return open(all);
    }

    public static Image open(byte[] data) throws IOException {
        init();
        Objects.requireNonNull(data, "data");
        byte[] prefix = data.length <= 32 ? data : java.util.Arrays.copyOf(data, 32);
        for (ImagePlugin p : PLUGINS) {
            if (p.accept(prefix, prefix.length)) {
                try {
                    Image im = p.open(data);
                    if (im != null) return im;
                } catch (Throwable ignored) {
                }
            }
        }
        for (ImagePlugin p : PLUGINS) {
            try {
                Image im = p.open(new ByteArrayInputStream(data));
                if (im != null) return im;
            } catch (Throwable ignored) {
            }
        }
        throw new UnidentifiedImageError("cannot identify image file from bytes");
    }

    public static void save(Image image, Path path, Map<String, Object> options) throws IOException {
        init();
        Objects.requireNonNull(image, "image");
        Objects.requireNonNull(path, "path");
        String ext = extensionOf(path.getFileName() == null ? "" : path.getFileName().toString());
        ImagePlugin p = forExtension(ext);
        if (p == null || !p.canSave()) {
            throw new IOException("no save plugin for extension: " + ext);
        }
        Map<String, Object> opts = options == null ? Map.of() : options;
        p.save(image, path, opts);
    }

    private static boolean prefixLooksGeneric(byte[] prefix) {
        return prefix == null || prefix.length == 0;
    }

    private static String extensionOf(String name) {
        int dot = name.lastIndexOf('.');
        if (dot < 0 || dot == name.length() - 1) return "";
        return name.substring(dot + 1).toLowerCase(Locale.ROOT);
    }

    public static boolean isPreinited() {
        return preinited;
    }

    public static boolean isInited() {
        return inited;
    }
}
