package org.bytedeco.pytorch.vision.pillow.codec;

import org.bytedeco.pytorch.vision.pillow.Image;
import org.bytedeco.pytorch.vision.pillow.UnidentifiedImageError;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;

import javax.imageio.IIOImage;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Generic JDK {@link javax.imageio.ImageIO} bridge plugin for PNG/JPEG/GIF/BMP/TIFF/….
 */
public final class ImageIOPlugin implements ImagePlugin {

    private final String format;
    private final List<String> extensions;
    private final List<String> mimeTypes;
    private final byte[] magic;
    private final String imageIoFormatName;

    public ImageIOPlugin(String format, List<String> extensions, List<String> mimeTypes,
                         byte[] magic, String imageIoFormatName) {
        this.format = Objects.requireNonNull(format);
        this.extensions = List.copyOf(extensions);
        this.mimeTypes = mimeTypes == null ? List.of() : List.copyOf(mimeTypes);
        this.magic = magic == null ? null : magic.clone();
        this.imageIoFormatName = imageIoFormatName == null ? format.toLowerCase(Locale.ROOT) : imageIoFormatName;
    }

    @Override
    public String format() {
        return format;
    }

    @Override
    public List<String> extensions() {
        return extensions;
    }

    @Override
    public List<String> mimeTypes() {
        return mimeTypes;
    }

    @Override
    public boolean accept(byte[] prefix, int length) {
        if (magic == null || magic.length == 0) {
            return true; // try later / by extension
        }
        if (prefix == null || length < magic.length) {
            return false;
        }
        for (int i = 0; i < magic.length; i++) {
            if (prefix[i] != magic[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public Image open(Path path) throws IOException {
        BufferedImage bi = javax.imageio.ImageIO.read(path.toFile());
        if (bi == null) {
            throw new UnidentifiedImageError("ImageIO cannot decode: " + path);
        }
        Image im = Image.fromBuffer(ImagingBuffer.fromBufferedImage(bi), format, Map.of());
        im.setFormat(format);
        return im;
    }

    @Override
    public Image open(InputStream in) throws IOException {
        BufferedImage bi = javax.imageio.ImageIO.read(in);
        if (bi == null) {
            throw new UnidentifiedImageError("ImageIO cannot decode stream as " + format);
        }
        Image im = Image.fromBuffer(ImagingBuffer.fromBufferedImage(bi), format, Map.of());
        im.setFormat(format);
        return im;
    }

    @Override
    public Image open(byte[] data) throws IOException {
        return open(new java.io.ByteArrayInputStream(data));
    }

    @Override
    public void save(Image image, Path path, Map<String, Object> options) throws IOException {
        Objects.requireNonNull(image, "image");
        if (path.getParent() != null) {
            Files.createDirectories(path.getParent());
        }
        String fmt = imageIoFormatName;
        Object f = options == null ? null : options.get("format");
        if (f != null) {
            fmt = String.valueOf(f).toLowerCase(Locale.ROOT);
            if ("jpeg".equals(fmt)) fmt = "jpg";
        }
        BufferedImage bi = image.toBufferedImage();
        if ("jpg".equals(fmt) || "jpeg".equals(fmt)) {
            float q = 0.9f;
            if (options != null && options.get("quality") != null) {
                Object qq = options.get("quality");
                if (qq instanceof Number n) {
                    float v = n.floatValue();
                    q = v > 1f ? v / 100f : v;
                }
            }
            try (OutputStream out = Files.newOutputStream(path)) {
                writeJpeg(bi, out, q);
            }
            return;
        }
        if (!javax.imageio.ImageIO.write(bi, fmt, path.toFile())) {
            // try alternate names
            if (!javax.imageio.ImageIO.write(bi, format.toLowerCase(Locale.ROOT), path.toFile())) {
                throw new IOException("no ImageIO writer for " + fmt);
            }
        }
    }

    @Override
    public void save(Image image, OutputStream out, String formatHint, Map<String, Object> options)
            throws IOException {
        String fmt = formatHint == null ? imageIoFormatName : formatHint.toLowerCase(Locale.ROOT);
        if ("jpeg".equals(fmt)) fmt = "jpg";
        BufferedImage bi = image.toBufferedImage();
        if ("jpg".equals(fmt) || "jpeg".equals(fmt)) {
            float q = 0.9f;
            if (options != null && options.get("quality") instanceof Number n) {
                float v = n.floatValue();
                q = v > 1f ? v / 100f : v;
            }
            writeJpeg(bi, out, q);
            return;
        }
        if (!javax.imageio.ImageIO.write(bi, fmt, out)) {
            throw new IOException("no ImageIO writer for " + fmt);
        }
    }

    private static void writeJpeg(BufferedImage img, OutputStream out, float quality) throws IOException {
        Iterator<ImageWriter> writers = javax.imageio.ImageIO.getImageWritersByFormatName("jpeg");
        if (!writers.hasNext()) {
            throw new IOException("no jpeg writer");
        }
        ImageWriter writer = writers.next();
        try (ImageOutputStream ios = javax.imageio.ImageIO.createImageOutputStream(out)) {
            writer.setOutput(ios);
            ImageWriteParam param = writer.getDefaultWriteParam();
            if (param.canWriteCompressed()) {
                param.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
                param.setCompressionQuality(Math.max(0.01f, Math.min(1f, quality)));
            }
            writer.write(null, new IIOImage(img, null, null), param);
        } finally {
            writer.dispose();
        }
    }

    @Override
    public String toString() {
        return "ImageIOPlugin(" + format + " ext=" + extensions + " magic="
                + (magic == null ? "any" : Arrays.toString(magic)) + ")";
    }
}
