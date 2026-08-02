package org.bytedeco.pytorch.vision.pillow.codec;

import org.bytedeco.pytorch.vision.pillow.Image;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * SPI for Pillow-style image plugins.
 */
public interface ImagePlugin {

    /** Format id, e.g. {@code PNG}, {@code JPEG}, {@code PPM}. */
    String format();

    /** File extensions without dot, lower-case (e.g. png, jpg, jpeg). */
    List<String> extensions();

    /** Optional MIME types. */
    default List<String> mimeTypes() {
        return List.of();
    }

    /**
     * Quick accept on leading bytes (may be empty → fall through to try open).
     * Return true if this plugin should attempt decode.
     */
    boolean accept(byte[] prefix, int length);

    Image open(Path path) throws IOException;

    Image open(InputStream in) throws IOException;

    default Image open(byte[] data) throws IOException {
        return open(new java.io.ByteArrayInputStream(data));
    }

    void save(Image image, Path path, Map<String, Object> options) throws IOException;

    void save(Image image, OutputStream out, String formatHint, Map<String, Object> options) throws IOException;

    default boolean canSave() {
        return true;
    }

    default boolean canRead() {
        return true;
    }
}
