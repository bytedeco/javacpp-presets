package org.bytedeco.pytorch.vision.pillow;

import org.bytedeco.pytorch.vision.pillow.features.Features;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;

/**
 * Facade entry: version / open / new / init / features shortcuts.
 */
public final class Pillow {
    private Pillow() {}

    public static String version() {
        return PillowVersion.VERSION;
    }

    public static String upstream_ref() {
        return PillowVersion.UPSTREAM_REF;
    }

    public static void preinit() {
        Image.preinit();
    }

    public static void init() {
        Image.init();
    }

    public static Image open(String path) throws IOException {
        return Image.open(path);
    }

    public static Image open(Path path) throws IOException {
        return Image.open(path);
    }

    public static Image open(InputStream in) throws IOException {
        return Image.open(in);
    }

    public static Image open(byte[] data) throws IOException {
        return Image.open(data);
    }

    public static Image new_(String mode, int width, int height) {
        return Image.new_(mode, width, height);
    }

    public static Image new_(String mode, int width, int height, Object color) {
        return Image.new_(mode, width, height, color);
    }

    public static Image create(String mode, int width, int height) {
        return Image.create(mode, width, height);
    }

    public static void pilinfo() {
        Features.pilinfo();
    }

    public static boolean check_codec(String name) {
        return Features.check_codec(name);
    }

    public static boolean checkCodec(String name) {
        return Features.checkCodec(name);
    }
}
