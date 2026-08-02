package org.bytedeco.pytorch.vision.pillow;

import java.util.Arrays;
import java.util.Objects;

/**
 * Palette for mode {@code P}/{@code PA} (Pillow {@code ImagePalette.ImagePalette}).
 */
public final class ImagePalette {

    private final String mode; // palette color mode, usually RGB
    private byte[] palette;    // flat RGB or RGBA entries
    private int rawModeBands;

    public ImagePalette() {
        this("RGB");
    }

    public ImagePalette(String mode) {
        this.mode = mode == null ? "RGB" : mode;
        this.rawModeBands = "RGBA".equals(this.mode) ? 4 : 3;
        this.palette = new byte[256 * rawModeBands];
    }

    public ImagePalette(String mode, byte[] data) {
        this(mode);
        if (data != null) {
            System.arraycopy(data, 0, palette, 0, Math.min(data.length, palette.length));
        }
    }

    public String mode() {
        return mode;
    }

    public byte[] palette() {
        return palette.clone();
    }

    public void putpalette(byte[] data) {
        Objects.requireNonNull(data, "data");
        this.palette = Arrays.copyOf(data, Math.max(data.length, 256 * rawModeBands));
    }

    public void putpalette(int[] rgbFlat) {
        Objects.requireNonNull(rgbFlat, "rgb");
        byte[] b = new byte[rgbFlat.length];
        for (int i = 0; i < rgbFlat.length; i++) {
            b[i] = (byte) (rgbFlat[i] & 0xff);
        }
        putpalette(b);
    }

    public byte[] getdata() {
        return palette.clone();
    }

    public int rawModeBands() {
        return rawModeBands;
    }

    public static ImagePalette raw(String mode, byte[] data) {
        return new ImagePalette(mode, data);
    }
}
