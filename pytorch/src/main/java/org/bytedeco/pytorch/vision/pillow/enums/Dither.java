package org.bytedeco.pytorch.vision.pillow.enums;

/** Aligns with PIL.Image.Dither. */
public enum Dither {
    NONE(0),
    ORDERED(1),
    RASTERIZE(2),
    FLOYDSTEINBERG(3);

    private final int value;

    Dither(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static Dither fromValue(int v) {
        for (Dither d : values()) {
            if (d.value == v) return d;
        }
        throw new IllegalArgumentException("unknown Dither: " + v);
    }
}
