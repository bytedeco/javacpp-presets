package org.bytedeco.pytorch.vision.pillow.enums;

/** Aligns with PIL.Image.Palette. */
public enum Palette {
    WEB(0),
    ADAPTIVE(1);

    private final int value;

    Palette(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static Palette fromValue(int v) {
        for (Palette p : values()) {
            if (p.value == v) return p;
        }
        throw new IllegalArgumentException("unknown Palette: " + v);
    }
}
