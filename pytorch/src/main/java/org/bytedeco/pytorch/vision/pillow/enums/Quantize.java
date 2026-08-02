package org.bytedeco.pytorch.vision.pillow.enums;

/** Aligns with PIL.Image.Quantize. */
public enum Quantize {
    MEDIANCUT(0),
    MAXCOVERAGE(1),
    FASTOCTREE(2),
    LIBIMAGEQUANT(3);

    private final int value;

    Quantize(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static Quantize fromValue(int v) {
        for (Quantize q : values()) {
            if (q.value == v) return q;
        }
        throw new IllegalArgumentException("unknown Quantize: " + v);
    }
}
