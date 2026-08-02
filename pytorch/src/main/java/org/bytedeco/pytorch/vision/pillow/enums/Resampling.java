package org.bytedeco.pytorch.vision.pillow.enums;

/**
 * Aligns with PIL.Image.Resampling IntEnum values.
 */
public enum Resampling {
    NEAREST(0),
    BOX(4),
    BILINEAR(2),
    HAMMING(5),
    BICUBIC(3),
    LANCZOS(1);

    private final int value;

    Resampling(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static Resampling fromValue(int v) {
        for (Resampling r : values()) {
            if (r.value == v) {
                return r;
            }
        }
        throw new IllegalArgumentException("unknown Resampling: " + v);
    }
}
