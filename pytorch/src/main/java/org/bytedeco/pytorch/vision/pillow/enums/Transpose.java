package org.bytedeco.pytorch.vision.pillow.enums;

/**
 * Aligns with PIL.Image.Transpose IntEnum.
 */
public enum Transpose {
    FLIP_LEFT_RIGHT(0),
    FLIP_TOP_BOTTOM(1),
    ROTATE_90(2),
    ROTATE_180(3),
    ROTATE_270(4),
    TRANSPOSE(5),
    TRANSVERSE(6);

    private final int value;

    Transpose(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static Transpose fromValue(int v) {
        for (Transpose t : values()) {
            if (t.value == v) {
                return t;
            }
        }
        throw new IllegalArgumentException("unknown Transpose: " + v);
    }
}
