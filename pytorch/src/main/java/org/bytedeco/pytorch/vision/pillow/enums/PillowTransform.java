package org.bytedeco.pytorch.vision.pillow.enums;

/**
 * Aligns with PIL.Image.Transform IntEnum.
 */
public enum PillowTransform {
    AFFINE(0),
    EXTENT(1),
    PERSPECTIVE(2),
    QUAD(3),
    MESH(4);

    private final int value;

    PillowTransform(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static PillowTransform fromValue(int v) {
        for (PillowTransform t : values()) {
            if (t.value == v) {
                return t;
            }
        }
        throw new IllegalArgumentException("unknown Transform: " + v);
    }
}
