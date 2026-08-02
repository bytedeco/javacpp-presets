package org.bytedeco.pytorch.vision.pillow;

import org.bytedeco.pytorch.vision.pillow.core.ModeInfo;

/**
 * Module-level mode helpers (Pillow {@code ImageMode}).
 */
public final class ImageMode {
    private ImageMode() {}

    public static String getmodebase(String mode) {
        return ModeInfo.getmodebase(mode);
    }

    public static String getmodetype(String mode) {
        return ModeInfo.getmodetype(mode);
    }

    public static int getmodebands(String mode) {
        return ModeInfo.getmodebands(mode);
    }

    public static String[] getmodebandnames(String mode) {
        return ModeInfo.getmodebandnames(mode);
    }

    public static ModeInfo getMode(String mode) {
        return ModeInfo.get(mode);
    }
}
