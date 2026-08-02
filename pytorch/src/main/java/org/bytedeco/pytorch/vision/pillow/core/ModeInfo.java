package org.bytedeco.pytorch.vision.pillow.core;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Mode table aligned with Pillow {@code ImageMode} / internal mode descriptors.
 * Provides {@code getmodebase}, {@code getmodetype}, {@code getmodebands}, {@code getmodebandnames}.
 */
public final class ModeInfo {

    private static final Map<String, ModeInfo> MODES = new LinkedHashMap<>();

    public static final ModeInfo MODE_1 = register(new ModeInfo("1", 1, 1, "L", "1", "L"));
    public static final ModeInfo MODE_L = register(new ModeInfo("L", 8, 1, "L", "L", "L"));
    public static final ModeInfo MODE_P = register(new ModeInfo("P", 8, 1, "P", "P", "P"));
    public static final ModeInfo MODE_RGB = register(new ModeInfo("RGB", 8, 3, "RGB", "L", "R", "G", "B"));
    public static final ModeInfo MODE_RGBA = register(new ModeInfo("RGBA", 8, 4, "RGB", "L", "R", "G", "B", "A"));
    public static final ModeInfo MODE_CMYK = register(new ModeInfo("CMYK", 8, 4, "RGB", "L", "C", "M", "Y", "K"));
    public static final ModeInfo MODE_YCbCr = register(new ModeInfo("YCbCr", 8, 3, "RGB", "L", "Y", "Cb", "Cr"));
    public static final ModeInfo MODE_LAB = register(new ModeInfo("LAB", 8, 3, "RGB", "L", "L", "A", "B"));
    public static final ModeInfo MODE_HSV = register(new ModeInfo("HSV", 8, 3, "RGB", "L", "H", "S", "V"));
    public static final ModeInfo MODE_I = register(new ModeInfo("I", 32, 1, "L", "I", "I"));
    public static final ModeInfo MODE_I16 = register(new ModeInfo("I;16", 16, 1, "L", "I", "I"));
    public static final ModeInfo MODE_F = register(new ModeInfo("F", 32, 1, "L", "F", "F"));
    public static final ModeInfo MODE_LA = register(new ModeInfo("LA", 8, 2, "L", "L", "L", "A"));
    public static final ModeInfo MODE_PA = register(new ModeInfo("PA", 8, 2, "RGB", "L", "P", "A"));
    public static final ModeInfo MODE_RGBa = register(new ModeInfo("RGBa", 8, 4, "RGB", "L", "R", "G", "B", "a"));

    private final String mode;
    private final int bits;
    private final int bands;
    private final String basemode;
    private final String basetype;
    private final String[] bandnames;
    private final int bytesPerPixel;

    private ModeInfo(String mode, int bits, int bands, String basemode, String basetype, String... bandnames) {
        this.mode = mode;
        this.bits = bits;
        this.bands = bands;
        this.basemode = basemode;
        this.basetype = basetype;
        this.bandnames = bandnames.clone();
        if ("1".equals(mode)) {
            this.bytesPerPixel = 1; // expanded to 1 byte/pix for arithmetic; bit-pack optional later
        } else if (bits <= 8) {
            this.bytesPerPixel = bands;
        } else if (bits == 16) {
            this.bytesPerPixel = bands * 2;
        } else {
            this.bytesPerPixel = bands * 4;
        }
    }

    private static ModeInfo register(ModeInfo m) {
        MODES.put(m.mode, m);
        return m;
    }

    public static ModeInfo get(String mode) {
        Objects.requireNonNull(mode, "mode");
        ModeInfo m = MODES.get(mode);
        if (m == null) {
            m = MODES.get(mode.toUpperCase(Locale.ROOT));
        }
        if (m == null) {
            throw new IllegalArgumentException("unrecognized mode " + mode);
        }
        return m;
    }

    public static boolean isKnown(String mode) {
        return mode != null && (MODES.containsKey(mode) || MODES.containsKey(mode.toUpperCase(Locale.ROOT)));
    }

    public static Map<String, ModeInfo> all() {
        return Collections.unmodifiableMap(MODES);
    }

    public static String getmodebase(String mode) {
        return get(mode).basemode;
    }

    public static String getmodetype(String mode) {
        return get(mode).basetype;
    }

    public static int getmodebands(String mode) {
        return get(mode).bands;
    }

    public static String[] getmodebandnames(String mode) {
        return get(mode).bandnames.clone();
    }

    public String mode() {
        return mode;
    }

    public int bits() {
        return bits;
    }

    public int bands() {
        return bands;
    }

    public String basemode() {
        return basemode;
    }

    public String basetype() {
        return basetype;
    }

    public String[] bandnames() {
        return bandnames.clone();
    }

    /** Storage bytes per pixel in {@link ImagingBuffer} layout. */
    public int bytesPerPixel() {
        return bytesPerPixel;
    }

    public boolean isByteMode() {
        return bits <= 8;
    }

    public boolean hasAlpha() {
        for (String b : bandnames) {
            if ("A".equalsIgnoreCase(b) || "a".equals(b)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return "ModeInfo(" + mode + ")";
    }
}
