package org.bytedeco.pytorch.vision.pillow.core;

import java.util.Objects;

/**
 * Mode conversion matrix paths (Pillow {@code Image.convert} core).
 * Stage-1: 1/L/P/RGB/RGBA/CMYK/YCbCr and basic greyscale luminance.
 */
public final class ModeConvert {

    private ModeConvert() {}

    public static ImagingBuffer convert(ImagingBuffer src, String targetMode) {
        return convert(src, targetMode, null);
    }

    public static ImagingBuffer convert(ImagingBuffer src, String targetMode, byte[] matrixOrNull) {
        Objects.requireNonNull(src, "src");
        Objects.requireNonNull(targetMode, "mode");
        if (src.mode().equals(targetMode)) {
            return src.copy();
        }
        ModeInfo dstInfo = ModeInfo.get(targetMode);
        // Expand P via palette to RGB first when needed
        ImagingBuffer work = src;
        if ("P".equals(src.mode()) && !"P".equals(targetMode) && !"1".equals(targetMode) && !"L".equals(targetMode)) {
            work = paletteToRgb(src);
        } else if ("P".equals(src.mode()) && ("L".equals(targetMode) || "1".equals(targetMode))) {
            work = paletteToRgb(src);
        }

        if ("RGB".equals(targetMode)) {
            return toRgb(work);
        }
        if ("RGBA".equals(targetMode)) {
            return toRgba(work);
        }
        if ("L".equals(targetMode)) {
            return toL(work);
        }
        if ("1".equals(targetMode)) {
            ImagingBuffer l = toL(work);
            ImagingBuffer out = new ImagingBuffer("1", l.width(), l.height());
            byte[] sd = l.data();
            byte[] dd = out.data();
            for (int i = 0; i < dd.length; i++) {
                dd[i] = (byte) ((sd[i] & 0xff) >= 128 ? 255 : 0);
            }
            return out;
        }
        if ("CMYK".equals(targetMode)) {
            return rgbToCmyk(toRgb(work));
        }
        if ("YCbCr".equals(targetMode)) {
            return rgbToYCbCr(toRgb(work));
        }
        if ("LA".equals(targetMode)) {
            ImagingBuffer l = toL(work);
            ImagingBuffer out = new ImagingBuffer("LA", l.width(), l.height());
            byte[] sd = l.data();
            byte[] dd = out.data();
            boolean hasA = work.modeInfo().hasAlpha();
            for (int i = 0, p = 0; i < sd.length; i++, p += 2) {
                dd[p] = sd[i];
                if (hasA && "RGBA".equals(work.mode())) {
                    dd[p + 1] = work.data()[i * 4 + 3];
                } else {
                    dd[p + 1] = (byte) 255;
                }
            }
            return out;
        }
        if ("P".equals(targetMode)) {
            // naive quantize: take R as index after RGB convert
            ImagingBuffer rgb = toRgb(work);
            ImagingBuffer out = new ImagingBuffer("P", rgb.width(), rgb.height());
            byte[] sd = rgb.data();
            byte[] dd = out.data();
            byte[] pal = new byte[768];
            for (int i = 0; i < 256; i++) {
                pal[i * 3] = pal[i * 3 + 1] = pal[i * 3 + 2] = (byte) i;
            }
            out.setPalette(pal);
            for (int i = 0, j = 0; j < dd.length; i += 3, j++) {
                int r = sd[i] & 0xff, g = sd[i + 1] & 0xff, b = sd[i + 2] & 0xff;
                dd[j] = (byte) ((r * 30 + g * 59 + b * 11) / 100);
            }
            return out;
        }
        if ("I".equals(targetMode) || "I;16".equals(targetMode) || "F".equals(targetMode)) {
            ImagingBuffer l = toL(work);
            ImagingBuffer out = new ImagingBuffer(targetMode, l.width(), l.height());
            for (int y = 0; y < l.height(); y++) {
                for (int x = 0; x < l.width(); x++) {
                    int v = l.getpixel(x, y)[0];
                    if ("F".equals(targetMode)) {
                        out.putpixelF(x, y, v);
                    } else {
                        out.putpixel(x, y, new int[]{v});
                    }
                }
            }
            return out;
        }
        // last resort: convert via RGB
        if (!"RGB".equals(work.mode())) {
            work = toRgb(work);
        }
        throw new IllegalArgumentException("conversion from " + src.mode() + " to " + targetMode + " not implemented");
    }

    private static ImagingBuffer paletteToRgb(ImagingBuffer src) {
        ImagingBuffer out = new ImagingBuffer("RGB", src.width(), src.height());
        byte[] pal = src.getPalette();
        byte[] sd = src.data();
        byte[] dd = out.data();
        for (int i = 0; i < sd.length; i++) {
            int idx = sd[i] & 0xff;
            int o = i * 3;
            if (pal != null && pal.length >= (idx + 1) * 3) {
                dd[o] = pal[idx * 3];
                dd[o + 1] = pal[idx * 3 + 1];
                dd[o + 2] = pal[idx * 3 + 2];
            } else {
                dd[o] = dd[o + 1] = dd[o + 2] = (byte) idx;
            }
        }
        return out;
    }

    private static ImagingBuffer toRgb(ImagingBuffer src) {
        String m = src.mode();
        if ("RGB".equals(m)) {
            return src.copy();
        }
        int w = src.width(), h = src.height();
        ImagingBuffer out = new ImagingBuffer("RGB", w, h);
        byte[] dd = out.data();
        if ("RGBA".equals(m) || "RGBa".equals(m)) {
            byte[] sd = src.data();
            for (int i = 0, j = 0; i < sd.length; i += 4, j += 3) {
                dd[j] = sd[i];
                dd[j + 1] = sd[i + 1];
                dd[j + 2] = sd[i + 2];
            }
            return out;
        }
        if ("L".equals(m) || "1".equals(m)) {
            byte[] sd = src.data();
            for (int i = 0, j = 0; i < sd.length; i++, j += 3) {
                dd[j] = dd[j + 1] = dd[j + 2] = sd[i];
            }
            return out;
        }
        if ("LA".equals(m)) {
            byte[] sd = src.data();
            for (int i = 0, j = 0; i < sd.length; i += 2, j += 3) {
                dd[j] = dd[j + 1] = dd[j + 2] = sd[i];
            }
            return out;
        }
        if ("CMYK".equals(m)) {
            return cmykToRgb(src);
        }
        if ("YCbCr".equals(m)) {
            return yCbCrToRgb(src);
        }
        // generic: replicate first band
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int[] p = src.getpixel(x, y);
                int v = p[0];
                int r = p.length > 0 ? p[0] : v;
                int g = p.length > 1 ? p[1] : r;
                int b = p.length > 2 ? p[2] : r;
                out.putpixel(x, y, new int[]{r, g, b});
            }
        }
        return out;
    }

    private static ImagingBuffer toRgba(ImagingBuffer src) {
        if ("RGBA".equals(src.mode())) {
            return src.copy();
        }
        ImagingBuffer rgb = toRgb(src);
        ImagingBuffer out = new ImagingBuffer("RGBA", rgb.width(), rgb.height());
        byte[] sd = rgb.data();
        byte[] dd = out.data();
        int alpha = 255;
        if ("LA".equals(src.mode())) {
            // handled below differently
        }
        for (int i = 0, j = 0; i < sd.length; i += 3, j += 4) {
            dd[j] = sd[i];
            dd[j + 1] = sd[i + 1];
            dd[j + 2] = sd[i + 2];
            dd[j + 3] = (byte) 255;
        }
        if ("LA".equals(src.mode())) {
            byte[] la = src.data();
            for (int i = 0, j = 0; j < dd.length; i += 2, j += 4) {
                dd[j] = dd[j + 1] = dd[j + 2] = la[i];
                dd[j + 3] = la[i + 1];
            }
        }
        return out;
    }

    private static ImagingBuffer toL(ImagingBuffer src) {
        if ("L".equals(src.mode()) || "1".equals(src.mode())) {
            return src.copy();
        }
        ImagingBuffer rgb = toRgb(src);
        ImagingBuffer out = new ImagingBuffer("L", rgb.width(), rgb.height());
        byte[] sd = rgb.data();
        byte[] dd = out.data();
        // ITU-R BT.601 luminance used by Pillow roughly: 299/587/114
        for (int i = 0, j = 0; i < sd.length; i += 3, j++) {
            int r = sd[i] & 0xff, g = sd[i + 1] & 0xff, b = sd[i + 2] & 0xff;
            dd[j] = (byte) ((r * 299 + g * 587 + b * 114 + 500) / 1000);
        }
        return out;
    }

    private static ImagingBuffer rgbToCmyk(ImagingBuffer rgb) {
        ImagingBuffer out = new ImagingBuffer("CMYK", rgb.width(), rgb.height());
        byte[] sd = rgb.data();
        byte[] dd = out.data();
        for (int i = 0, j = 0; i < sd.length; i += 3, j += 4) {
            int r = sd[i] & 0xff, g = sd[i + 1] & 0xff, b = sd[i + 2] & 0xff;
            int c = 255 - r, m = 255 - g, y = 255 - b;
            int k = Math.min(c, Math.min(m, y));
            if (k == 255) {
                dd[j] = dd[j + 1] = dd[j + 2] = 0;
                dd[j + 3] = (byte) 255;
            } else {
                dd[j] = (byte) ((c - k) * 255 / (255 - k));
                dd[j + 1] = (byte) ((m - k) * 255 / (255 - k));
                dd[j + 2] = (byte) ((y - k) * 255 / (255 - k));
                dd[j + 3] = (byte) k;
            }
        }
        return out;
    }

    private static ImagingBuffer cmykToRgb(ImagingBuffer cmyk) {
        ImagingBuffer out = new ImagingBuffer("RGB", cmyk.width(), cmyk.height());
        byte[] sd = cmyk.data();
        byte[] dd = out.data();
        for (int i = 0, j = 0; i < sd.length; i += 4, j += 3) {
            int c = sd[i] & 0xff, m = sd[i + 1] & 0xff, y = sd[i + 2] & 0xff, k = sd[i + 3] & 0xff;
            dd[j] = (byte) (255 - Math.min(255, c + k));
            dd[j + 1] = (byte) (255 - Math.min(255, m + k));
            dd[j + 2] = (byte) (255 - Math.min(255, y + k));
        }
        return out;
    }

    private static ImagingBuffer rgbToYCbCr(ImagingBuffer rgb) {
        ImagingBuffer out = new ImagingBuffer("YCbCr", rgb.width(), rgb.height());
        byte[] sd = rgb.data();
        byte[] dd = out.data();
        for (int i = 0, j = 0; i < sd.length; i += 3, j += 3) {
            int r = sd[i] & 0xff, g = sd[i + 1] & 0xff, b = sd[i + 2] & 0xff;
            int y = (r * 299 + g * 587 + b * 114 + 500) / 1000;
            int cb = 128 + (-r * 169 - g * 331 + b * 500 + 500) / 1000;
            int cr = 128 + (r * 500 - g * 419 - b * 81 + 500) / 1000;
            dd[j] = (byte) ImagingBuffer.clamp8(y);
            dd[j + 1] = (byte) ImagingBuffer.clamp8(cb);
            dd[j + 2] = (byte) ImagingBuffer.clamp8(cr);
        }
        return out;
    }

    private static ImagingBuffer yCbCrToRgb(ImagingBuffer ycbcr) {
        ImagingBuffer out = new ImagingBuffer("RGB", ycbcr.width(), ycbcr.height());
        byte[] sd = ycbcr.data();
        byte[] dd = out.data();
        for (int i = 0, j = 0; i < sd.length; i += 3, j += 3) {
            int y = sd[i] & 0xff, cb = (sd[i + 1] & 0xff) - 128, cr = (sd[i + 2] & 0xff) - 128;
            int r = y + (cr * 1402) / 1000;
            int g = y - (cb * 344 + cr * 714) / 1000;
            int b = y + (cb * 1772) / 1000;
            dd[j] = (byte) ImagingBuffer.clamp8(r);
            dd[j + 1] = (byte) ImagingBuffer.clamp8(g);
            dd[j + 2] = (byte) ImagingBuffer.clamp8(b);
        }
        return out;
    }
}
