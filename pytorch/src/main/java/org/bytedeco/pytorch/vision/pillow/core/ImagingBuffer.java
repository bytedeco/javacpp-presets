package org.bytedeco.pytorch.vision.pillow.core;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.util.Arrays;
import java.util.Objects;

/**
 * Contiguous pure-Java pixel store (Pillow {@code _imaging} semantic stand-in).
 *
 * <p>Layout for 8-bit modes: {@code bands} bytes per pixel, row-major, no padding.
 * {@code I}/{@code F}/{@code I;16} use little-endian multi-byte cells.
 * Mode {@code 1} is expanded to 0/255 in an L-like byte plane for arithmetic.
 */
public final class ImagingBuffer implements Cloneable {

    private final ModeInfo modeInfo;
    private final int width;
    private final int height;
    private final byte[] data;
    private byte[] palette; // optional, length 768 (RGB) or 1024 (RGBA) for mode P/PA

    public ImagingBuffer(String mode, int width, int height) {
        this(ModeInfo.get(mode), width, height, null);
    }

    public ImagingBuffer(ModeInfo modeInfo, int width, int height) {
        this(modeInfo, width, height, null);
    }

    public ImagingBuffer(ModeInfo modeInfo, int width, int height, byte[] data) {
        this.modeInfo = Objects.requireNonNull(modeInfo, "mode");
        if (width < 0 || height < 0) {
            throw new IllegalArgumentException("size must be non-negative, got " + width + "x" + height);
        }
        DecompressionBomb.check(width, height);
        this.width = width;
        this.height = height;
        int need = sizeBytes(modeInfo, width, height);
        if (data == null) {
            this.data = new byte[need];
        } else {
            if (data.length < need) {
                throw new IllegalArgumentException(
                        "data length " + data.length + " < required " + need);
            }
            this.data = data;
        }
    }

    public static int sizeBytes(ModeInfo mode, int w, int h) {
        long n = (long) mode.bytesPerPixel() * (long) w * (long) h;
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("image too large");
        }
        return (int) n;
    }

    public static ImagingBuffer zeros(String mode, int width, int height) {
        return new ImagingBuffer(mode, width, height);
    }

    public static ImagingBuffer filled(String mode, int width, int height, int[] colorBands) {
        ImagingBuffer buf = new ImagingBuffer(mode, width, height);
        buf.fill(colorBands);
        return buf;
    }

    public ModeInfo modeInfo() {
        return modeInfo;
    }

    public String mode() {
        return modeInfo.mode();
    }

    public int width() {
        return width;
    }

    public int height() {
        return height;
    }

    public int[] size() {
        return new int[]{width, height};
    }

    public int bands() {
        return modeInfo.bands();
    }

    public int stride() {
        return modeInfo.bytesPerPixel() * width;
    }

    /** Mutable raw storage; callers must respect mode layout. */
    public byte[] data() {
        return data;
    }

    public byte[] getPalette() {
        return palette == null ? null : palette.clone();
    }

    public void setPalette(byte[] palette) {
        this.palette = palette == null ? null : palette.clone();
    }

    public void fill(int[] colorBands) {
        Objects.requireNonNull(colorBands, "color");
        int bpp = modeInfo.bytesPerPixel();
        int b = modeInfo.bands();
        if (modeInfo.isByteMode()) {
            byte[] pix = new byte[bpp];
            for (int i = 0; i < b && i < colorBands.length; i++) {
                pix[i] = (byte) clamp8(colorBands[i]);
            }
            if (width == 0 || height == 0) {
                return;
            }
            // fill first row then copy rows
            for (int x = 0; x < width; x++) {
                System.arraycopy(pix, 0, data, x * bpp, bpp);
            }
            int row = stride();
            for (int y = 1; y < height; y++) {
                System.arraycopy(data, 0, data, y * row, row);
            }
            return;
        }
        // multi-byte modes: write per-pixel
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                putpixel(x, y, colorBands);
            }
        }
    }

    public int[] getpixel(int x, int y) {
        checkXY(x, y);
        int bpp = modeInfo.bytesPerPixel();
        int off = (y * width + x) * bpp;
        int b = modeInfo.bands();
        int[] out = new int[b];
        if (modeInfo.isByteMode()) {
            for (int i = 0; i < b; i++) {
                out[i] = data[off + i] & 0xff;
            }
            return out;
        }
        if ("I;16".equals(modeInfo.mode())) {
            out[0] = (data[off] & 0xff) | ((data[off + 1] & 0xff) << 8);
            return out;
        }
        if ("I".equals(modeInfo.mode())) {
            out[0] = (data[off] & 0xff)
                    | ((data[off + 1] & 0xff) << 8)
                    | ((data[off + 2] & 0xff) << 16)
                    | ((data[off + 3] & 0xff) << 24);
            return out;
        }
        if ("F".equals(modeInfo.mode())) {
            int bits = (data[off] & 0xff)
                    | ((data[off + 1] & 0xff) << 8)
                    | ((data[off + 2] & 0xff) << 16)
                    | ((data[off + 3] & 0xff) << 24);
            out[0] = Float.floatToIntBits(Float.intBitsToFloat(bits)); // store raw bits as int view
            // better: return float via separate API; keep int bits for put/get symmetry of raw
            out[0] = bits;
            return out;
        }
        for (int i = 0; i < b; i++) {
            out[i] = data[off + i] & 0xff;
        }
        return out;
    }

    public float getpixelF(int x, int y) {
        checkXY(x, y);
        if (!"F".equals(modeInfo.mode())) {
            return getpixel(x, y)[0];
        }
        int off = (y * width + x) * 4;
        int bits = (data[off] & 0xff)
                | ((data[off + 1] & 0xff) << 8)
                | ((data[off + 2] & 0xff) << 16)
                | ((data[off + 3] & 0xff) << 24);
        return Float.intBitsToFloat(bits);
    }

    public void putpixel(int x, int y, int[] colorBands) {
        checkXY(x, y);
        Objects.requireNonNull(colorBands, "color");
        int bpp = modeInfo.bytesPerPixel();
        int off = (y * width + x) * bpp;
        int b = modeInfo.bands();
        if (modeInfo.isByteMode()) {
            for (int i = 0; i < b; i++) {
                int v = i < colorBands.length ? colorBands[i] : 0;
                data[off + i] = (byte) clamp8(v);
            }
            return;
        }
        if ("I;16".equals(modeInfo.mode())) {
            int v = colorBands.length > 0 ? colorBands[0] : 0;
            data[off] = (byte) (v & 0xff);
            data[off + 1] = (byte) ((v >> 8) & 0xff);
            return;
        }
        if ("I".equals(modeInfo.mode())) {
            int v = colorBands.length > 0 ? colorBands[0] : 0;
            data[off] = (byte) (v & 0xff);
            data[off + 1] = (byte) ((v >> 8) & 0xff);
            data[off + 2] = (byte) ((v >> 16) & 0xff);
            data[off + 3] = (byte) ((v >> 24) & 0xff);
            return;
        }
        if ("F".equals(modeInfo.mode())) {
            float f = colorBands.length > 0 ? Float.intBitsToFloat(colorBands[0]) : 0f;
            // if caller passed raw int bits already, also accept putpixelF
            int bits = colorBands[0];
            data[off] = (byte) (bits & 0xff);
            data[off + 1] = (byte) ((bits >> 8) & 0xff);
            data[off + 2] = (byte) ((bits >> 16) & 0xff);
            data[off + 3] = (byte) ((bits >> 24) & 0xff);
            return;
        }
        for (int i = 0; i < b; i++) {
            data[off + i] = (byte) clamp8(i < colorBands.length ? colorBands[i] : 0);
        }
    }

    public void putpixelF(int x, int y, float value) {
        checkXY(x, y);
        if (!"F".equals(modeInfo.mode())) {
            putpixel(x, y, new int[]{Math.round(value)});
            return;
        }
        int bits = Float.floatToIntBits(value);
        int off = (y * width + x) * 4;
        data[off] = (byte) (bits & 0xff);
        data[off + 1] = (byte) ((bits >> 8) & 0xff);
        data[off + 2] = (byte) ((bits >> 16) & 0xff);
        data[off + 3] = (byte) ((bits >> 24) & 0xff);
    }

    public void putdata(int[] flat, int offset, int stride) {
        int n = width * height;
        int b = modeInfo.bands();
        int idx = offset;
        for (int i = 0; i < n; i++) {
            int[] pix = new int[b];
            for (int c = 0; c < b; c++) {
                pix[c] = flat[idx + c];
            }
            int x = i % width;
            int y = i / width;
            putpixel(x, y, pix);
            idx += stride;
        }
    }

    public int[] getdata() {
        int n = width * height;
        int b = modeInfo.bands();
        int[] out = new int[n * b];
        int o = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] p = getpixel(x, y);
                for (int c = 0; c < b; c++) {
                    out[o++] = p[c];
                }
            }
        }
        return out;
    }

    public ImagingBuffer copy() {
        ImagingBuffer c = new ImagingBuffer(modeInfo, width, height, Arrays.copyOf(data, data.length));
        if (palette != null) {
            c.palette = palette.clone();
        }
        return c;
    }

    @Override
    public ImagingBuffer clone() {
        return copy();
    }

    public ImagingBuffer crop(int left, int upper, int right, int lower) {
        if (left < 0) left = 0;
        if (upper < 0) upper = 0;
        if (right > width) right = width;
        if (lower > height) lower = height;
        if (right < left) right = left;
        if (lower < upper) lower = upper;
        int nw = right - left;
        int nh = lower - upper;
        ImagingBuffer out = new ImagingBuffer(modeInfo, nw, nh);
        int bpp = modeInfo.bytesPerPixel();
        int srcStride = stride();
        int dstStride = out.stride();
        for (int y = 0; y < nh; y++) {
            int srcOff = ((upper + y) * width + left) * bpp;
            int dstOff = y * nw * bpp;
            System.arraycopy(data, srcOff, out.data, dstOff, nw * bpp);
        }
        if (palette != null) {
            out.palette = palette.clone();
        }
        return out;
    }

    /**
     * Paste {@code src} at (x, y). Modes must match (or src converted by caller).
     * Optional box on src: left, upper, right, lower.
     */
    public void paste(ImagingBuffer src, int x, int y) {
        paste(src, x, y, 0, 0, src.width, src.height, null);
    }

    public void paste(ImagingBuffer src, int x, int y, int srcLeft, int srcUpper, int srcRight, int srcLower,
                      ImagingBuffer mask) {
        Objects.requireNonNull(src, "src");
        if (!src.modeInfo.mode().equals(modeInfo.mode()) && mask == null) {
            // allow if same basetype byte layout size
            if (src.modeInfo.bytesPerPixel() != modeInfo.bytesPerPixel()) {
                throw new IllegalArgumentException(
                        "mode mismatch paste " + src.mode() + " onto " + mode());
            }
        }
        int sw = srcRight - srcLeft;
        int sh = srcLower - srcUpper;
        int bpp = modeInfo.bytesPerPixel();
        for (int sy = 0; sy < sh; sy++) {
            int dy = y + sy;
            if (dy < 0 || dy >= height) continue;
            int srcY = srcUpper + sy;
            for (int sx = 0; sx < sw; sx++) {
                int dx = x + sx;
                if (dx < 0 || dx >= width) continue;
                int srcX = srcLeft + sx;
                if (mask != null) {
                    int[] m = mask.getpixel(Math.min(srcX, mask.width - 1), Math.min(srcY, mask.height - 1));
                    int alpha = m[m.length - 1];
                    if (alpha <= 0) continue;
                    if (alpha >= 255) {
                        putpixel(dx, dy, src.getpixel(srcX, srcY));
                    } else {
                        int[] sp = src.getpixel(srcX, srcY);
                        int[] dp = getpixel(dx, dy);
                        int[] out = new int[dp.length];
                        for (int c = 0; c < out.length; c++) {
                            int s = c < sp.length ? sp[c] : 0;
                            out[c] = (s * alpha + dp[c] * (255 - alpha) + 127) / 255;
                        }
                        putpixel(dx, dy, out);
                    }
                } else {
                    int srcOff = (srcY * src.width + srcX) * src.modeInfo.bytesPerPixel();
                    int dstOff = (dy * width + dx) * bpp;
                    System.arraycopy(src.data, srcOff, data, dstOff, Math.min(bpp, src.modeInfo.bytesPerPixel()));
                }
            }
        }
    }

    public byte[] tobytes() {
        return Arrays.copyOf(data, data.length);
    }

    public static ImagingBuffer frombytes(String mode, int width, int height, byte[] bytes) {
        ModeInfo mi = ModeInfo.get(mode);
        int need = sizeBytes(mi, width, height);
        byte[] copy = Arrays.copyOf(Objects.requireNonNull(bytes, "bytes"), need);
        if (bytes.length < need) {
            // zero-pad already via copyOf
        } else if (bytes.length > need) {
            copy = Arrays.copyOf(bytes, need);
        }
        return new ImagingBuffer(mi, width, height, copy);
    }

    /**
     * Build from {@link BufferedImage}. RGB/ARGB/gray paths mirror
     * {@link org.bytedeco.pytorch.vision.utils.ImageTensors} channel order (R,G,B[,A]).
     */
    public static ImagingBuffer fromBufferedImage(BufferedImage image) {
        Objects.requireNonNull(image, "image");
        int w = image.getWidth();
        int h = image.getHeight();
        int type = image.getType();
        if (type == BufferedImage.TYPE_BYTE_GRAY) {
            ImagingBuffer buf = new ImagingBuffer("L", w, h);
            if (image.getRaster().getDataBuffer() instanceof DataBufferByte dbb) {
                byte[] src = dbb.getData();
                System.arraycopy(src, 0, buf.data, 0, Math.min(src.length, buf.data.length));
            } else {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int p = image.getRGB(x, y);
                        int g = (p >> 16) & 0xff; // gray stored in R for TYPE_BYTE_GRAY getRGB
                        buf.data[y * w + x] = (byte) g;
                    }
                }
            }
            return buf;
        }
        boolean hasAlpha = image.getColorModel().hasAlpha();
        if (hasAlpha) {
            ImagingBuffer buf = new ImagingBuffer("RGBA", w, h);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int p = image.getRGB(x, y);
                    int a = (p >> 24) & 0xff;
                    int r = (p >> 16) & 0xff;
                    int g = (p >> 8) & 0xff;
                    int b = p & 0xff;
                    int off = (y * w + x) * 4;
                    buf.data[off] = (byte) r;
                    buf.data[off + 1] = (byte) g;
                    buf.data[off + 2] = (byte) b;
                    buf.data[off + 3] = (byte) a;
                }
            }
            return buf;
        }
        ImagingBuffer buf = new ImagingBuffer("RGB", w, h);
        if (image.getRaster().getDataBuffer() instanceof DataBufferInt dbi
                && (type == BufferedImage.TYPE_INT_RGB || type == BufferedImage.TYPE_INT_ARGB
                || type == BufferedImage.TYPE_INT_BGR)) {
            int[] pixels = dbi.getData();
            for (int i = 0; i < w * h; i++) {
                int p = pixels[i];
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;
                int off = i * 3;
                buf.data[off] = (byte) r;
                buf.data[off + 1] = (byte) g;
                buf.data[off + 2] = (byte) b;
            }
        } else {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int p = image.getRGB(x, y);
                    int r = (p >> 16) & 0xff;
                    int g = (p >> 8) & 0xff;
                    int b = p & 0xff;
                    int off = (y * w + x) * 3;
                    buf.data[off] = (byte) r;
                    buf.data[off + 1] = (byte) g;
                    buf.data[off + 2] = (byte) b;
                }
            }
        }
        return buf;
    }

    /**
     * Convert to {@link BufferedImage}. Does not depend on Tensor; pure AWT.
     * For training tensors use {@link org.bytedeco.pytorch.vision.pillow.tensor.PillowTensors}.
     */
    public BufferedImage toBufferedImage() {
        if ("L".equals(modeInfo.mode()) || "1".equals(modeInfo.mode())) {
            BufferedImage gray = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            byte[] pixels = ((DataBufferByte) gray.getRaster().getDataBuffer()).getData();
            System.arraycopy(data, 0, pixels, 0, Math.min(data.length, pixels.length));
            return gray;
        }
        if ("RGBA".equals(modeInfo.mode()) || "RGBa".equals(modeInfo.mode())) {
            BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int off = (y * width + x) * 4;
                    int r = data[off] & 0xff;
                    int g = data[off + 1] & 0xff;
                    int b = data[off + 2] & 0xff;
                    int a = data[off + 3] & 0xff;
                    img.setRGB(x, y, (a << 24) | (r << 16) | (g << 8) | b);
                }
            }
            return img;
        }
        if ("RGB".equals(modeInfo.mode()) || "YCbCr".equals(modeInfo.mode())
                || "LAB".equals(modeInfo.mode()) || "HSV".equals(modeInfo.mode())) {
            BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int off = (y * width + x) * modeInfo.bytesPerPixel();
                    int r = data[off] & 0xff;
                    int g = data[off + 1] & 0xff;
                    int b = data[off + 2] & 0xff;
                    img.setRGB(x, y, (r << 16) | (g << 8) | b);
                }
            }
            return img;
        }
        if ("P".equals(modeInfo.mode())) {
            // expand via palette if present else gray indices
            BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int idx = data[y * width + x] & 0xff;
                    int r, g, b;
                    if (palette != null && palette.length >= 768) {
                        r = palette[idx * 3] & 0xff;
                        g = palette[idx * 3 + 1] & 0xff;
                        b = palette[idx * 3 + 2] & 0xff;
                    } else {
                        r = g = b = idx;
                    }
                    img.setRGB(x, y, (r << 16) | (g << 8) | b);
                }
            }
            return img;
        }
        // fallback: treat first 3 bands or L
        ImagingBuffer rgb = ModeConvert.convert(this, "RGB");
        return rgb.toBufferedImage();
    }

    private void checkXY(int x, int y) {
        if (x < 0 || y < 0 || x >= width || y >= height) {
            throw new IndexOutOfBoundsException(
                    "pixel (" + x + "," + y + ") out of bounds " + width + "x" + height);
        }
    }

    public static int clamp8(int v) {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return v;
    }

    @Override
    public String toString() {
        return "ImagingBuffer(mode=" + mode() + " size=" + width + "x" + height + ")";
    }
}
