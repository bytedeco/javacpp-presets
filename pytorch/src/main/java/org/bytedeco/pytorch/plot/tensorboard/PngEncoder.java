package org.bytedeco.pytorch.plot.tensorboard;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.zip.CRC32;
import java.util.zip.Deflater;

/**
 * Minimal PNG encoder (no AWT / ImageIO required).
 * Supports grayscale (1), RGB (3), RGBA (4) 8-bit images in HWC layout.
 */
public final class PngEncoder {
    private PngEncoder() {}

    /**
     * @param hwc row-major pixels, length = height * width * channels
     * @param channels 1, 3, or 4
     */
    public static byte[] encodeHWC(byte[] hwc, int height, int width, int channels) {
        if (channels != 1 && channels != 3 && channels != 4) {
            throw new IllegalArgumentException("PNG channels must be 1, 3, or 4, got " + channels);
        }
        if (hwc == null || hwc.length < (long) height * width * channels) {
            throw new IllegalArgumentException("pixel buffer too small");
        }
        try {
            // filter 0 (None) per scanline
            byte[] raw = new byte[(width * channels + 1) * height];
            int src = 0;
            int dst = 0;
            int stride = width * channels;
            for (int y = 0; y < height; y++) {
                raw[dst++] = 0; // filter None
                System.arraycopy(hwc, src, raw, dst, stride);
                src += stride;
                dst += stride;
            }

            Deflater def = new Deflater(Deflater.DEFAULT_COMPRESSION);
            def.setInput(raw);
            def.finish();
            ByteArrayOutputStream idat = new ByteArrayOutputStream(raw.length / 2 + 64);
            byte[] tmp = new byte[8192];
            while (!def.finished()) {
                int n = def.deflate(tmp);
                idat.write(tmp, 0, n);
            }
            def.end();
            byte[] compressed = idat.toByteArray();

            ByteArrayOutputStream png = new ByteArrayOutputStream(compressed.length + 128);
            // signature
            png.write(new byte[]{(byte) 137, 80, 78, 71, 13, 10, 26, 10});
            writeChunk(png, "IHDR", ihdr(width, height, channels));
            writeChunk(png, "IDAT", compressed);
            writeChunk(png, "IEND", new byte[0]);
            return png.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException("PNG encode failed", e);
        }
    }

    /** float HWC in [0,1] or [0,255] → PNG. Values are auto-scaled by max. */
    public static byte[] encodeFloatHWC(float[] hwc, int height, int width, int channels) {
        byte[] u8 = new byte[height * width * channels];
        float max = 0f;
        for (float v : hwc) {
            float a = Math.abs(v);
            if (a > max) max = a;
        }
        float scale = max <= 1.0001f ? 255f : 1f;
        for (int i = 0; i < u8.length; i++) {
            float v = hwc[i] * scale;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            u8[i] = (byte) (int) (v + 0.5f);
        }
        return encodeHWC(u8, height, width, channels);
    }

    public static byte[] encodeDoubleHWC(double[] hwc, int height, int width, int channels) {
        float[] f = new float[hwc.length];
        for (int i = 0; i < hwc.length; i++) f[i] = (float) hwc[i];
        return encodeFloatHWC(f, height, width, channels);
    }

    /**
     * Build a sprite sheet (near-square grid) from N images of shape (C,H,W) or HWC batch.
     * Input: N images as flat HWC arrays of equal size; returns PNG bytes of the sheet
     * and single image (w,h).
     */
    public static Sprite makeSpriteHWC(byte[][] imagesHWC, int h, int w, int c) {
        int n = imagesHWC.length;
        int ncol = (int) Math.ceil(Math.sqrt(n));
        int nrow = (int) Math.ceil((double) n / ncol);
        // pad to square grid like pytorch embedding sprite
        int side = Math.max(ncol, nrow);
        ncol = side;
        nrow = side;
        byte[] sheet = new byte[nrow * h * ncol * w * c];
        // fill black
        Arrays.fill(sheet, (byte) 0);
        for (int i = 0; i < n; i++) {
            int row = i / ncol;
            int col = i % ncol;
            byte[] img = imagesHWC[i];
            for (int y = 0; y < h; y++) {
                int srcOff = y * w * c;
                int dstOff = ((row * h + y) * (ncol * w) + col * w) * c;
                System.arraycopy(img, srcOff, sheet, dstOff, w * c);
            }
        }
        byte[] png = encodeHWC(sheet, nrow * h, ncol * w, c == 1 ? 1 : 3);
        // force RGB for sprite if gray
        if (c == 1) {
            // re-encode as RGB
            byte[] rgb = new byte[nrow * h * ncol * w * 3];
            for (int i = 0, j = 0; i < sheet.length; i++) {
                byte g = sheet[i];
                rgb[j++] = g; rgb[j++] = g; rgb[j++] = g;
            }
            png = encodeHWC(rgb, nrow * h, ncol * w, 3);
        } else if (c == 4) {
            byte[] rgb = new byte[nrow * h * ncol * w * 3];
            for (int i = 0, j = 0; i < sheet.length; i += 4) {
                rgb[j++] = sheet[i];
                rgb[j++] = sheet[i + 1];
                rgb[j++] = sheet[i + 2];
            }
            png = encodeHWC(rgb, nrow * h, ncol * w, 3);
        }
        return new Sprite(png, w, h);
    }

    public record Sprite(byte[] png, int singleWidth, int singleHeight) {}

    private static byte[] ihdr(int width, int height, int channels) {
        byte[] b = new byte[13];
        putInt(b, 0, width);
        putInt(b, 4, height);
        b[8] = 8; // bit depth
        b[9] = (byte) (channels == 1 ? 0 : channels == 3 ? 2 : 6); // color type
        b[10] = 0; // compression
        b[11] = 0; // filter
        b[12] = 0; // interlace
        return b;
    }

    private static void putInt(byte[] b, int off, int v) {
        b[off] = (byte) (v >>> 24);
        b[off + 1] = (byte) (v >>> 16);
        b[off + 2] = (byte) (v >>> 8);
        b[off + 3] = (byte) v;
    }

    private static void writeChunk(ByteArrayOutputStream png, String type, byte[] data) throws IOException {
        byte[] typeBytes = type.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        putIntBAOS(png, data.length);
        png.write(typeBytes);
        png.write(data);
        CRC32 crc = new CRC32();
        crc.update(typeBytes);
        crc.update(data);
        putIntBAOS(png, (int) crc.getValue());
    }

    private static void putIntBAOS(ByteArrayOutputStream out, int v) {
        out.write((v >>> 24) & 0xff);
        out.write((v >>> 16) & 0xff);
        out.write((v >>> 8) & 0xff);
        out.write(v & 0xff);
    }
}
