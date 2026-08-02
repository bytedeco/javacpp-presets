package org.bytedeco.pytorch.vision.pillow;

import org.bytedeco.pytorch.vision.pillow.codec.CodecRegistry;
import org.bytedeco.pytorch.vision.pillow.core.DecompressionBomb;
import org.bytedeco.pytorch.vision.pillow.core.ImagingBuffer;
import org.bytedeco.pytorch.vision.pillow.core.ModeConvert;
import org.bytedeco.pytorch.vision.pillow.core.ModeInfo;
import org.bytedeco.pytorch.vision.pillow.core.PixelAccess;
import org.bytedeco.pytorch.vision.pillow.core.Resample;
import org.bytedeco.pytorch.vision.pillow.enums.Resampling;
import org.bytedeco.pytorch.vision.pillow.enums.Transpose;

import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Pillow {@code PIL.Image.Image} + module-level factories.
 *
 * <p>Snake_case names match upstream; camelCase aliases provided where useful.
 * {@code Image.new} → {@link #new_} / {@link #create}.
 */
public class Image implements Closeable, AutoCloseable {

    /** Mirrors Pillow {@code Image.MAX_IMAGE_PIXELS}. */
    public static long MAX_IMAGE_PIXELS = DecompressionBomb.MAX_IMAGE_PIXELS;

    private ImagingBuffer im;
    private String mode;
    private int width;
    private int height;
    private String format; // set by open/save plugins
    private final Map<String, Object> info = new LinkedHashMap<>();
    private ImagePalette palette;
    private boolean closed;
    private int frameIndex;
    private int nFrames = 1;

    protected Image(ImagingBuffer buffer) {
        Objects.requireNonNull(buffer, "buffer");
        this.im = buffer;
        this.mode = buffer.mode();
        this.width = buffer.width();
        this.height = buffer.height();
        if (buffer.getPalette() != null) {
            this.palette = new ImagePalette("RGB", buffer.getPalette());
        }
    }

    // ── module factories ───────────────────────────────────────────────────

    public static void preinit() {
        CodecRegistry.preinit();
    }

    public static void init() {
        CodecRegistry.init();
    }

    public static Image open(String path) throws IOException {
        return open(Path.of(path));
    }

    public static Image open(File file) throws IOException {
        return open(file.toPath());
    }

    public static Image open(Path path) throws IOException {
        return CodecRegistry.open(path);
    }

    public static Image open(InputStream in) throws IOException {
        return CodecRegistry.open(in);
    }

    public static Image open(byte[] data) throws IOException {
        return CodecRegistry.open(data);
    }

    /** Python {@code Image.new}; {@code new} is a Java keyword. */
    public static Image new_(String mode, int[] size) {
        Objects.requireNonNull(size, "size");
        if (size.length < 2) throw new IllegalArgumentException("size needs width,height");
        return new_(mode, size[0], size[1], null);
    }

    public static Image new_(String mode, int width, int height) {
        return new_(mode, width, height, null);
    }

    public static Image new_(String mode, int[] size, Object color) {
        Objects.requireNonNull(size, "size");
        return new_(mode, size[0], size[1], color);
    }

    public static Image new_(String mode, int width, int height, Object color) {
        ModeInfo.get(mode); // validate
        DecompressionBomb.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS;
        ImagingBuffer buf;
        if (color == null) {
            buf = ImagingBuffer.zeros(mode, width, height);
        } else {
            int[] bands = colorToBands(mode, color);
            buf = ImagingBuffer.filled(mode, width, height, bands);
        }
        return new Image(buf);
    }

    /** Alias for {@link #new_}. */
    public static Image create(String mode, int width, int height) {
        return new_(mode, width, height);
    }

    public static Image create(String mode, int width, int height, Object color) {
        return new_(mode, width, height, color);
    }

    public static Image frombytes(String mode, int[] size, byte[] data) {
        Objects.requireNonNull(size, "size");
        return frombytes(mode, size[0], size[1], data);
    }

    public static Image frombytes(String mode, int width, int height, byte[] data) {
        return new Image(ImagingBuffer.frombytes(mode, width, height, data));
    }

    public static Image frombuffer(String mode, int[] size, byte[] data) {
        // best-effort: copy (true shared buffer semantics limited in managed Java)
        return frombytes(mode, size, data);
    }

    public static Image fromarray(BufferedImage bi) {
        return new Image(ImagingBuffer.fromBufferedImage(bi));
    }

    public static Image fromBufferedImage(BufferedImage bi) {
        return fromarray(bi);
    }

    /** Package/plugin constructor. */
    public static Image fromBuffer(ImagingBuffer buffer) {
        return new Image(buffer);
    }

    public static Image fromBuffer(ImagingBuffer buffer, String format, Map<String, Object> info) {
        Image im = new Image(buffer);
        im.format = format;
        if (info != null) {
            im.info.putAll(info);
        }
        return im;
    }

    // ── state ──────────────────────────────────────────────────────────────

    public String mode() {
        ensureOpen();
        return mode;
    }

    public String getMode() {
        return mode();
    }

    public int width() {
        ensureOpen();
        return width;
    }

    public int getWidth() {
        return width();
    }

    public int height() {
        ensureOpen();
        return height;
    }

    public int getHeight() {
        return height();
    }

    public int[] size() {
        ensureOpen();
        return new int[]{width, height};
    }

    public String format() {
        return format;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public Map<String, Object> info() {
        return info;
    }

    public Map<String, Object> getInfo() {
        return info;
    }

    public ImagingBuffer getImagingBuffer() {
        ensureOpen();
        return im;
    }

    public ImagingBuffer imagingBuffer() {
        return getImagingBuffer();
    }

    public boolean isClosed() {
        return closed;
    }

    // ── load / close ───────────────────────────────────────────────────────

    public PixelAccess load() {
        ensureOpen();
        return new PixelAccess(im);
    }

    public Image verify() {
        ensureOpen();
        return this;
    }

    @Override
    public void close() {
        closed = true;
        im = null;
    }

    public Image copy() {
        ensureOpen();
        Image c = new Image(im.copy());
        c.format = format;
        c.info.putAll(info);
        if (palette != null) {
            c.palette = new ImagePalette(palette.mode(), palette.palette());
        }
        c.nFrames = nFrames;
        return c;
    }

    // ── pixels ─────────────────────────────────────────────────────────────

    public int[] getpixel(int x, int y) {
        ensureOpen();
        return im.getpixel(x, y);
    }

    public int[] getpixel(int[] xy) {
        return getpixel(xy[0], xy[1]);
    }

    public void putpixel(int x, int y, int[] color) {
        ensureOpen();
        im.putpixel(x, y, color);
    }

    public void putpixel(int x, int y, int gray) {
        putpixel(x, y, new int[]{gray});
    }

    public void putpixel(int[] xy, int[] color) {
        putpixel(xy[0], xy[1], color);
    }

    public int[] getdata() {
        ensureOpen();
        return im.getdata();
    }

    public void putdata(int[] flat) {
        ensureOpen();
        int b = im.bands();
        im.putdata(flat, 0, b);
    }

    public void putdata(int[] flat, int offset, int stride) {
        ensureOpen();
        im.putdata(flat, offset, stride);
    }

    public byte[] tobytes() {
        ensureOpen();
        return im.tobytes();
    }

    public byte[] tobitmap() {
        return tobytes();
    }

    // ── mode / bands ───────────────────────────────────────────────────────

    public String[] getbands() {
        ensureOpen();
        return ModeInfo.getmodebandnames(mode);
    }

    public Image convert(String newMode) {
        ensureOpen();
        if (mode.equals(newMode)) {
            return copy();
        }
        ImagingBuffer converted = ModeConvert.convert(im, newMode);
        Image out = new Image(converted);
        out.format = format;
        out.info.putAll(info);
        return out;
    }

    public Image[] split() {
        ensureOpen();
        String[] bands = getbands();
        Image[] out = new Image[bands.length];
        if (im.modeInfo().isByteMode()) {
            int b = bands.length;
            byte[] sd = im.data();
            for (int c = 0; c < b; c++) {
                ImagingBuffer ch = new ImagingBuffer("L", width, height);
                byte[] dd = ch.data();
                for (int i = 0, p = c; i < dd.length; i++, p += b) {
                    dd[i] = sd[p];
                }
                out[c] = new Image(ch);
            }
            return out;
        }
        for (int c = 0; c < bands.length; c++) {
            ImagingBuffer ch = new ImagingBuffer("L", width, height);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int[] p = im.getpixel(x, y);
                    ch.putpixel(x, y, new int[]{c < p.length ? p[c] : 0});
                }
            }
            out[c] = new Image(ch);
        }
        return out;
    }

    public Image getchannel(int channel) {
        Image[] ch = split();
        if (channel < 0 || channel >= ch.length) {
            throw new IllegalArgumentException("channel " + channel);
        }
        return ch[channel];
    }

    public Image getchannel(String name) {
        String[] bands = getbands();
        for (int i = 0; i < bands.length; i++) {
            if (bands[i].equalsIgnoreCase(name)) {
                return getchannel(i);
            }
        }
        throw new IllegalArgumentException("no channel " + name);
    }

    // ── geometry ───────────────────────────────────────────────────────────

    public Image crop(int[] box) {
        Objects.requireNonNull(box, "box");
        if (box.length < 4) throw new IllegalArgumentException("box needs 4 ints");
        return crop(box[0], box[1], box[2], box[3]);
    }

    public Image crop(int left, int upper, int right, int lower) {
        ensureOpen();
        return new Image(im.crop(left, upper, right, lower));
    }

    public Image resize(int[] size) {
        return resize(size, Resampling.BICUBIC);
    }

    public Image resize(int[] size, Resampling resample) {
        Objects.requireNonNull(size, "size");
        return resize(size[0], size[1], resample);
    }

    public Image resize(int w, int h) {
        return resize(w, h, Resampling.BICUBIC);
    }

    public Image resize(int w, int h, Resampling resample) {
        ensureOpen();
        Resampling r = resample == null ? Resampling.BICUBIC : resample;
        return new Image(Resample.resize(im, w, h, r));
    }

    public Image reduce(int factor) {
        return reduce(factor, factor);
    }

    public Image reduce(int fx, int fy) {
        ensureOpen();
        int nw = Math.max(1, width / Math.max(1, fx));
        int nh = Math.max(1, height / Math.max(1, fy));
        return resize(nw, nh, Resampling.BOX);
    }

    public Image thumbnail(int[] size) {
        return thumbnail(size, Resampling.BICUBIC);
    }

    public Image thumbnail(int[] size, Resampling resample) {
        ensureOpen();
        Objects.requireNonNull(size, "size");
        int tw = size[0], th = size[1];
        if (width <= tw && height <= th) {
            return this;
        }
        double wr = (double) tw / width;
        double hr = (double) th / height;
        double r = Math.min(wr, hr);
        int nw = Math.max(1, (int) Math.round(width * r));
        int nh = Math.max(1, (int) Math.round(height * r));
        ImagingBuffer resized = Resample.resize(im, nw, nh, resample == null ? Resampling.BICUBIC : resample);
        this.im = resized;
        this.width = nw;
        this.height = nh;
        return this;
    }

    public Image rotate(double degrees) {
        return rotate(degrees, Resampling.NEAREST, false, null);
    }

    public Image rotate(double degrees, Resampling resample, boolean expand) {
        return rotate(degrees, resample, expand, null);
    }

    public Image rotate(double degrees, Resampling resample, boolean expand, Object fillcolor) {
        ensureOpen();
        double a = ((degrees % 360) + 360) % 360;
        if (Math.abs(a) < 1e-9) {
            return copy();
        }
        // exact orthogonal rotates
        if (Math.abs(a - 90) < 1e-6) {
            return transpose(Transpose.ROTATE_90);
        }
        if (Math.abs(a - 180) < 1e-6) {
            return transpose(Transpose.ROTATE_180);
        }
        if (Math.abs(a - 270) < 1e-6) {
            return transpose(Transpose.ROTATE_270);
        }
        // general rotate about center
        double rad = Math.toRadians(-degrees); // Pillow: positive CCW; y-down → negate for screen
        double cos = Math.cos(rad);
        double sin = Math.sin(rad);
        int srcW = width, srcH = height;
        double cx = (srcW - 1) / 2.0;
        double cy = (srcH - 1) / 2.0;
        int outW = srcW, outH = srcH;
        if (expand) {
            double[] xs = {0, srcW - 1, 0, srcW - 1};
            double[] ys = {0, 0, srcH - 1, srcH - 1};
            double minX = Double.POSITIVE_INFINITY, maxX = Double.NEGATIVE_INFINITY;
            double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < 4; i++) {
                double dx = xs[i] - cx, dy = ys[i] - cy;
                double nx = cx + dx * cos - dy * sin;
                double ny = cy + dx * sin + dy * cos;
                minX = Math.min(minX, nx);
                maxX = Math.max(maxX, nx);
                minY = Math.min(minY, ny);
                maxY = Math.max(maxY, ny);
            }
            outW = (int) Math.ceil(maxX - minX + 1);
            outH = (int) Math.ceil(maxY - minY + 1);
            cx = (outW - 1) / 2.0;
            cy = (outH - 1) / 2.0;
            // remap using original center offset
            // simpler approach: inverse map from out to src with expanded canvas center
        }
        int[] fill = nullToBands(fillcolor);
        ImagingBuffer out = new ImagingBuffer(mode, outW, outH);
        if (fill != null) {
            out.fill(fill);
        }
        double ocx = (outW - 1) / 2.0;
        double ocy = (outH - 1) / 2.0;
        double scx = (srcW - 1) / 2.0;
        double scy = (srcH - 1) / 2.0;
        double invCos = Math.cos(Math.toRadians(degrees)); // inverse of display rotate
        double invSin = Math.sin(Math.toRadians(degrees));
        // Pillow positive angle is counter-clockwise for the image content.
        // Inverse mapping: src = R(-θ) * (dst - center) + srcCenter
        invCos = Math.cos(rad);
        invSin = Math.sin(-rad);
        // Actually rad = toRadians(-degrees) already; inverse of forward R(θ_screen):
        // forward screen: [cos -sin; sin cos] with rad=-deg for CCW content
        // inverse: rad_inv = -rad = +deg in screen space
        double ic = Math.cos(-rad);
        double is = Math.sin(-rad);
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                double dx = x - ocx;
                double dy = y - ocy;
                double sx = scx + dx * ic - dy * is;
                double sy = scy + dx * is + dy * ic;
                int ix = (int) Math.round(sx);
                int iy = (int) Math.round(sy);
                if (ix >= 0 && iy >= 0 && ix < srcW && iy < srcH) {
                    out.putpixel(x, y, im.getpixel(ix, iy));
                }
            }
        }
        return new Image(out);
    }

    public Image transpose(Transpose method) {
        ensureOpen();
        Objects.requireNonNull(method, "method");
        int w = width, h = height;
        ImagingBuffer out;
        switch (method) {
            case FLIP_LEFT_RIGHT -> {
                out = new ImagingBuffer(mode, w, h);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        out.putpixel(w - 1 - x, y, im.getpixel(x, y));
                    }
                }
            }
            case FLIP_TOP_BOTTOM -> {
                out = new ImagingBuffer(mode, w, h);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        out.putpixel(x, h - 1 - y, im.getpixel(x, y));
                    }
                }
            }
            case ROTATE_180 -> {
                out = new ImagingBuffer(mode, w, h);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        out.putpixel(w - 1 - x, h - 1 - y, im.getpixel(x, y));
                    }
                }
            }
            case ROTATE_90 -> {
                // CCW 90
                out = new ImagingBuffer(mode, h, w);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        out.putpixel(y, w - 1 - x, im.getpixel(x, y));
                    }
                }
            }
            case ROTATE_270 -> {
                out = new ImagingBuffer(mode, h, w);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        out.putpixel(h - 1 - y, x, im.getpixel(x, y));
                    }
                }
            }
            case TRANSPOSE -> {
                out = new ImagingBuffer(mode, h, w);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        out.putpixel(y, x, im.getpixel(x, y));
                    }
                }
            }
            case TRANSVERSE -> {
                out = new ImagingBuffer(mode, h, w);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        out.putpixel(h - 1 - y, w - 1 - x, im.getpixel(x, y));
                    }
                }
            }
            default -> throw new IllegalArgumentException("transpose " + method);
        }
        if (im.getPalette() != null) {
            out.setPalette(im.getPalette());
        }
        return new Image(out);
    }

    // ── paste / composite ──────────────────────────────────────────────────

    public void paste(Image src, int[] box) {
        Objects.requireNonNull(box, "box");
        paste(src, box[0], box[1], null);
    }

    public void paste(Image src, int x, int y) {
        paste(src, x, y, null);
    }

    public void paste(Image src, int x, int y, Image mask) {
        ensureOpen();
        Objects.requireNonNull(src, "src");
        ImagingBuffer srcBuf = src.im;
        if (!src.mode.equals(this.mode)) {
            srcBuf = ModeConvert.convert(src.im, this.mode);
        }
        ImagingBuffer maskBuf = mask == null ? null : mask.im;
        im.paste(srcBuf, x, y, 0, 0, srcBuf.width(), srcBuf.height(), maskBuf);
    }

    public void paste(Image src, int[] box, Image mask) {
        paste(src, box[0], box[1], mask);
    }

    public static Image alpha_composite(Image im1, Image im2) {
        return alphaComposite(im1, im2);
    }

    public static Image alphaComposite(Image im1, Image im2) {
        Objects.requireNonNull(im1, "im1");
        Objects.requireNonNull(im2, "im2");
        Image a = im1.mode().equals("RGBA") ? im1 : im1.convert("RGBA");
        Image b = im2.mode().equals("RGBA") ? im2 : im2.convert("RGBA");
        if (a.width() != b.width() || a.height() != b.height()) {
            throw new IllegalArgumentException("images must match size");
        }
        ImagingBuffer out = new ImagingBuffer("RGBA", a.width(), a.height());
        for (int y = 0; y < a.height(); y++) {
            for (int x = 0; x < a.width(); x++) {
                int[] p1 = a.getpixel(x, y);
                int[] p2 = b.getpixel(x, y);
                int a2 = p2[3];
                int a1 = p1[3];
                int outA = a2 + a1 * (255 - a2) / 255;
                int[] o = new int[4];
                if (outA == 0) {
                    out.putpixel(x, y, o);
                    continue;
                }
                for (int c = 0; c < 3; c++) {
                    o[c] = (p2[c] * a2 + p1[c] * a1 * (255 - a2) / 255) / outA;
                }
                o[3] = outA;
                out.putpixel(x, y, o);
            }
        }
        return new Image(out);
    }

    public Image point(java.util.function.IntUnaryOperator lut) {
        ensureOpen();
        Objects.requireNonNull(lut, "lut");
        ImagingBuffer out = im.copy();
        if (!im.modeInfo().isByteMode()) {
            throw new UnsupportedOperationException("point() on mode " + mode);
        }
        int b = im.bands();
        byte[] d = out.data();
        for (int i = 0; i < d.length; i++) {
            d[i] = (byte) ImagingBuffer.clamp8(lut.applyAsInt(d[i] & 0xff));
        }
        return new Image(out);
    }

    public Image point(int[] lut) {
        ensureOpen();
        Objects.requireNonNull(lut, "lut");
        return point(v -> {
            int idx = Math.max(0, Math.min(lut.length - 1, v));
            return lut[idx];
        });
    }

    // ── stats ──────────────────────────────────────────────────────────────

    public int[] getbbox() {
        ensureOpen();
        int b = im.bands();
        int minX = width, minY = height, maxX = -1, maxY = -1;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] p = im.getpixel(x, y);
                boolean nonzero = false;
                for (int c = 0; c < p.length; c++) {
                    if (p[c] != 0) {
                        nonzero = true;
                        break;
                    }
                }
                if (nonzero) {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < minX) return null;
        return new int[]{minX, minY, maxX + 1, maxY + 1};
    }

    public int[][] getextrema() {
        ensureOpen();
        int b = im.bands();
        int[] mins = new int[b];
        int[] maxs = new int[b];
        Arrays.fill(mins, Integer.MAX_VALUE);
        Arrays.fill(maxs, Integer.MIN_VALUE);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] p = im.getpixel(x, y);
                for (int c = 0; c < b; c++) {
                    mins[c] = Math.min(mins[c], p[c]);
                    maxs[c] = Math.max(maxs[c], p[c]);
                }
            }
        }
        if (b == 1) {
            return new int[][]{{mins[0], maxs[0]}};
        }
        int[][] out = new int[b][2];
        for (int c = 0; c < b; c++) {
            out[c][0] = mins[c];
            out[c][1] = maxs[c];
        }
        return out;
    }

    public int[] histogram() {
        ensureOpen();
        int b = Math.max(1, im.bands());
        int[] hist = new int[256 * b];
        if (im.modeInfo().isByteMode()) {
            byte[] d = im.data();
            for (int i = 0; i < d.length; i++) {
                int band = i % b;
                hist[band * 256 + (d[i] & 0xff)]++;
            }
        } else {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int[] p = im.getpixel(x, y);
                    for (int c = 0; c < b; c++) {
                        int v = Math.max(0, Math.min(255, p[c]));
                        hist[c * 256 + v]++;
                    }
                }
            }
        }
        return hist;
    }

    // ── palette ────────────────────────────────────────────────────────────

    public java.util.List<Integer> getpalette() {
        ensureOpen();
        byte[] p = im.getPalette();
        if (p == null && palette != null) {
            p = palette.palette();
        }
        if (p == null) return null;
        java.util.ArrayList<Integer> list = new java.util.ArrayList<>(p.length);
        for (byte b : p) {
            list.add(b & 0xff);
        }
        return list;
    }

    public void putpalette(byte[] data) {
        ensureOpen();
        im.setPalette(data);
        this.palette = new ImagePalette("RGB", data);
        if (!"P".equals(mode) && !"PA".equals(mode)) {
            // Pillow allows putpalette then mode may still be P after convert; keep data
        }
    }

    public void putpalette(int[] data) {
        byte[] b = new byte[data.length];
        for (int i = 0; i < data.length; i++) b[i] = (byte) data[i];
        putpalette(b);
    }

    public void putalpha(Image alpha) {
        ensureOpen();
        Objects.requireNonNull(alpha, "alpha");
        Image a = alpha.mode().equals("L") ? alpha : alpha.convert("L");
        if (!"RGBA".equals(mode)) {
            ImagingBuffer rgba = ModeConvert.convert(im, "RGBA");
            this.im = rgba;
            this.mode = "RGBA";
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] p = im.getpixel(x, y);
                int[] av = a.getpixel(Math.min(x, a.width() - 1), Math.min(y, a.height() - 1));
                p[3] = av[0];
                im.putpixel(x, y, p);
            }
        }
    }

    public void putalpha(int alpha) {
        ensureOpen();
        if (!"RGBA".equals(mode) && !"LA".equals(mode)) {
            ImagingBuffer converted = ModeConvert.convert(im, "RGBA".equals(ModeInfo.get(mode).basemode()) || mode.startsWith("RGB") ? "RGBA" : "LA");
            // prefer RGBA for RGB bases
            if ("RGB".equals(mode) || "P".equals(mode) || "YCbCr".equals(mode)) {
                converted = ModeConvert.convert(im, "RGBA");
                this.mode = "RGBA";
            } else if ("L".equals(mode) || "1".equals(mode)) {
                converted = ModeConvert.convert(im, "LA");
                this.mode = "LA";
            } else {
                converted = ModeConvert.convert(im, "RGBA");
                this.mode = "RGBA";
            }
            this.im = converted;
            this.width = im.width();
            this.height = im.height();
        }
        int b = im.bands();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int[] p = im.getpixel(x, y);
                p[b - 1] = alpha;
                im.putpixel(x, y, p);
            }
        }
    }

    // ── I/O ────────────────────────────────────────────────────────────────

    public void save(String path) throws IOException {
        save(Path.of(path), null, Map.of());
    }

    public void save(Path path) throws IOException {
        save(path, null, Map.of());
    }

    public void save(File file) throws IOException {
        save(file.toPath(), null, Map.of());
    }

    public void save(String path, String format) throws IOException {
        save(Path.of(path), format, Map.of());
    }

    public void save(Path path, String format, Map<String, Object> options) throws IOException {
        ensureOpen();
        Map<String, Object> opts = options == null ? new HashMap<>() : new HashMap<>(options);
        if (format != null) {
            opts.put("format", format);
        }
        CodecRegistry.save(this, path, opts);
    }

    public void save(OutputStream out, String format) throws IOException {
        save(out, format, Map.of());
    }

    public void save(OutputStream out, String format, Map<String, Object> options) throws IOException {
        ensureOpen();
        Objects.requireNonNull(format, "format");
        var plugin = CodecRegistry.forFormat(format);
        if (plugin == null) {
            plugin = CodecRegistry.forExtension(format);
        }
        if (plugin == null) {
            throw new IOException("no save handler for format " + format);
        }
        plugin.save(this, out, format, options == null ? Map.of() : options);
    }

    public void show() {
        ensureOpen();
        // best-effort: write temp PNG and open — may no-op headless
        try {
            Path tmp = java.nio.file.Files.createTempFile("pillow-show-", ".png");
            save(tmp, "PNG", Map.of());
            try {
                if (java.awt.Desktop.isDesktopSupported()) {
                    java.awt.Desktop.getDesktop().open(tmp.toFile());
                }
            } catch (Throwable ignored) {
            }
        } catch (IOException ignored) {
        }
    }

    public BufferedImage toBufferedImage() {
        ensureOpen();
        return im.toBufferedImage();
    }

    public int tell() {
        return frameIndex;
    }

    public void seek(int frame) throws IOException {
        if (frame != 0 && nFrames <= 1) {
            throw new EOFExceptionPillow("no more frames");
        }
        if (frame < 0 || frame >= nFrames) {
            throw new EOFExceptionPillow("frame " + frame + " out of range");
        }
        frameIndex = frame;
    }

    public void setNFrames(int n) {
        this.nFrames = Math.max(1, n);
    }

    public int n_frames() {
        return nFrames;
    }

    public int nFrames() {
        return nFrames;
    }

    public Image draft(String mode, int[] size) {
        // documented no-op / hint for decoders; stage-1 keeps pixels
        return this;
    }

    // ── helpers ────────────────────────────────────────────────────────────

    private void ensureOpen() {
        if (closed || im == null) {
            throw new IllegalStateException("image is closed");
        }
    }

    private static int[] colorToBands(String mode, Object color) {
        ModeInfo mi = ModeInfo.get(mode);
        int b = mi.bands();
        int[] out = new int[b];
        if (color instanceof Number n) {
            Arrays.fill(out, n.intValue());
            return out;
        }
        if (color instanceof int[] arr) {
            for (int i = 0; i < b; i++) {
                out[i] = i < arr.length ? arr[i] : 0;
            }
            return out;
        }
        if (color instanceof Integer[] arr) {
            for (int i = 0; i < b; i++) {
                out[i] = i < arr.length && arr[i] != null ? arr[i] : 0;
            }
            return out;
        }
        if (color instanceof String s) {
            // limited: #RRGGBB or gray name skip — parse hex
            String hex = s.startsWith("#") ? s.substring(1) : s;
            if (hex.length() == 6) {
                int rgb = Integer.parseInt(hex, 16);
                int r = (rgb >> 16) & 0xff, g = (rgb >> 8) & 0xff, bl = rgb & 0xff;
                if (b >= 3) {
                    out[0] = r;
                    out[1] = g;
                    out[2] = bl;
                    if (b > 3) out[3] = 255;
                } else {
                    out[0] = (r * 299 + g * 587 + bl * 114) / 1000;
                }
                return out;
            }
            if (hex.length() == 8) {
                long v = Long.parseLong(hex, 16);
                out[0] = (int) ((v >> 24) & 0xff);
                if (b > 1) out[1] = (int) ((v >> 16) & 0xff);
                if (b > 2) out[2] = (int) ((v >> 8) & 0xff);
                if (b > 3) out[3] = (int) (v & 0xff);
                return out;
            }
        }
        throw new IllegalArgumentException("unsupported color: " + color);
    }

    private static int[] nullToBands(Object fillcolor) {
        if (fillcolor == null) return null;
        if (fillcolor instanceof int[] a) return a;
        if (fillcolor instanceof Number n) return new int[]{n.intValue()};
        return null;
    }

    /** Local EOF for seek beyond frames (avoid java.io.EOFException checked noise in API). */
    public static final class EOFExceptionPillow extends IOException {
        public EOFExceptionPillow(String msg) {
            super(msg);
        }
    }

    @Override
    public String toString() {
        return "Image(mode=" + mode + " size=" + width + "x" + height
                + (format != null ? " format=" + format : "") + ")";
    }

    // re-export enums as nested-style constants for Pillow familiarity
    public static final Resampling NEAREST = Resampling.NEAREST;
    public static final Resampling BOX = Resampling.BOX;
    public static final Resampling BILINEAR = Resampling.BILINEAR;
    public static final Resampling HAMMING = Resampling.HAMMING;
    public static final Resampling BICUBIC = Resampling.BICUBIC;
    public static final Resampling LANCZOS = Resampling.LANCZOS;

    public static final Transpose FLIP_LEFT_RIGHT = Transpose.FLIP_LEFT_RIGHT;
    public static final Transpose FLIP_TOP_BOTTOM = Transpose.FLIP_TOP_BOTTOM;
    public static final Transpose ROTATE_90 = Transpose.ROTATE_90;
    public static final Transpose ROTATE_180 = Transpose.ROTATE_180;
    public static final Transpose ROTATE_270 = Transpose.ROTATE_270;
    public static final Transpose TRANSPOSE = Transpose.TRANSPOSE;
    public static final Transpose TRANSVERSE = Transpose.TRANSVERSE;
}
