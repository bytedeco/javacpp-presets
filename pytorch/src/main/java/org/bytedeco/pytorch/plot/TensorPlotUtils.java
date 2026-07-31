package org.bytedeco.pytorch.plot;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.LongHeaderOnlyArrayRef;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Conversion and layout helpers for multi-dimensional {@link Tensor} plotting.
 *
 * <p>Shape/rank helpers mirror {@code TensorBridge} semantics but are self-contained
 * so the plot package compiles independently. All tensors are moved to contiguous
 * CPU Double before extraction. Soft numel guard defaults to 16M.
 *
 * <h2>Rank policy (used by {@link Matplotlib} / {@link TensorPlot})</h2>
 * <ul>
 *   <li>0 — reject</li>
 *   <li>1 — line / hist / bar / box / violin</li>
 *   <li>2 — heatmap / imshow / multi-series plot / scatter (N,2)</li>
 *   <li>3 — imshow (C,H,W) or image grid (N,H,W)</li>
 *   <li>4 — image grid NCHW / NHWC</li>
 *   <li>≥5 — leading-dim slices into rank-4/3 path (capped)</li>
 * </ul>
 */
public final class TensorPlotUtils {
    /** Soft upper bound on elements accepted for plotting (override via {@link #setMaxNumel(long)}). */
    private static volatile long maxNumel = 16_000_000L;

    private TensorPlotUtils() {}

    // ---- layout -------------------------------------------------------------

    /**
     * Image tensor memory layout.
     * <ul>
     *   <li>{@link #AUTO} — heuristic detection</li>
     *   <li>{@link #HW} — (H, W) grayscale</li>
     *   <li>{@link #CHW} — (C, H, W)</li>
     *   <li>{@link #HWC} — (H, W, C)</li>
     *   <li>{@link #NCHW} — (N, C, H, W)</li>
     *   <li>{@link #NHWC} — (N, H, W, C)</li>
     *   <li>{@link #NHW} — (N, H, W) batch of grayscale</li>
     * </ul>
     */
    public enum Layout {
        AUTO, HW, CHW, HWC, NCHW, NHWC, NHW
    }

    /** One displayable plane: grayscale matrix and/or pre-rendered RGB image. */
    public static final class Plane {
        public final double[][] gray;       // may be null if rgb is set
        public final BufferedImage rgb;    // may be null if gray is set
        public final String label;

        public Plane(double[][] gray, String label) {
            this.gray = gray;
            this.rgb = null;
            this.label = label == null ? "" : label;
        }

        public Plane(BufferedImage rgb, String label) {
            this.gray = null;
            this.rgb = rgb;
            this.label = label == null ? "" : label;
        }

        public int height() {
            if (rgb != null) return rgb.getHeight();
            return gray == null || gray.length == 0 ? 0 : gray.length;
        }

        public int width() {
            if (rgb != null) return rgb.getWidth();
            return gray == null || gray.length == 0 || gray[0] == null ? 0 : gray[0].length;
        }
    }

    // ---- config -------------------------------------------------------------

    public static long maxNumel() { return maxNumel; }

    public static void setMaxNumel(long n) {
        if (n < 1) throw new IllegalArgumentException("maxNumel must be >= 1");
        maxNumel = n;
    }

    // ---- basic shape / convert ----------------------------------------------

    public static long[] shape(Tensor t) {
        requireNonNull(t);
        return sizesAsArray(t.sizes());
    }

    public static int rank(Tensor t) {
        return shape(t).length;
    }

    private static long[] sizesAsArray(LongHeaderOnlyArrayRef ref) {
        long len = ref.size();
        if (len == 0) return new long[0];
        return ref.vec().get();
    }

    public static void requireNonNull(Tensor t) {
        Objects.requireNonNull(t, "tensor");
    }

    public static void guardNumel(Tensor t) {
        requireNonNull(t);
        long n = t.numel();
        if (n > maxNumel) {
            throw new IllegalArgumentException(
                "tensor numel " + n + " exceeds plot limit " + maxNumel
                    + " (call TensorPlotUtils.setMaxNumel to raise)");
        }
    }

    public static void rejectScalar(Tensor t) {
        if (rank(t) == 0) {
            throw new IllegalArgumentException("scalar (rank-0) tensor cannot be plotted; reshape to 1D");
        }
    }

    /** Contiguous CPU Double tensor (caller owns lifetime of returned view chain). */
    public static Tensor toCpuDouble(Tensor t) {
        requireNonNull(t);
        guardNumel(t);
        return t.contiguous().to(ScalarType.Double).cpu();
    }

    /** Flatten any rank to 1-D double[] (existing Matplotlib hist/plot semantics). */
    public static double[] asDouble1D(Tensor t) {
        Tensor d = toCpuDouble(t);
        int n = (int) d.numel();
        double[] out = new double[n];
        if (n == 0) return out;
        DoublePointer ptr = d.data_ptr_double();
        ptr.get(out);
        return out;
    }

    /**
     * Rank 1 → 1×N row; rank 2 → rows×cols; higher ranks rejected
     * (use {@link #firstPlaneAsMatrix(Tensor)} or image APIs).
     */
    public static double[][] asMatrix2D(Tensor t) {
        Tensor d = toCpuDouble(t);
        long[] sh = shape(d);
        if (sh.length == 0) {
            throw new IllegalArgumentException("scalar tensor has no matrix view");
        }
        if (sh.length == 1) {
            double[] row = asDouble1D(d);
            return new double[][]{row};
        }
        if (sh.length == 2) {
            int rows = (int) sh[0];
            int cols = (int) sh[1];
            double[] flat = asDouble1D(d);
            double[][] m = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                System.arraycopy(flat, i * cols, m[i], 0, cols);
            }
            return m;
        }
        throw new IllegalArgumentException(
            "asMatrix2D requires rank 1 or 2, got rank " + sh.length + " shape " + Arrays.toString(sh)
                + "; use imshow/imageGrid for higher ranks");
    }

    /** First leading slice reduced until rank ≤ 2, returned as matrix. */
    public static double[][] firstPlaneAsMatrix(Tensor t) {
        Tensor cur = toCpuDouble(t);
        while (rank(cur) > 2) {
            cur = cur.select(0, 0);
        }
        return asMatrix2D(cur);
    }

    /** Index array 0..n-1. */
    public static double[] indexArray(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = i;
        return x;
    }

    /** Take {@code t[index, ...]} along dim 0. */
    public static Tensor sliceLeading(Tensor t, int index) {
        requireNonNull(t);
        long[] sh = shape(t);
        if (sh.length == 0) throw new IllegalArgumentException("cannot slice scalar");
        if (index < 0 || index >= sh[0]) {
            throw new IndexOutOfBoundsException("index " + index + " for dim0 size " + sh[0]);
        }
        return t.select(0, index);
    }

    // ---- normalize ----------------------------------------------------------

    public static double[] minMax(double[][] m) {
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        if (m != null) {
            for (double[] row : m) {
                if (row == null) continue;
                for (double v : row) {
                    if (Double.isNaN(v)) continue;
                    lo = Math.min(lo, v);
                    hi = Math.max(hi, v);
                }
            }
        }
        if (lo == Double.POSITIVE_INFINITY) { lo = 0; hi = 1; }
        if (hi <= lo) hi = lo + 1;
        return new double[]{lo, hi};
    }

    public static double[] minMax(double[] a) {
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        if (a != null) {
            for (double v : a) {
                if (Double.isNaN(v)) continue;
                lo = Math.min(lo, v);
                hi = Math.max(hi, v);
            }
        }
        if (lo == Double.POSITIVE_INFINITY) { lo = 0; hi = 1; }
        if (hi <= lo) hi = lo + 1;
        return new double[]{lo, hi};
    }

    /** In-place min-max normalize matrix to [0,1]. */
    public static double[][] normalize01(double[][] m) {
        double[] mm = minMax(m);
        return normalize01(m, mm[0], mm[1]);
    }

    public static double[][] normalize01(double[][] m, double lo, double hi) {
        if (m == null) return null;
        double span = hi - lo;
        if (span == 0) span = 1;
        double[][] out = new double[m.length][];
        for (int r = 0; r < m.length; r++) {
            if (m[r] == null) { out[r] = null; continue; }
            out[r] = new double[m[r].length];
            for (int c = 0; c < m[r].length; c++) {
                double v = m[r][c];
                out[r][c] = Double.isNaN(v) ? Double.NaN : (v - lo) / span;
            }
        }
        return out;
    }

    // ---- layout detection ---------------------------------------------------

    public static Layout detectLayout(Tensor t) {
        long[] sh = shape(t);
        int r = sh.length;
        if (r == 2) return Layout.HW;
        if (r == 3) {
            long d0 = sh[0], d2 = sh[2];
            boolean c0 = isChannelCount(d0);
            boolean cLast = isChannelCount(d2);
            if (c0 && !cLast) return Layout.CHW;
            if (cLast && !c0) return Layout.HWC;
            if (c0) return Layout.CHW; // prefer CHW when both look like channels
            return Layout.NHW;        // batch of grayscale
        }
        if (r == 4) {
            long d1 = sh[1], d3 = sh[3];
            boolean c1 = isChannelCount(d1);
            boolean cLast = isChannelCount(d3);
            if (c1 && !cLast) return Layout.NCHW;
            if (cLast && !c1) return Layout.NHWC;
            if (c1) return Layout.NCHW;
            // fallback: treat as NCHW-like batch
            return Layout.NCHW;
        }
        if (r == 1) return Layout.HW; // degenerate; callers should not imshow 1D
        // rank ≥ 5: peel leading dims later
        return Layout.NCHW;
    }

    private static boolean isChannelCount(long c) {
        return c == 1 || c == 3 || c == 4;
    }

    public static Layout resolveLayout(Tensor t, Layout layout) {
        if (layout == null || layout == Layout.AUTO) return detectLayout(t);
        return layout;
    }

    // ---- image / plane extraction -------------------------------------------

    /**
     * Extract display planes from a tensor for imshow / imageGrid.
     *
     * @param maxImages cap on leading batch size (default used by callers: 16)
     */
    public static List<Plane> extractPlanes(Tensor t, Layout layout, int maxImages) {
        requireNonNull(t);
        rejectScalar(t);
        guardNumel(t);
        if (maxImages < 1) maxImages = 1;

        Tensor d = toCpuDouble(t);
        long[] sh = shape(d);
        int r = sh.length;

        // Peel leading dims for rank ≥ 5 into a virtual batch along dim0 of the remainder.
        if (r >= 5) {
            return extractHighRank(d, maxImages);
        }

        Layout L = resolveLayout(d, layout);

        switch (L) {
            case HW: {
                if (r != 2) {
                    // allow rank1 as 1×N
                    if (r == 1) return List.of(new Plane(asMatrix2D(d), "0"));
                    // fall through: take first plane
                    return List.of(new Plane(firstPlaneAsMatrix(d), "0"));
                }
                return List.of(new Plane(asMatrix2D(d), "0"));
            }
            case CHW: {
                if (r != 3) throw layoutError(L, sh);
                return List.of(planeFromCHW(d, "0"));
            }
            case HWC: {
                if (r != 3) throw layoutError(L, sh);
                return List.of(planeFromHWC(d, "0"));
            }
            case NHW: {
                if (r != 3) throw layoutError(L, sh);
                int n = (int) Math.min(sh[0], maxImages);
                List<Plane> out = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    out.add(new Plane(asMatrix2D(d.select(0, i)), String.valueOf(i)));
                }
                return out;
            }
            case NCHW: {
                if (r == 3) {
                    // treat as single CHW
                    return List.of(planeFromCHW(d, "0"));
                }
                if (r != 4) throw layoutError(L, sh);
                int n = (int) Math.min(sh[0], maxImages);
                List<Plane> out = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    out.add(planeFromCHW(d.select(0, i), String.valueOf(i)));
                }
                return out;
            }
            case NHWC: {
                if (r == 3) {
                    return List.of(planeFromHWC(d, "0"));
                }
                if (r != 4) throw layoutError(L, sh);
                int n = (int) Math.min(sh[0], maxImages);
                List<Plane> out = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    out.add(planeFromHWC(d.select(0, i), String.valueOf(i)));
                }
                return out;
            }
            default:
                throw new IllegalArgumentException("unsupported layout " + L);
        }
    }

    private static List<Plane> extractHighRank(Tensor d, int maxImages) {
        long[] sh = shape(d);
        int n = (int) Math.min(sh[0], maxImages);
        List<Plane> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Tensor slice = d.select(0, i);
            // recurse with AUTO on remainder
            List<Plane> sub = extractPlanes(slice, Layout.AUTO, 1);
            if (!sub.isEmpty()) {
                Plane p = sub.get(0);
                String lab = i + (p.label.isEmpty() ? "" : "/" + p.label);
                if (p.rgb != null) out.add(new Plane(p.rgb, lab));
                else out.add(new Plane(p.gray, lab));
            }
        }
        return out;
    }

    private static IllegalArgumentException layoutError(Layout L, long[] sh) {
        return new IllegalArgumentException(
            "layout " + L + " incompatible with shape " + Arrays.toString(sh));
    }

    /** (C,H,W) → gray or RGB plane. C&gt;4 → first channel as gray (grid path uses channels separately). */
    public static Plane planeFromCHW(Tensor chw, String label) {
        long[] sh = shape(chw);
        if (sh.length != 3) throw new IllegalArgumentException("CHW expects rank 3, got " + Arrays.toString(sh));
        int c = (int) sh[0];
        if (c == 1) {
            return new Plane(asMatrix2D(chw.select(0, 0)), label);
        }
        if (c == 3 || c == 4) {
            return new Plane(rgbFromCHW(chw), label);
        }
        // too many channels: show channel 0 as gray; callers may imageGrid channels instead
        return new Plane(asMatrix2D(chw.select(0, 0)), label + "/c0");
    }

    public static Plane planeFromHWC(Tensor hwc, String label) {
        long[] sh = shape(hwc);
        if (sh.length != 3) throw new IllegalArgumentException("HWC expects rank 3, got " + Arrays.toString(sh));
        int c = (int) sh[2];
        if (c == 1) {
            // (H,W,1) → squeeze channel
            int h = (int) sh[0], w = (int) sh[1];
            double[] flat = asDouble1D(hwc);
            double[][] m = new double[h][w];
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) m[i][j] = flat[i * w + j]; // C=1 contiguous H,W,1
            }
            // Actually HWC C=1 layout is [h,w,1] so stride: flat[i*(w*1) + j*1 + 0]
            return new Plane(m, label);
        }
        if (c == 3 || c == 4) {
            return new Plane(rgbFromHWC(hwc), label);
        }
        // take channel 0
        int h = (int) sh[0], w = (int) sh[1];
        double[] flat = asDouble1D(hwc);
        double[][] m = new double[h][w];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
                m[i][j] = flat[(i * w + j) * c];
        return new Plane(m, label + "/c0");
    }

    /** Channel-major RGB(A) → BufferedImage. Values min-max normalized per-tensor to 0..255. */
    public static BufferedImage rgbFromCHW(Tensor chw) {
        long[] sh = shape(chw);
        int c = (int) sh[0], h = (int) sh[1], w = (int) sh[2];
        double[] flat = asDouble1D(chw);
        // flat layout: c, h, w contiguous → index ((ci*h + yi)*w + xi)
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (double v : flat) {
            if (Double.isNaN(v)) continue;
            lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
        if (lo == Double.POSITIVE_INFINITY) { lo = 0; hi = 1; }
        if (hi <= lo) hi = lo + 1;
        double span = hi - lo;

        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int ri = scale255(flat[(int) ((0L * h + y) * w + x)], lo, span);
                int gi = c > 1 ? scale255(flat[(int) ((1L * h + y) * w + x)], lo, span) : ri;
                int bi = c > 2 ? scale255(flat[(int) ((2L * h + y) * w + x)], lo, span) : ri;
                // alpha ignored for TYPE_INT_RGB
                img.setRGB(x, y, (ri << 16) | (gi << 8) | bi);
            }
        }
        return img;
    }

    public static BufferedImage rgbFromHWC(Tensor hwc) {
        long[] sh = shape(hwc);
        int h = (int) sh[0], w = (int) sh[1], c = (int) sh[2];
        double[] flat = asDouble1D(hwc);
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (double v : flat) {
            if (Double.isNaN(v)) continue;
            lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
        if (lo == Double.POSITIVE_INFINITY) { lo = 0; hi = 1; }
        if (hi <= lo) hi = lo + 1;
        double span = hi - lo;

        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int base = (y * w + x) * c;
                int ri = scale255(flat[base], lo, span);
                int gi = c > 1 ? scale255(flat[base + 1], lo, span) : ri;
                int bi = c > 2 ? scale255(flat[base + 2], lo, span) : ri;
                img.setRGB(x, y, (ri << 16) | (gi << 8) | bi);
            }
        }
        return img;
    }

    private static int scale255(double v, double lo, double span) {
        if (Double.isNaN(v)) return 0;
        double t = (v - lo) / span;
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        return (int) Math.round(t * 255.0);
    }

    /**
     * For rank-3 CHW with C&gt;4: return one gray plane per channel (capped).
     */
    public static List<Plane> channelPlanesCHW(Tensor chw, int maxChannels) {
        long[] sh = shape(chw);
        if (sh.length != 3) throw new IllegalArgumentException("expected CHW rank 3");
        int c = (int) Math.min(sh[0], maxChannels);
        List<Plane> out = new ArrayList<>(c);
        for (int i = 0; i < c; i++) {
            out.add(new Plane(asMatrix2D(chw.select(0, i)), "c" + i));
        }
        return out;
    }

    // ---- series helpers for rank-2 plot -------------------------------------

    /**
     * Treat rank-2 matrix rows as series (each row length = cols = x length).
     * Returns list of y arrays; x should be indexArray(cols) or provided separately.
     */
    public static List<double[]> rowsAsSeries(double[][] m) {
        List<double[]> series = new ArrayList<>();
        if (m == null) return series;
        for (double[] row : m) {
            if (row != null) series.add(row);
        }
        return series;
    }

    /**
     * Treat rank-2 matrix columns as series.
     */
    public static List<double[]> colsAsSeries(double[][] m) {
        List<double[]> series = new ArrayList<>();
        if (m == null || m.length == 0) return series;
        int rows = m.length;
        int cols = m[0].length;
        for (int c = 0; c < cols; c++) {
            double[] col = new double[rows];
            for (int r = 0; r < rows; r++) col[r] = m[r][c];
            series.add(col);
        }
        return series;
    }

    /** True if shape is (N,2) or (2,N) suitable for scatter pair. */
    public static boolean isScatterPairShape(long[] sh) {
        return sh != null && sh.length == 2
            && (sh[1] == 2 || sh[0] == 2);
    }

    public static double[][] scatterXY(Tensor t) {
        double[][] m = asMatrix2D(t);
        long[] sh = shape(t);
        if (sh[1] == 2) {
            // (N,2)
            int n = m.length;
            double[] x = new double[n], y = new double[n];
            for (int i = 0; i < n; i++) { x[i] = m[i][0]; y[i] = m[i][1]; }
            return new double[][]{x, y};
        }
        if (sh[0] == 2) {
            // (2,N)
            return new double[][]{m[0], m[1]};
        }
        throw new IllegalArgumentException(
            "scatter(Tensor) expects shape (N,2) or (2,N), got " + Arrays.toString(sh));
    }

    // ---- heat colormap (shared with ImageGridChart) -------------------------

    /** Blue → white → red, t in [0,1]. */
    public static Color divergingColor(float t) {
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        if (t < 0.5f) return lerp(new Color(0x21, 0x66, 0xac), Color.WHITE, t * 2f);
        return lerp(Color.WHITE, new Color(0xb2, 0x18, 0x2b), (t - 0.5f) * 2f);
    }

    /** Viridis-ish dark→bright sequential for images. */
    public static Color sequentialColor(float t) {
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        // simple dark-blue → cyan → yellow
        if (t < 0.5f) return lerp(new Color(0x0d, 0x08, 0x8a), new Color(0x00, 0xb0, 0xb0), t * 2f);
        return lerp(new Color(0x00, 0xb0, 0xb0), new Color(0xfb, 0xe5, 0x1e), (t - 0.5f) * 2f);
    }

    public static Color lerp(Color a, Color b, float t) {
        if (t < 0) t = 0;
        if (t > 1) t = 1;
        int r = (int) (a.getRed() + (b.getRed() - a.getRed()) * t);
        int g = (int) (a.getGreen() + (b.getGreen() - a.getGreen()) * t);
        int bl = (int) (a.getBlue() + (b.getBlue() - a.getBlue()) * t);
        return new Color(r, g, bl);
    }

    public static String shapeString(Tensor t) {
        return Arrays.toString(shape(t));
    }

    public static String layoutName(Layout L) {
        return L == null ? "null" : L.name().toLowerCase(Locale.ROOT);
    }
}
