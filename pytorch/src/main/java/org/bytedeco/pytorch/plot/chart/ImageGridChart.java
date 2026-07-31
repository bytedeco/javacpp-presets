package org.bytedeco.pytorch.plot.chart;

import org.bytedeco.pytorch.plot.TensorPlotUtils;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * Montage / image-grid chart for multi-slice and multi-channel tensors.
 *
 * <p>Each cell is either a grayscale {@code double[][]} (sequential colormap)
 * or a pre-rendered RGB {@link BufferedImage}. Pure AWT — headless-safe via
 * {@link #savefig(String)}.
 *
 * <pre>
 *   ImageGridChart g = Matplotlib.imageGrid(batchNCHW).setCols(4);
 *   g.setTitle("batch").savefig("/tmp/grid.png");
 * </pre>
 */
public final class ImageGridChart extends BaseChart {
    private final List<TensorPlotUtils.Plane> planes = new ArrayList<>();
    private int cols = 0; // 0 = auto square-ish
    private int gap = 4;
    private boolean normalize = true;
    private boolean showIndices = true;
    private boolean sequential = true; // sequential vs diverging for gray planes
    private Double normLo = null, normHi = null; // shared norm across gray planes when set

    public ImageGridChart(String title) {
        super(title == null ? "Image Grid" : title);
        this.width = 800;
        this.height = 600;
    }

    public ImageGridChart(String title, List<TensorPlotUtils.Plane> planes) {
        this(title);
        if (planes != null) this.planes.addAll(planes);
    }

    public ImageGridChart addPlane(TensorPlotUtils.Plane p) {
        if (p != null) planes.add(p);
        return this;
    }

    public ImageGridChart addGray(double[][] matrix, String label) {
        planes.add(new TensorPlotUtils.Plane(matrix, label));
        return this;
    }

    public ImageGridChart addRgb(BufferedImage img, String label) {
        planes.add(new TensorPlotUtils.Plane(img, label));
        return this;
    }

    /** Number of columns; 0 (default) picks ceil(sqrt(n)). */
    public ImageGridChart setCols(int cols) {
        this.cols = Math.max(0, cols);
        return this;
    }

    public ImageGridChart setGap(int gap) {
        this.gap = Math.max(0, gap);
        return this;
    }

    public ImageGridChart setNormalize(boolean v) {
        this.normalize = v;
        return this;
    }

    /** Shared value range for grayscale planes (disables per-plane min-max when both non-null). */
    public ImageGridChart setValueRange(double lo, double hi) {
        this.normLo = lo;
        this.normHi = hi;
        return this;
    }

    public ImageGridChart setShowIndices(boolean v) {
        this.showIndices = v;
        return this;
    }

    /** {@code true} sequential colormap; {@code false} diverging (blue-white-red). */
    public ImageGridChart setSequential(boolean v) {
        this.sequential = v;
        return this;
    }

    public int planeCount() { return planes.size(); }

    public List<TensorPlotUtils.Plane> planes() { return List.copyOf(planes); }

    @Override public ImageGridChart setTitle(String t) { super.setTitle(t); return this; }
    @Override public ImageGridChart setSize(int w, int h) { super.setSize(w, h); return this; }
    @Override public ImageGridChart setXAxisLabel(String l) { super.setXAxisLabel(l); return this; }
    @Override public ImageGridChart setYAxisLabel(String l) { super.setYAxisLabel(l); return this; }

    @Override
    public BufferedImage render() {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(background != null ? background : Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

        int n = planes.size();
        int topPad = (title != null && !title.isEmpty()) ? 36 : 12;
        int bottomPad = 12;
        int sidePad = 12;

        if (title != null && !title.isEmpty()) {
            g.setColor(Color.BLACK);
            g.setFont(new Font("SansSerif", Font.BOLD, 16));
            FontMetrics fm = g.getFontMetrics();
            g.drawString(title, Math.max(sidePad, (width - fm.stringWidth(title)) / 2), 24);
        }

        if (n == 0) {
            g.setColor(Color.DARK_GRAY);
            g.setFont(new Font("SansSerif", Font.PLAIN, 14));
            g.drawString("(empty grid)", sidePad, topPad + 20);
            g.dispose();
            return img;
        }

        int nCols = cols > 0 ? cols : (int) Math.ceil(Math.sqrt(n));
        nCols = Math.max(1, Math.min(nCols, n));
        int nRows = (n + nCols - 1) / nCols;

        int availW = width - 2 * sidePad - gap * (nCols - 1);
        int availH = height - topPad - bottomPad - gap * (nRows - 1);
        int cellW = Math.max(1, availW / nCols);
        int cellH = Math.max(1, availH / nRows);

        // Shared gray range if requested
        double glo = normLo != null ? normLo : Double.POSITIVE_INFINITY;
        double ghi = normHi != null ? normHi : Double.NEGATIVE_INFINITY;
        if (normalize && normLo == null) {
            for (TensorPlotUtils.Plane p : planes) {
                if (p.gray == null) continue;
                double[] mm = TensorPlotUtils.minMax(p.gray);
                glo = Math.min(glo, mm[0]);
                ghi = Math.max(ghi, mm[1]);
            }
            if (glo == Double.POSITIVE_INFINITY) { glo = 0; ghi = 1; }
            if (ghi <= glo) ghi = glo + 1;
        }

        g.setFont(new Font("SansSerif", Font.PLAIN, 11));
        for (int i = 0; i < n; i++) {
            int row = i / nCols;
            int col = i % nCols;
            int x0 = sidePad + col * (cellW + gap);
            int y0 = topPad + row * (cellH + gap);
            TensorPlotUtils.Plane p = planes.get(i);

            // cell background
            g.setColor(new Color(245, 245, 245));
            g.fillRect(x0, y0, cellW, cellH);
            g.setColor(new Color(200, 200, 200));
            g.drawRect(x0, y0, cellW, cellH);

            int labelH = showIndices ? 14 : 0;
            int innerW = cellW - 2;
            int innerH = cellH - 2 - labelH;
            int ix = x0 + 1;
            int iy = y0 + 1;

            if (p.rgb != null) {
                drawScaled(g, p.rgb, ix, iy, innerW, Math.max(1, innerH));
            } else if (p.gray != null) {
                BufferedImage tile = grayToImage(p.gray, glo, ghi, normalize || normLo != null);
                drawScaled(g, tile, ix, iy, innerW, Math.max(1, innerH));
            }

            if (showIndices) {
                String lab = p.label == null || p.label.isEmpty() ? String.valueOf(i) : p.label;
                g.setColor(Color.DARK_GRAY);
                FontMetrics fm = g.getFontMetrics();
                g.drawString(lab, x0 + 3, y0 + cellH - 3);
                // also tiny index top-left if label differs
                if (!lab.equals(String.valueOf(i))) {
                    g.drawString(String.valueOf(i), x0 + 3, y0 + fm.getAscent() + 1);
                }
            }
        }

        g.dispose();
        return img;
    }

    private BufferedImage grayToImage(double[][] m, double glo, double ghi, boolean doNorm) {
        int rows = m.length;
        int cols = rows == 0 || m[0] == null ? 0 : m[0].length;
        BufferedImage tile = new BufferedImage(Math.max(1, cols), Math.max(1, rows), BufferedImage.TYPE_INT_RGB);
        if (rows == 0 || cols == 0) return tile;

        double lo = glo, hi = ghi;
        if (!doNorm) {
            // raw: assume already roughly 0..1 or 0..255 — still map via minmax of this plane
            double[] mm = TensorPlotUtils.minMax(m);
            lo = mm[0]; hi = mm[1];
        } else if (normLo == null) {
            // use shared glo/ghi already computed, or per-plane if shared was empty
            if (lo == Double.POSITIVE_INFINITY) {
                double[] mm = TensorPlotUtils.minMax(m);
                lo = mm[0]; hi = mm[1];
            }
        }
        double span = hi - lo;
        if (span == 0) span = 1;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double v = m[r][c];
                float t = Double.isNaN(v) ? 0f : (float) ((v - lo) / span);
                Color color = sequential
                    ? TensorPlotUtils.sequentialColor(t)
                    : TensorPlotUtils.divergingColor(t);
                tile.setRGB(c, r, color.getRGB());
            }
        }
        return tile;
    }

    private static void drawScaled(Graphics2D g, BufferedImage src, int x, int y, int w, int h) {
        if (src == null || w <= 0 || h <= 0) return;
        // letterbox to preserve aspect
        int sw = src.getWidth(), sh = src.getHeight();
        if (sw <= 0 || sh <= 0) return;
        double scale = Math.min(w / (double) sw, h / (double) sh);
        int dw = Math.max(1, (int) Math.round(sw * scale));
        int dh = Math.max(1, (int) Math.round(sh * scale));
        int ox = x + (w - dw) / 2;
        int oy = y + (h - dh) / 2;
        g.drawImage(src, ox, oy, dw, dh, null);
    }
}
